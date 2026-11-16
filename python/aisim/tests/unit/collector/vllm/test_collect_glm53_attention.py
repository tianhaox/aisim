# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash attention collector inside a collection job: which sets a pass runs.

The runner stops an attempt at the first failing request set. The pipeline entry reads the
attempt directories to find the sets that are still open, skips a failed set for the rest
of its ``gpu_memory_utilization`` and retries it at the next one, so a resumed job
continues where the previous one stopped and a set that fits at no value is reported.
"""

import json
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.glm53flash_attention_contract import CHECKPOINTS, MACHINE_ARCH
from collector.glm53flash_attention_launch import freeze
from collector.vllm import collect_glm53_attention as glm

pytestmark = pytest.mark.unit

CONFIGS = Path(__file__).resolve().parents[4] / "src" / "aisim_core" / "model_configs"
NVFP4_CONFIG = CONFIGS / "nvidia--GLM-5.3-Flash-NVFP4_config.json"


def _freeze(output: Path, name: str, only_sets: list[str], utilization: float) -> Path:
    args = SimpleNamespace(
        backend="vllm",
        checkpoint="nvfp4",
        tp=2,
        system="b300_sxm",
        systems_dir=None,
        arch=MACHINE_ARCH[platform.machine()],
        gpus_per_node=None,
        kv_cache_dtype="fp8",
        config=str(NVFP4_CONFIG),
        sweep=str(glm.SWEEP),
        context_class="regular",
        attempt=str(output / name),
        source_commit="test",
        allocator_max_split_mb=None,
        sglang_mem_fraction=None,
        only_sets=only_sets,
        skip_sets=None,
        vllm_gpu_memory_utilization=utilization,
    )
    return freeze(args)[0]


def _record(attempt: Path, passed: list[str], failed: str | None = None, complete: bool = False) -> None:
    manifest = json.loads((attempt / "manifest.json").read_text())
    targets = {s["set_id"]: len(s["targets"]) for s in manifest["plan"]["sets"]}
    raw = attempt / "raw"
    raw.mkdir()
    lines = [{"set_id": s, "status": "passed"} for s in passed for _ in range(targets[s])]
    if failed:
        lines.append(
            {"set_id": failed, "status": "failed", "error_type": "RuntimeError", "error": "CUDA out of memory"}
        )
    (raw / "progress.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    if complete:
        (raw / "COMPLETE").write_text("done\n")


def test_an_attempt_reports_the_sets_it_finished_and_the_set_it_failed(tmp_path):
    sets = ["decode-b3", "prefill-b1-q256-c1", "prefill-b4-q256-c1"]
    attempt = _freeze(tmp_path, "nvfp4-tp2-regular-00-u85", sets, 0.85)
    _record(attempt, ["prefill-b1-q256-c1"], failed="decode-b3")
    state = glm.attempt_state(attempt)
    assert state["done"] == ["prefill-b1-q256-c1"] and not state["complete"]
    assert list(state["failed"]) == ["decode-b3"] and "out of memory" in state["failed"]["decode-b3"]
    assert state["utilization"] == 0.85 and sorted(state["selected"]) == sets


def test_a_set_with_only_some_targets_measured_stays_open(tmp_path):
    attempt = _freeze(tmp_path, "nvfp4-tp2-regular-00-u85", ["decode-b3"], 0.85)
    raw = attempt / "raw"
    raw.mkdir()
    (raw / "progress.jsonl").write_text(json.dumps({"set_id": "decode-b3", "status": "passed"}) + "\n")
    assert glm.attempt_state(attempt)["done"] == []
    remaining, failures = glm.open_sets(tmp_path, "nvfp4", 2, "regular")
    assert "decode-b3" in remaining and failures == {}


def test_a_deployment_skips_a_failed_set_and_retries_it_at_the_next_utilization(tmp_path, monkeypatch):
    """Two sets: one fails at 0.85 and passes at 0.7; the other fits at no value."""
    monkeypatch.setattr(glm, "class_sets", lambda context_class: ["decode-b3", "prefill-b1-q256-c1", "decode-b6"])
    calls = []

    def fake_attempt(*, output, name, only_sets, utilization, **kwargs):
        calls.append((utilization, list(only_sets)))
        attempt = _freeze(output, name, only_sets, utilization)
        fits = {"decode-b3": {0.7}, "prefill-b1-q256-c1": set(glm.UTILIZATION_LADDER), "decode-b6": set()}
        passed, failed = [], None
        for set_id in sorted(only_sets):  # the runner walks the plan in order and stops at a failure
            if utilization in fits[set_id]:
                passed.append(set_id)
            else:
                failed = set_id
                break
        _record(attempt, passed, failed=failed, complete=failed is None)
        return {**glm.attempt_state(attempt), "returncode": 0 if failed is None else 1, "seconds": 1.0}

    monkeypatch.setattr(glm, "run_attempt", fake_attempt)
    summary = glm.run_deployment(
        output=tmp_path,
        system="b300_sxm",
        checkpoint="nvfp4",
        tp_size=2,
        kv_cache_dtype="fp8",
        context_class="regular",
        model_path=tmp_path,
    )
    assert summary["open_sets"] == ["decode-b6"]
    assert sorted(summary["failures"]["decode-b6"]) == sorted(str(u) for u in glm.UTILIZATION_LADDER)
    # a failed set is not selected again at the same utilization
    for set_id in ("decode-b3", "decode-b6"):
        tried = [u for u, sets in calls if sets and sorted(sets)[0] == set_id]
        assert len(tried) == len(set(tried))
    # the runner stops at the first failure, so the set behind two failing ones runs in the third pass
    assert calls[0] == (0.85, ["decode-b3", "prefill-b1-q256-c1", "decode-b6"])
    assert calls[2] == (0.85, ["prefill-b1-q256-c1"])
    # and a finished set is never selected again
    assert all("prefill-b1-q256-c1" not in sets for _, sets in calls[3:])
    assert all("decode-b3" not in sets for u, sets in calls if u not in (0.85, 0.7))
    saved = json.loads((tmp_path / "nvfp4-tp2-regular.summary.json").read_text())
    assert saved["open_sets"] == ["decode-b6"] and saved["deployment"] == "nvfp4-tp2"


def test_a_resumed_deployment_reads_what_earlier_attempts_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(glm, "class_sets", lambda context_class: ["decode-b3", "prefill-b1-q256-c1"])
    earlier = _freeze(tmp_path, "nvfp4-tp2-regular-00-u85", ["decode-b3", "prefill-b1-q256-c1"], 0.85)
    _record(earlier, ["decode-b3", "prefill-b1-q256-c1"], complete=True)
    monkeypatch.setattr(glm, "run_attempt", lambda **kwargs: pytest.fail("nothing is open"))
    summary = glm.run_deployment(
        output=tmp_path,
        system="b300_sxm",
        checkpoint="nvfp4",
        tp_size=2,
        kv_cache_dtype="fp8",
        context_class="regular",
        model_path=tmp_path,
    )
    assert summary["open_sets"] == [] and summary["passes"] == []


def test_a_staged_checkpoint_is_used_only_with_the_pinned_config(tmp_path, monkeypatch):
    monkeypatch.setenv(glm.CHECKPOINT_ROOT_ENV, str(tmp_path))
    staged = tmp_path / "nvfp4"
    staged.mkdir()
    (staged / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="does not hold config.json"):
        glm.fetch_checkpoint("nvfp4")
    (staged / "config.json").write_text(NVFP4_CONFIG.read_text())
    path, info = glm.fetch_checkpoint("nvfp4")
    assert path == staged and info["staged"] and info["revision"] == CHECKPOINTS["nvfp4"][1]


def test_the_kv_identity_follows_the_sm():
    assert glm.serving_kv_cache_dtype(90) == "bfloat16"
    assert glm.serving_kv_cache_dtype(100) == glm.serving_kv_cache_dtype(103) == "fp8"
