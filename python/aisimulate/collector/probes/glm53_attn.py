# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scratch probe (exploration branch only): one GLM-5.3-Flash attention attempt in a pipeline job.

Arguments: ``checkpoint=fp8|nvfp4``, ``tp=<N>``, ``sets=<set id>+<set id>...``, ``util=<gpu
memory utilization>``, ``cls=regular|long``, optional ``kv=fp8|bfloat16`` (default: the SM's
serving identity) and ``fetch=0`` (only report the checkpoint state).
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from pathlib import Path


def _du(label: str, out_dir: Path) -> None:
    """What the archived result directory holds (the pipeline snapshots it to a small volume)."""
    root = Path(out_dir).parent
    files = [(p.stat().st_size, str(p.relative_to(root))) for p in root.rglob("*") if p.is_file()]
    top = sorted(files, reverse=True)[:6]
    print("GLM53_DU " + json.dumps({"at": label, "bytes": sum(s for s, _ in files), "top": top}), flush=True)


def _node() -> None:
    """Which node this is and whether it has IMEX channels: NCCL_MNNVL_ENABLE=1 (the NVL72
    serving environment) fails communicator init on a node without a working channel."""
    imex = Path("/dev/nvidia-caps-imex-channels")
    report = {
        "host": platform.node(),
        "slurm": {k: v for k, v in os.environ.items() if k in ("SLURMD_NODENAME", "SLURM_JOB_NODELIST")},
        "imex_channels": sorted(p.name for p in imex.iterdir()) if imex.is_dir() else None,
    }
    print("GLM53_NODE " + json.dumps(report), flush=True)


def run(args: dict[str, str], out_dir: Path) -> dict:
    _node()
    from collector.helper import get_sm_version
    from collector.vllm import collect_glm53_attention as glm

    if args.get("rank0") == "1":
        # one GPU: rank 0 of the TP deployment alone (worker.emulate_rank0)
        os.environ["GLM53_ATTN_SINGLE_RANK"] = "1"
    if args.get("blocking") == "1":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    checkpoint = args["checkpoint"]
    tp_size = int(args["tp"])
    context_class = args.get("cls", "regular")
    sets = [s for s in args.get("sets", "").split("+") if s]
    system, gpus = glm.device_system()
    sm_version = get_sm_version()
    kv = args.get("kv") or glm.serving_kv_cache_dtype(sm_version)
    result: dict = {"system": system, "gpus": gpus, "sm": sm_version, "kv_cache_dtype": kv}
    started = time.time()
    model_path, fetched = glm.fetch_checkpoint(checkpoint)
    result["checkpoint"] = {**fetched, "path": str(model_path), "wall_seconds": round(time.time() - started, 1)}
    print("GLM53_FETCH " + json.dumps(result), flush=True)
    if args.get("fetch") == "0":
        return result
    _du("before", out_dir)

    def report() -> None:
        while True:
            time.sleep(600)
            _du("periodic", out_dir)

    threading.Thread(target=report, daemon=True).start()
    if args.get("mode") == "deploy":
        # every open set of the class, in passes over the utilization ladder
        with glm.exclusive(Path(out_dir)):
            summary = glm.run_deployment(
                output=Path(out_dir),
                system=system,
                checkpoint=checkpoint,
                tp_size=tp_size,
                kv_cache_dtype=kv,
                context_class=context_class,
                model_path=model_path,
            )
        result["deployment"] = summary
        _du("after-deployment", out_dir)
        result["failed"] = bool(summary["open_sets"])
        print("GLM53_DEPLOY_BEGIN " + json.dumps(result, default=str) + " GLM53_DEPLOY_END", flush=True)
        return result
    name = f"{checkpoint}-tp{tp_size}-{context_class}-probe-u{int(float(args.get('util', '0.85')) * 100)}"
    if args.get("rank0") == "1":
        name += "-rank0"
    with glm.exclusive(Path(out_dir)):
        state = glm.run_attempt(
            output=Path(out_dir),
            name=name,
            system=system,
            checkpoint=checkpoint,
            tp_size=tp_size,
            kv_cache_dtype=kv,
            context_class=context_class,
            only_sets=sets or None,
            utilization=float(args.get("util", "0.85")),
            model_path=model_path,
        )
    result["attempt"] = state
    attempt = Path(out_dir) / name
    log = (attempt / "runner.log").read_text(errors="replace")
    result["runner_log_tail"] = log[-6000:]
    if args.get("rank0") == "1":
        # per-target median of rank 0's measured repetitions (the contract needs every rank)
        import statistics
        from collections import defaultdict

        samples = defaultdict(list)
        raw = attempt / "raw" / "rank-0.jsonl"
        for line in raw.read_text().splitlines() if raw.exists() else []:
            record = json.loads(line)
            samples[record["target_id"]].append(record)
        result["rank0_targets"] = len(samples)
        result["rank0_record_keys"] = sorted(next(iter(samples.values()))[0]) if samples else []
    try:
        from collector.glm53flash_attention_contract import load_attempt, write_parquet

        _, rows = load_attempt(attempt, partial=not state["complete"])
        write_parquet(rows, attempt / "rows.parquet")
        result["rows"] = [
            [json.loads(r["geometry"])["is_context"], r["batch_size"], r["prefix"], r["x"], r["latency"]] for r in rows
        ]
        result["kernel_sources"] = sorted({r["kernel_source"] for r in rows})
        result["used_cuda_graph"] = sorted({bool(r["used_cuda_graph"]) for r in rows})
    except Exception as exc:
        result["rows_error"] = f"{type(exc).__name__}: {exc}"[:800]
    result["failed"] = not state["done"]
    print("GLM53_ATTEMPT_BEGIN " + json.dumps(result, default=str) + " GLM53_ATTEMPT_END", flush=True)
    return result
