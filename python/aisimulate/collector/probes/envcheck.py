# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scratch probe (exploration branch only): what a collection job can reach and hold.

Answers, for a real-checkpoint collector: can the container download from the Hugging Face
Hub and how fast, how much disk the job volume has, which GPUs it sees, and whether the
image ships what the GLM-5.3-Flash model needs. Arguments: ``repo=<org/name>`` (several
separated by ``+``), ``mib=<download sample size>``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def _try(fn):
    try:
        return fn()
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {exc}"[:300]


def _get(url: str, headers: dict | None = None, proxy: bool = True, timeout: int = 60):
    opener = urllib.request.build_opener() if proxy else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers=headers or {})
    return opener.open(req, timeout=timeout)


def _hub(repo: str, mib: int, proxy: bool) -> dict:
    out: dict = {}
    t0 = time.time()
    with _get(f"https://huggingface.co/api/models/{repo}?blobs=true", proxy=proxy) as resp:
        info = json.loads(resp.read())
    out["api_seconds"] = round(time.time() - t0, 2)
    out["sha"] = info.get("sha")
    out["gated"] = info.get("gated")
    files = [(s["rfilename"], s.get("size") or 0) for s in info.get("siblings", [])]
    weights = [f for f in files if f[0].endswith(".safetensors")]
    out["files"] = len(files)
    out["safetensors"] = len(weights)
    out["safetensors_gib"] = round(sum(size for _, size in weights) / (1 << 30), 1)
    if weights and mib > 0:
        name = weights[0][0]
        want = mib << 20
        t1 = time.time()
        got = 0
        url = f"https://huggingface.co/{repo}/resolve/{info.get('sha')}/{name}"
        with _get(url, headers={"Range": f"bytes=0-{want - 1}"}, proxy=proxy, timeout=120) as resp:
            out["download_status"] = resp.status
            while True:
                chunk = resp.read(8 << 20)
                if not chunk:
                    break
                got += len(chunk)
        seconds = time.time() - t1
        out["download_mib"] = round(got / (1 << 20), 1)
        out["download_seconds"] = round(seconds, 2)
        out["download_mib_per_s"] = round(got / (1 << 20) / max(seconds, 1e-6), 1)
    return out


def run(args: dict[str, str], out_dir: Path) -> dict:
    repos = [r for r in args.get("repo", "").split("+") if r]
    mib = int(args.get("mib", "512"))
    result: dict = {}
    if args.get("fill"):
        # does a large file beside the result directory break the pipeline's result snapshots?
        target = Path(args.get("where", "/mnt/ci_job")) / "envcheck-fill.bin"
        chunk = b"\0" * (64 << 20)
        started = time.time()
        with target.open("wb") as stream:
            for _ in range(int(args["fill"]) * 16):
                stream.write(chunk)
        result["fill"] = {"path": str(target), "gib": int(args["fill"]), "seconds": round(time.time() - started, 1)}
        (out_dir / "fill_marker_before_wait.txt").write_text(json.dumps(result["fill"]))
        time.sleep(int(args.get("wait", "240")))
        (out_dir / "fill_marker_after_wait.txt").write_text("done")
        print("ENVCHECK_FILL " + json.dumps(result["fill"]), flush=True)
        return result
    result["disk"] = {
        p: _try(lambda p=p: {k: round(v / (1 << 30), 1) for k, v in shutil.disk_usage(p)._asdict().items()})
        for p in ("/mnt/ci_job", "/tmp", "/dev/shm", str(Path.cwd()), "/root")
    }
    result["df"] = _try(lambda: subprocess.run(["df", "-h"], capture_output=True, text=True, timeout=30).stdout[-2500:])
    result["mnt_ci_job"] = _try(lambda: sorted(os.listdir("/mnt/ci_job"))[:40])
    result["cpu_count"] = os.cpu_count()
    result["meminfo"] = _try(lambda: Path("/proc/meminfo").read_text().splitlines()[:3])
    result["gpus"] = _try(
        lambda: subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        .stdout.strip()
        .splitlines()
    )
    secret = ("TOKEN", "KEY", "SECRET", "PASSWORD")
    result["env"] = {
        k: ("<set>" if any(s in k.upper() for s in secret) else v[:200])
        for k, v in sorted(os.environ.items())
        if k.upper().startswith(("HF_", "HUGGING", "NCCL", "VLLM", "CUDA_VISIBLE", "TRANSFORMERS"))
        or k.lower().endswith("_proxy")
    }

    def versions():
        import importlib.metadata as md

        found = {}
        for dist in (
            "vllm",
            "transformers",
            "torch",
            "flashinfer-python",
            "huggingface_hub",
            "hf_transfer",
            "safetensors",
        ):
            found[dist] = _try(lambda dist=dist: md.version(dist))
        return found

    result["versions"] = _try(versions)
    result["glm5next_text_config"] = _try(lambda: str(__import__("transformers").Glm5NextTextConfig))
    result["vllm_glm5next"] = _try(
        lambda: sorted(os.listdir(Path(__import__("vllm").__file__).parent / "models" / "glm5next"))
    )
    result["hub"] = {}
    for repo in repos:
        result["hub"][repo] = {
            "env_proxy": _try(lambda repo=repo: _hub(repo, mib, True)),
            "direct": _try(lambda repo=repo: _hub(repo, 0, False)),
        }
    print("ENVCHECK_JSON_BEGIN " + json.dumps(result, default=str) + " ENVCHECK_JSON_END", flush=True)
    return result
