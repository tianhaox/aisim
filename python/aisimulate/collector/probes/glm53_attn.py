# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scratch probe (exploration branch only): one GLM-5.3-Flash attention attempt in a pipeline job.

Arguments: ``checkpoint=fp8|nvfp4``, ``tp=<N>``, ``sets=<set id>+<set id>...``, ``util=<gpu
memory utilization>``, ``cls=regular|long``, optional ``kv=fp8|bfloat16`` (default: the SM's
serving identity) and ``fetch=0`` (only report the checkpoint state).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def run(args: dict[str, str], out_dir: Path) -> dict:
    from collector.helper import get_sm_version
    from collector.vllm import collect_glm53_attention as glm

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
    name = f"{checkpoint}-tp{tp_size}-{context_class}-probe-u{int(float(args.get('util', '0.85')) * 100)}"
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
