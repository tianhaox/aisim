# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The all-reduce collector sets SGLang's all-reduce switches as a default server does.

sglang v0.5.23 ``distributed/bootstrap.py::_set_all_reduce_flags`` sets four module switches
before the parallel groups are built. The collector imported ``set_mscclpp_all_reduce``, which
the release no longer has (it is ``set_mscclpp``), so every SGLang all-reduce run died at import."""

import ast
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector" / "network" / "collect_all_reduce.py"
_SWITCHES = ("set_custom_all_reduce", "set_mscclpp", "set_torch_symm_mem_all_reduce", "set_flashinfer_allreduce_only")


def _functions(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {"sys": sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def _fake_sglang(monkeypatch, state):
    """parallel_state with exactly the names sglang v0.5.23 exports for this collector."""
    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    comm = types.ModuleType("sglang.srt.distributed.communication_op")
    comm.tensor_model_parallel_all_reduce = lambda x: x
    par = types.ModuleType("sglang.srt.distributed.parallel_state")
    plain = ("destroy_model_parallel", "graph_capture", "init_distributed_environment", "initialize_model_parallel")
    for name in plain:
        setattr(par, name, lambda *a, **k: None)
    for name in _SWITCHES:
        setattr(par, name, lambda enable, name=name: state.__setitem__(name, enable))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.communication_op", comm)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state", par)


def test_the_collector_imports_the_switches_the_pinned_release_has(monkeypatch):
    state = {}
    _fake_sglang(monkeypatch, state)
    ns = _functions("import_sglang", "set_sglang_serving_all_reduce_flags")
    mods = ns["import_sglang"]()
    ns["set_sglang_serving_all_reduce_flags"](mods)
    assert state == {
        "set_custom_all_reduce": True,
        "set_mscclpp": False,
        "set_torch_symm_mem_all_reduce": False,
        "set_flashinfer_allreduce_only": False,
    }


def test_the_switches_are_set_before_the_groups_are_built():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    setup = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "setup_sglang_distributed")
    text = ast.unparse(setup)
    assert text.index("set_sglang_serving_all_reduce_flags(") < text.index("sglang_mods['initialize_model_parallel'](")
