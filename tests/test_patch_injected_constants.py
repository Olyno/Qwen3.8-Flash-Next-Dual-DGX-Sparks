"""Injected-block hygiene for the v030 text patches.

A patch script keeps its injected code in string literals. Anything the injected
code references as a module-level constant must be defined *inside* that text —
the host module never defines the patch's own names. This test exists because a
refactor moved one such constant (`_DRAFT_HEAD_FP8_ENV`) from the injected block
to the script's module scope; every unit test still passed and the container
failed at boot with `NameError: name '_DRAFT_HEAD_FP8_ENV' is not defined`.
"""
import ast
import importlib.util
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# names the host vllm module provides for the injected fragments
HOST_NAMES = {
    "os", "torch", "nn", "logger", "get_pp_group", "get_tp_group",
    "get_tensor_model_parallel_world_size", "tensor_model_parallel_all_gather",
    "ValueError", "Exception", "int", "str", "tuple", "bool", "float", "len",
    "getattr", "setattr", "range", "enumerate", "isinstance", "type", "min",
    "max", "sum", "abs", "print", "super", "bytes", "list", "dict", "set",
}


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def injected_blocks():
    """(label, text) for every fragment a patch script writes into vllm."""
    out = []
    draft = load(REPO / "patches" / "patch_mtp_draft_vocab_v030.py")
    out.append(("patch_mtp_draft_vocab_v030:FP8_HEAD_KERNELS",
                draft.FP8_HEAD_KERNELS))
    out.append(("patch_mtp_draft_vocab_v030:FP8_DRAFT_BLOCK",
                draft.FP8_HEAD_KERNELS + draft.FP8_DRAFT_BLOCK))
    # GTT_FP8 is a partial region (it ends mid-branch by design), so it cannot
    # be parsed standalone; the patch script checks the assembled file instead.
    target = REPO / "patches" / "patch_target_head_fp8_v030.py"
    if target.exists():
        mod = load(target)
        out.append(("patch_target_head_fp8_v030:block",
                    mod.FP8_HEAD_KERNELS + getattr(mod, "TARGET_BLOCK", "")))
    return out


def parse_fragment(text):
    """Parse a block that may be a module fragment (an indented function body)."""
    try:
        return ast.parse(text)
    except IndentationError:
        return ast.parse("def _fragment():\n" + textwrap.indent(
            textwrap.dedent(text), "    "))


def local_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


@pytest.mark.parametrize("label,text", injected_blocks())
def test_injected_constants_are_defined_in_the_block(label, text):
    tree = parse_fragment(text)
    defined = local_names(tree) | HOST_NAMES
    missing = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            # constants and private helpers are what patches commonly forget
            if name.startswith("_") or name.isupper():
                if name not in defined:
                    missing.add(name)
    assert not missing, (
        f"{label} references names that are not defined inside the injected "
        f"text and do not exist in the host module: {sorted(missing)}")
