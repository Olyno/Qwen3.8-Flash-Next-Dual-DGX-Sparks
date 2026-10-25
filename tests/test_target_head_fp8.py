"""Tests for patches/patch_target_head_fp8_v030.py.

The anchor fixture below is copied from the pinned v0.32.0
qwen4_exp/nvidia/model.py, so the text-patch tests run anywhere. The
quantization tests exec the block the patch injects; they skip when torch is
not installed. When engine/patches.sh has extracted the real image file, the
same patch runs against it too.
"""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PATCHES = REPO / "patches"
SCRIPT = PATCHES / "patch_target_head_fp8_v030.py"
REAL_ORIG = PATCHES / "v030_target_head" / "orig" / "model.py"

ENV = "VLLM_TARGET_HEAD_FP8"

MODEL_ORIG = '''\
from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


class Qwen4ExpModel(nn.Module):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpForCausalLM(
    nn.Module,
):
    def __init__(self, vllm_config) -> None:
        self.vllm_config = vllm_config
        self.lm_head = ParallelLMHead(248320, 2560)
        self.logits_processor = LogitsProcessor(248320)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)
'''


def load_patch():
    spec = importlib.util.spec_from_file_location(
        "patch_target_head_fp8_v030", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exec_defs(src, names):
    """Exec only the named module-level assignments / defs from `src`."""
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            if any(getattr(t, "id", "") in names for t in node.targets):
                keep.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(module), "<injected>", "exec"), ns)
    return ns


def fixture(tmp_path, text=MODEL_ORIG):
    work = tmp_path / "v030_target_head"
    (work / "orig").mkdir(parents=True)
    (work / "orig" / "model.py").write_text(text)
    return work


def run_patch(work):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(work)],
        capture_output=True,
        text=True,
    )


def test_target_head_is_full_vocabulary_and_inert_by_default():
    patch = load_patch()
    block = patch.TARGET_BLOCK
    # The attach quantizes the target's own head, never a draft slice.
    assert 'rows = getattr(getattr(model, "lm_head", None), "weight", None)' in block
    assert "_draft_lm_head_weight" not in block
    assert "no slice" in block
    # Off unless the env is set: the gate is read from the environment.
    assert ENV in block
    assert 'os.environ.get(_TARGET_HEAD_FP8_ENV, "")' in block
    # LogitsProcessor keeps its stock apply() -> all-gather -> trim path.
    assert "quant_method = _Fp8HeadQuantMethod(head)" in block
    assert "class _Fp8HeadQuantMethod" in block


def test_env_parsing():
    patch = load_patch()
    parse = exec_defs(patch.FP8_HEAD_KERNELS, {"_fp8_head_kernels"})[
        "_fp8_head_kernels"
    ]
    assert parse("", ENV) == ()
    assert parse("1", ENV) == ("marlin", "w8a8")
    assert parse("marlin", ENV) == ("marlin",)
    assert parse("w8a8", ENV) == ("w8a8",)
    with pytest.raises(ValueError, match=ENV):
        parse("banana", ENV)


def test_quantize_round_trip_and_scale():
    torch = pytest.importorskip("torch")
    patch = load_patch()
    ns = exec_defs(patch.FP8_HEAD_KERNELS, {"_FP8_MAX", "_quantize_rows_fp8"})
    quantize = ns["_quantize_rows_fp8"]

    w = torch.randn(4096, 256)
    w[0] *= 50.0  # a wide row
    w[1] = 0.0  # all-zero row: finite scale, zero output
    w[2] *= 1e-3  # a narrow row
    q, scale = quantize(w)

    assert q.shape == w.shape
    assert q.dtype == torch.float8_e4m3fn
    assert scale.shape == (4096,)
    assert scale.dtype == torch.float32
    assert scale[1] == 1.0

    deq = q.float() * scale[:, None]
    assert torch.isfinite(deq).all()
    assert torch.count_nonzero(deq[1]) == 0
    # E4M3 (3 mantissa bits) with a per-row scale: within half a step of the
    # row's amax, plus the bf16 rounding of the scale itself.
    row_err = (deq - w).abs().amax(dim=1)
    row_amax = w.abs().amax(dim=1)
    assert bool((row_err <= 0.07 * row_amax + 1e-6).all()), (row_err / row_amax).max()


def test_patch_is_idempotent_on_the_fixture(tmp_path):
    work = fixture(tmp_path)
    first = run_patch(work)
    assert first.returncode == 0, first.stderr
    out = (work / "model_v030.py").read_text()
    ast.parse(out)
    assert out.count("_attach_target_head_fp8(self)") == 1
    assert out.count("def _attach_target_head_fp8") == 1
    assert 'logger = init_logger(__name__)' in out
    assert f'"{ENV}"' in out

    second = run_patch(work)
    assert second.returncode == 0, second.stderr
    assert (work / "model_v030.py").read_text() == out

    # Feeding the patched file back in must refuse, not double-inject.
    (work / "orig" / "model.py").write_text(out)
    third = run_patch(work)
    assert third.returncode != 0
    assert "already patched" in third.stderr


def test_missing_anchor_fails_loudly(tmp_path):
    broken = MODEL_ORIG.replace(
        "from vllm.distributed import get_pp_group\n",
        "from vllm.distributed import get_pp_group as _pp\n",
    )
    work = fixture(tmp_path, broken)
    r = run_patch(work)
    assert r.returncode != 0
    assert "anchor" in r.stderr and "not unique/missing" in r.stderr
    assert not (work / "model_v030.py").exists()


@pytest.mark.skipif(
    not REAL_ORIG.is_file(),
    reason="no extracted v0.32 model.py; boot the prod recipe once (RECIPE=prod ./start.sh)",
)
def test_patch_applies_to_the_extracted_image_file(tmp_path):
    work = fixture(tmp_path, REAL_ORIG.read_text())
    r = run_patch(work)
    assert r.returncode == 0, r.stderr
    out = (work / "model_v030.py").read_text()
    ast.parse(out)
    # Applied to the language model only; the multimodal wrapper delegates.
    assert out.count("_attach_target_head_fp8(self)") == 1
