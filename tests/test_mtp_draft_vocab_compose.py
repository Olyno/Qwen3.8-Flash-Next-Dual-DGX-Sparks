"""The FP8-dense draft head and the reduced draft vocab must compose.

Supported order, on both lanes: the FP8-dense overlay rewrites the draft
lm_head to a quantized head (per-row scale) first, then the draft-vocab patch
slices that file into its own output, leaving the overlay file untouched. The
reverse order (slice first, or feeding an already-sliced file back in) must be
refused, and the slice must carry a quantized head's per-row scale with the
rows. No torch, vLLM or image extract needed: the injected
``_attach_draft_vocab`` is lifted out of the composed output and run against a
tiny fake torch and a fake quantized head.
"""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
LEGACY_PATCHER = REPO / "patches" / "patch_mtp_draft_vocab.py"
V030_PATCHER = REPO / "patches" / "patch_mtp_draft_vocab_v030.py"

LM_HEAD_STOCK = '            prefix=maybe_prefix(prefix, "lm_head"),\n'
LM_HEAD_QUANT = ('            quant_config=self.quant_config,  # [fp8dense overlay]\n'
                 + LM_HEAD_STOCK)

# Stock qwen3_8_flash_next/nvidia/mtp.py, reduced to the anchors the legacy
# draft-vocab patch matches.
LEGACY_MTP = '''\
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group


def _remap_ignored_layers(
    names, start
):
    return [n for n in names if n]


class LegacyMTP(nn.Module):
    def __init__(self, config, vllm_config):
        super().__init__()
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        loader = make_loader()
        return loader.load_weights(remap_weight_names())
'''

# Stock qwen4_exp/nvidia/mtp.py, reduced to the anchors the v030 patch matches.
V030_MTP = '''\
import torch
from torch import nn

from vllm.distributed import get_pp_group


def _remap_ignored_layers(
    names, start
):
    return [n for n in names if n]


class Qwen4ExpMultiTokenPredictor(nn.Module):
    def __init__(self, config, vllm_config):
        super().__init__()
        self.quant_config = vllm_config.quant_config
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        loader = make_loader()
        return loader.load_weights(remap_weight_names(), mapper=mapper)
'''


# --------------------------------------------------------------------------
# Just enough of torch to run the injected slice.
# --------------------------------------------------------------------------
class _DType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "torch." + self.name

    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, _DType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class _Tensor:
    def __init__(self, data, dtype):
        self._values = [list(row) if isinstance(row, list) else row for row in data]
        self.dtype = dtype
        self.device = "cpu"

    @property
    def data(self):
        # Real torch exposes the storage as a tensor; .data.index_select works.
        return self

    @property
    def values(self):
        return self._values

    @property
    def shape(self):
        if self._values and isinstance(self._values[0], list):
            return (len(self._values), len(self._values[0]))
        return (len(self._values),)

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def element_size(self):
        return 1 if "float8" in self.dtype.name or self.dtype.name == "uint8" else 2

    def index_select(self, dim, index):
        assert dim == 0
        return _Tensor([self._values[i] for i in index.values], self.dtype)

    def to(self, dtype):
        return _Tensor(self._values, dtype)

    def unsqueeze(self, dim):
        assert dim == 1
        return _Tensor([[v] for v in self._values], self.dtype)

    def contiguous(self):
        return self

    def __mul__(self, other):
        # Only the [N, K] * [N, 1] broadcast the patch performs.
        if other.shape[1:] == (1,):
            return _Tensor([[v * other.values[i][0] for v in row]
                            for i, row in enumerate(self._values)], self.dtype)
        return _Tensor([[x * y for x, y in zip(ra, rb)]
                        for ra, rb in zip(self._values, other.values)], self.dtype)


class _FakeTorch:
    long = _DType("long")
    uint8 = _DType("uint8")
    int8 = _DType("int8")
    bfloat16 = _DType("bfloat16")
    float8_e4m3fn = _DType("float8_e4m3fn")

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return _Tensor(list(values), dtype)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, fmt, *args):
        self.messages.append(fmt % args if args else fmt)

    def warning(self, fmt, *args):
        self.messages.append(fmt % args if args else fmt)


class _FakeModel:
    def __init__(self, weight, scale):
        rows = weight.shape[0]
        self.lm_head = types.SimpleNamespace(
            weight=weight,
            weight_scale=scale,
            shard_indices=types.SimpleNamespace(
                org_vocab_start_index=0, org_vocab_end_index=rows),
            org_vocab_size=rows,
            tp_size=1,
        )
        self.logits_processor = types.SimpleNamespace(scale=1.0)

    def register_buffer(self, name, tensor, persistent=False):
        setattr(self, name, tensor)


def _attach_vocab_from(path):
    """Lift _attach_draft_vocab out of a composed mtp.py and exec it."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_attach_draft_vocab":
            src = ast.unparse(node)
            break
    else:
        raise AssertionError(f"_attach_draft_vocab not found in {path}")
    logger = _Logger()
    namespace = {
        "os": os,
        "nn": types.SimpleNamespace(Module=object),
        "torch": _FakeTorch(),
        "logger": logger,
    }
    exec(compile(src, str(path), "exec"), namespace)
    return namespace["_attach_draft_vocab"], logger


class ComposedDraftHead(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))

    def stage_fp8_dense_head(self, source):
        """What the FP8-dense overlay does to mtp.py: quantize the draft head."""
        self.assertIn(LM_HEAD_STOCK, source)
        return source.replace(LM_HEAD_STOCK, LM_HEAD_QUANT)

    def run_patcher(self, script, inp, out, *extra):
        return subprocess.run(
            [sys.executable, str(script), "--input", str(inp),
             "--output", str(out), *extra],
            capture_output=True, text=True,
        )

    def compose(self, name, stock, script):
        # Stage 1: FP8-dense head rewrite (the overlay's diff, same effect).
        stage1 = self.tmp / f"{name}_fp8dense_mtp.py"
        stage1.write_text(self.stage_fp8_dense_head(stock))
        # Stage 2: the slice, on that file, into a separate output.
        stage2 = self.tmp / f"{name}_patched_mtp.py"
        r = self.run_patcher(script, stage1, stage2)
        self.assertEqual(r.returncode, 0, r.stderr)
        return stage1, stage2

    def test_legacy_lane_composes(self):
        stage1, stage2 = self.compose("legacy", LEGACY_MTP, LEGACY_PATCHER)
        before = stage1.read_text()
        patched = stage2.read_text()
        ast.parse(patched)
        self.assertNotIn("_attach_draft_vocab", before)  # input not clobbered
        self.assertIn("quant_config=self.quant_config", before)
        self.assertIn("quant_config=self.quant_config", patched)
        self.assertEqual(patched.count("def _attach_draft_vocab"), 1)
        self.assertEqual(patched.count("def get_top_tokens"), 1)
        self.assertIn("_attach_draft_vocab(self)", patched)
        self.assertEqual(stage1.read_text(), before)

    def test_v030_lane_composes(self):
        stage1, stage2 = self.compose("v030", V030_MTP, V030_PATCHER)
        before = stage1.read_text()
        patched = stage2.read_text()
        ast.parse(patched)
        self.assertNotIn("_attach_draft_vocab", before)
        self.assertIn("quant_config=self.quant_config", patched)
        self.assertEqual(patched.count("def _attach_draft_vocab"), 1)
        self.assertIn("_attach_draft_vocab(self)", patched)

    def test_rerun_does_not_double_apply(self):
        for name, stock, script in (("legacy", LEGACY_MTP, LEGACY_PATCHER),
                                    ("v030", V030_MTP, V030_PATCHER)):
            with self.subTest(lane=name):
                stage1, stage2 = self.compose(name, stock, script)
                first = stage2.read_text()
                r = self.run_patcher(script, stage1, stage2)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(stage2.read_text(), first)
                self.assertEqual(first.count("def _attach_draft_vocab"), 1)

    def test_reverse_order_is_refused(self):
        for name, stock, script in (("legacy", LEGACY_MTP, LEGACY_PATCHER),
                                    ("v030", V030_MTP, V030_PATCHER)):
            with self.subTest(lane=name):
                _, sliced = self.compose(name, stock, script)
                again = self.tmp / f"{name}_again.py"
                r = self.run_patcher(script, sliced, again)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("already sliced", r.stderr)
                self.assertIn("fp8_dense", r.stderr)
                self.assertIn("supported order", r.stderr)
                self.assertFalse(again.exists())

    def test_missing_input_fails(self):
        r = self.run_patcher(LEGACY_PATCHER, self.tmp / "nope.py",
                             self.tmp / "out.py")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing", r.stderr)

    def test_slice_carries_the_scale(self):
        _, composed = self.compose("legacy", LEGACY_MTP, LEGACY_PATCHER)
        ids = self.tmp / "draft_vocab.txt"
        ids.write_text("0\n2\n")
        for dtype_name in ("uint8", "float8_e4m3fn"):
            with self.subTest(dtype=dtype_name):
                dtype = getattr(_FakeTorch, dtype_name)
                weight = _Tensor([[1.0, 2.0], [3.0, 4.0],
                                  [5.0, 6.0], [7.0, 8.0]], dtype)
                scale = _Tensor([0.5, 0.25, 0.125, 0.0625], _FakeTorch.bfloat16)
                attach, logger = _attach_vocab_from(composed)
                model = _FakeModel(weight, scale)
                with mock.patch.dict(os.environ,
                                     {"VLLM_MTP_DRAFT_VOCAB": str(ids)}):
                    attach(model)
                reduced = model._draft_lm_head_weight
                # rows 0 and 2, dequantized once in BF16 with their own scale
                self.assertEqual(reduced.dtype, _FakeTorch.bfloat16)
                self.assertEqual(reduced.values, [[0.5, 1.0], [0.625, 0.75]])
                self.assertEqual(model._draft_id_to_target_id.values, [0, 2])
                self.assertTrue(any("dequantized with their scale"
                                    in m for m in logger.messages))

    @unittest.skipUnless(
        (REPO / "overlays" / "fp8dense_v030" / "orig" / "mtp.py").is_file()
        and shutil.which("git"),
        "no v0.32 stock mtp.py extract; boot the fp8 recipe once",
    )
    def test_real_v030_stock_survives_the_fp8dense_diff(self):
        # The in-repo diff must leave every draft-vocab anchor intact, so the
        # chain works on the real extract and not just the synthetic fixture.
        # git apply -p2 targets the +++ path, i.e. ./mtp.py under cwd.
        stage1 = self.tmp / "mtp.py"
        shutil.copy(REPO / "overlays" / "fp8dense_v030" / "orig" / "mtp.py", stage1)
        r = subprocess.run(
            ["git", "apply", "-p2",
             str(REPO / "overlays" / "fp8dense_v030" / "mtp.diff")],
            cwd=self.tmp, capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("quant_config=self.quant_config", stage1.read_text())
        stage2 = self.tmp / "real_patched_mtp.py"
        r = self.run_patcher(V030_PATCHER, stage1, stage2)
        self.assertEqual(r.returncode, 0, r.stderr)
        patched = stage2.read_text()
        ast.parse(patched)
        self.assertEqual(patched.count("def _attach_draft_vocab"), 1)
        self.assertIn("quant_config=self.quant_config", patched)

    def test_quantized_head_without_usable_scale_is_refused(self):
        _, composed = self.compose("legacy", LEGACY_MTP, LEGACY_PATCHER)
        ids = self.tmp / "draft_vocab.txt"
        ids.write_text("0\n2\n")
        weight = _Tensor([[1.0, 2.0], [3.0, 4.0],
                          [5.0, 6.0], [7.0, 8.0]], _FakeTorch.uint8)
        for label, scale in (
            ("absent", None),
            ("short", _Tensor([0.5, 0.25, 0.125], _FakeTorch.bfloat16)),
        ):
            with self.subTest(scale=label):
                attach, logger = _attach_vocab_from(composed)
                model = _FakeModel(weight, scale)
                with mock.patch.dict(os.environ,
                                     {"VLLM_MTP_DRAFT_VOCAB": str(ids)}):
                    attach(model)
                self.assertFalse(hasattr(model, "_draft_lm_head_weight"))
                self.assertTrue(any("without a per-row" in m
                                    for m in logger.messages))


if __name__ == "__main__":
    unittest.main()
