"""Source pins, anchor application and eligibility tests for
patches/patch_gdn_direct_out.py.

The anchors below are copied verbatim from the pinned v0.32.0
qwen_gdn_linear_attn.py, so the fake-orig class runs the
anchor/count/parse logic anywhere. The eligibility tests are pure
python (duck-typed tensors, no torch, no image): the shipped predicate
must accept exactly the destination flashinfer can write in place. The
real-orig class needs the extracted image file and skips until
engine/patches.sh has run once.
"""
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "patches" / "patch_gdn_direct_out.py"
ORIG = (
    REPO / "patches" / "gdn_direct_out" / "orig" / "qwen_gdn_linear_attn.py"
)
ENV = "VLLM_GDN_DIRECT_OUT"
TOUCHED = (
    PATCHER,
    REPO / "engine" / "patches.sh",
    REPO / "engine" / "config.sh",
    REPO / "engine" / "recipe.py",
    REPO / "recipes" / "prod.yaml",
    REPO / ".gitignore",
)


def load_patcher():
    spec = importlib.util.spec_from_file_location("gdn_direct_out_patch", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    # inspect.getsource() on the predicate needs the module findable.
    sys.modules["gdn_direct_out_patch"] = mod
    spec.loader.exec_module(mod)
    return mod


PATCH = load_patcher()

# Verbatim from the image's
# vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py.
MODULE_ANCHOR = "logger = init_logger(__name__)\n"
FI_SIG_ANCHOR = """\
def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
"""
FI_CALL_ANCHOR = """\
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        backend="flashinfer",
    )
"""
CUDA_ANCHOR = """\
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state
"""
INIT_ANCHOR = """\
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
"""
PREFILL_CALL_ANCHOR = """\
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                aiter_prefill_metadata=attn_metadata.aiter_prefill_metadata,
            )
"""
MERGE_ANCHOR = """\
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
"""
# Verbatim stock initial allocation; this overlay must leave it alone.
ZERO_INIT = """\
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
"""


def fake_gdn_src():
    """Minimal parseable module carrying every anchor verbatim."""
    return (
        "import torch\n"
        "from vllm.logger import init_logger\n\n"
        + MODULE_ANCHOR
        + "\n\nMAX_FUSED_GDN_MTP_TOKENS = 8\n\n\n"
        "def fla_chunk_gated_delta_rule(**kwargs):\n"
        "    return None\n\n\n"
        + FI_SIG_ANCHOR
        + "    from flashinfer.gdn_prefill import (\n"
        "        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,\n"
        "    )\n"
        "    fi_g = g\n"
        "    fi_beta = beta\n"
        "    fi_state = initial_state\n"
        + FI_CALL_ANCHOR
        + "    return result\n\n\n"
        "class ChunkGatedDeltaRule:\n"
        "    def forward_cuda(\n"
        "        self,\n"
        "        q,\n"
        "        k,\n"
        "        v,\n"
        "        g,\n"
        "        beta,\n"
        "        initial_state,\n"
        "        output_final_state,\n"
        "        cu_seqlens=None,\n"
        "        use_qk_l2norm_in_kernel=True,\n"
        "        core_attn_out=None,\n"
        "        aiter_prefill_metadata=None,\n"
        "    ):\n"
        + CUDA_ANCHOR
        + "\n\n"
        "class GatedDeltaNetAttention:\n"
        "    def forward(self, hidden_states, num_tokens):\n"
        + ZERO_INIT
        + "        return core_attn_out\n\n"
        "    def _forward_core(self, mixed_qkv, b, a, core_attn_out):\n"
        "        attn_metadata = self._metadata\n"
        "        spec_sequence_masks = attn_metadata.spec_sequence_masks\n"
        "        split_non_spec = False\n"
        + INIT_ANCHOR
        + "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n"
        "        num_decode_tokens = attn_metadata.num_decode_tokens\n"
        "        if attn_metadata.num_prefills > 0:\n"
        "            query_non_spec = mixed_qkv\n"
        "            key_non_spec = mixed_qkv\n"
        "            value_non_spec = mixed_qkv\n"
        "            g_non_spec = None\n"
        "            beta_non_spec = None\n"
        "            initial_state = None\n"
        + PREFILL_CALL_ANCHOR
        + "            core_attn_out_non_spec = core_attn_out_non_spec\n"
        "        elif attn_metadata.num_decodes > 0:\n"
        "            core_attn_out_non_spec = mixed_qkv\n"
        "        else:\n"
        "            core_attn_out_non_spec = None\n"
        "        if spec_sequence_masks is not None and core_attn_out_non_spec:\n"
        "            core_attn_out.index_copy_(0, None, core_attn_out_non_spec)\n"
        "        elif spec_sequence_masks is not None:\n"
        "            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec\n"
        + MERGE_ANCHOR
        + "\n"
    )


def write_orig(tmp, src, name="gdn_direct_out"):
    out = tmp / name
    (out / "orig").mkdir(parents=True)
    (out / "orig" / "qwen_gdn_linear_attn.py").write_text(src)
    return out


def run_patcher(out):
    return subprocess.run(
        [sys.executable, str(PATCHER), str(out)], capture_output=True, text=True
    )


class FakeTensor:
    """Duck-typed stand-in for the attributes the predicate reads."""

    def __init__(self, shape=(6, 32, 128), dtype="bfloat16",
                 device="cuda:0", contiguous=True, ptr=0):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self._ptr = ptr

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr


SHAPE = (6, 32, 128)
DTYPE = "bfloat16"
DEVICE = "cuda:0"


class Eligibility(unittest.TestCase):
    """_gdn_direct_out_target: only a valid kernel destination is accepted."""

    def target(self, dest):
        return PATCH._gdn_direct_out_target(dest, SHAPE, DTYPE, DEVICE)

    def test_eligible_destination_is_returned_as_is(self):
        dest = FakeTensor()
        self.assertIs(self.target(dest), dest)

    def test_none_is_stock(self):
        self.assertIsNone(self.target(None))

    def test_rank_must_be_three(self):
        self.assertIsNone(self.target(FakeTensor(shape=(6, 4096))))
        self.assertIsNone(self.target(FakeTensor(shape=(6, 32, 128, 1))))

    def test_shape_must_match_exactly(self):
        self.assertIsNone(self.target(FakeTensor(shape=(6, 32, 127))))
        self.assertIsNone(self.target(FakeTensor(shape=(5, 32, 128))))

    def test_dtype_must_match(self):
        self.assertIsNone(self.target(FakeTensor(dtype="float32")))
        self.assertIsNone(self.target(FakeTensor(dtype="float16")))

    def test_device_must_match(self):
        self.assertIsNone(self.target(FakeTensor(device="cuda:1")))
        self.assertIsNone(self.target(FakeTensor(device="cpu")))

    def test_non_contiguous_is_stock(self):
        self.assertIsNone(self.target(FakeTensor(contiguous=False)))

    def test_alignment_is_checked(self):
        self.assertIsNotNone(self.target(FakeTensor(ptr=512)))
        self.assertIsNone(self.target(FakeTensor(ptr=8)))
        self.assertIsNone(self.target(FakeTensor(ptr=513)))

    def test_non_tensor_is_stock_not_a_crash(self):
        self.assertIsNone(self.target(object()))

    def test_failing_attribute_is_stock_not_a_crash(self):
        class Bad(FakeTensor):
            def data_ptr(self):
                raise RuntimeError("no pointer")

        self.assertIsNone(self.target(Bad()))

    def test_generated_helper_agrees(self):
        with_env = exec_helper("1")
        off = exec_helper(None)
        for dest in (
            FakeTensor(),
            FakeTensor(contiguous=False),
            FakeTensor(dtype="float32"),
            FakeTensor(shape=(1, 2, 3)),
            None,
        ):
            for ns in (with_env, off):
                self.assertIs(
                    ns["_gdn_direct_out_target"](dest, SHAPE, DTYPE, DEVICE),
                    PATCH._gdn_direct_out_target(dest, SHAPE, DTYPE, DEVICE),
                )


def exec_helper(gate):
    """Exec the shipped helper block with the env gate as it would boot."""
    ns = {"os": os}
    old = os.environ.pop(ENV, None)
    try:
        if gate is not None:
            os.environ[ENV] = gate
        exec(compile(ast.parse(PATCH.helper_block()), "<helper>", "exec"), ns)
    finally:
        os.environ.pop(ENV, None)
        if old is not None:
            os.environ[ENV] = old
    return ns


class Gate(unittest.TestCase):
    """Env gate: off is stock, exactly 1 arms the direct path."""

    def test_gate_off_by_default(self):
        for gate in (None, "0", "true", "yes", ""):
            self.assertFalse(exec_helper(gate)["_GDN_DIRECT_OUT_ON"], gate)

    def test_gate_on_only_for_one(self):
        self.assertTrue(exec_helper("1")["_GDN_DIRECT_OUT_ON"])
        self.assertTrue(exec_helper(" 1 ")["_GDN_DIRECT_OUT_ON"])


class PatchApplication(unittest.TestCase):
    """Anchor application, parsing and fail-closed behavior."""

    def test_fake_orig_applies_and_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_orig(Path(tmp), fake_gdn_src())
            res = run_patcher(out)
            self.assertEqual(res.returncode, 0, res.stderr)
            patched = (out / "qwen_gdn_linear_attn_v030.py").read_text()
            ast.parse(patched)
            self.assertIn("output=output,", patched)
            self.assertIn("output=direct_out,", patched)
            self.assertIn("core_attn_out=prefill_out_direct,", patched)
            self.assertIn("elif prefill_out_direct is None:", patched)
            self.assertIn(
                "if core_attn_out is not None and direct_out is None:", patched
            )
            self.assertIn("_GDN_DIRECT_OUT_ON = (", patched)
            self.assertIn(ZERO_INIT, patched, "the zero init must stay stock")

    def test_idempotence_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_orig(Path(tmp), fake_gdn_src())
            self.assertEqual(run_patcher(out).returncode, 0)
            patched = (out / "qwen_gdn_linear_attn_v030.py").read_text()
            with self.assertRaises(SystemExit) as ctx:
                PATCH.patch(patched)
            self.assertIn("already patched", str(ctx.exception))

    def test_anchor_drift_fails_closed(self):
        drifted = fake_gdn_src().replace(
            "                use_qk_l2norm_in_kernel=False,\n",
            "                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n",
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = write_orig(Path(tmp), drifted)
            res = run_patcher(out)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("not unique/missing", res.stderr)
            self.assertFalse(
                (out / "qwen_gdn_linear_attn_v030.py").exists(),
                "no output may be written on an anchor mismatch",
            )

    def test_duplicate_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_orig(Path(tmp), fake_gdn_src() + MERGE_ANCHOR)
            res = run_patcher(out)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("count=2", res.stderr)

    def test_missing_orig_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "empty"
            out.mkdir()
            res = run_patcher(out)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("missing", res.stderr)
            self.assertIn("engine/patches.sh", res.stderr)

    def test_real_orig_applies_when_extracted(self):
        if not ORIG.is_file():
            self.skipTest("image file not extracted yet")
        with tempfile.TemporaryDirectory() as tmp:
            out = write_orig(Path(tmp), ORIG.read_text())
            res = run_patcher(out)
            self.assertEqual(res.returncode, 0, res.stderr)
            patched = (out / "qwen_gdn_linear_attn_v030.py").read_text()
            ast.parse(patched)
            self.assertIn("prefill_out_direct", patched)


def find_def(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no def {name}")


def calls_to(fn, func_name):
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == func_name:
            out.append(node)
        elif isinstance(f, ast.Attribute) and f.attr == func_name:
            out.append(node)
    return out


def kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def if_containing(fn, marker):
    """If nodes whose body directly assigns a Name matching marker."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == marker
                for t in stmt.targets
            ):
                out.append(node)
    return out


class AstPins(unittest.TestCase):
    """Control-flow pins on the patched module itself."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_orig(Path(tmp), fake_gdn_src())
            res = run_patcher(out)
            assert res.returncode == 0, res.stderr
            cls.src = (out / "qwen_gdn_linear_attn_v030.py").read_text()
        cls.tree = ast.parse(cls.src)

    def test_kernel_gets_the_output_argument(self):
        fn = find_def(self.tree, "fi_chunk_gated_delta_rule")
        calls = calls_to(fn, "chunk_gated_delta_rule_fi")
        self.assertEqual(len(calls), 1)
        value = kwarg(calls[0], "output")
        self.assertIsInstance(value, ast.Name)
        self.assertEqual(value.id, "output")

    def test_forward_cuda_passes_the_eligible_destination(self):
        fn = find_def(self.tree, "forward_cuda")
        calls = calls_to(fn, "fi_chunk_gated_delta_rule")
        self.assertEqual(len(calls), 1)
        value = kwarg(calls[0], "output")
        self.assertIsInstance(value, ast.Name)
        self.assertEqual(value.id, "direct_out")

    def test_copy_is_guarded_by_the_eligibility_result(self):
        fn = find_def(self.tree, "forward_cuda")
        copies = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "copy_"
        ]
        self.assertEqual(len(copies), 1, "one copy_ call, the stock fallback")
        # It must be inside the `core_attn_out is not None and direct_out is
        # None` branch, so an eligible destination never launches it.
        guarded = False
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            text = ast.dump(node.test)
            if "core_attn_out" in text and "direct_out" in text:
                for stmt in node.body:
                    if any(
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "copy_"
                        for n in ast.walk(stmt)
                    ):
                        guarded = True
        self.assertTrue(guarded, "the stock copy_ must be behind the fallback")
        self.assertIn(
            "logger.warning_once", ast.unparse(fn)
        )

    def test_merge_site_skips_the_assignment_for_a_direct_write(self):
        fn = find_def(self.tree, "_forward_core")
        calls = calls_to(fn, "chunk_gated_delta_rule")
        self.assertEqual(len(calls), 1)
        value = kwarg(calls[0], "core_attn_out")
        self.assertIsInstance(value, ast.Name)
        self.assertEqual(value.id, "prefill_out_direct")
        guards = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "prefill_out_direct is None"
        ]
        self.assertEqual(
            len(guards), 1, "the stock merge assignment must be guarded"
        )
        assigned = any(
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Subscript)
            for stmt in guards[0].body
        )
        self.assertTrue(assigned, "the guarded body is the stock merge copy")

    def test_gate_decides_between_direct_and_stock(self):
        # Gate on + eligible destination: the predicate hands it to the
        # kernel and the copy condition (`core_attn_out is not None and
        # direct_out is None`) stays false.
        on = exec_helper("1")
        dest = FakeTensor()
        direct = on["_gdn_direct_out_target"](dest, SHAPE, DTYPE, DEVICE)
        self.assertIs(direct, dest)
        self.assertFalse((dest is not None) and (direct is None))
        # Gate off: the call site's flag is the initial None, so forward_cuda
        # receives core_attn_out=None, allocates (output=None) and merges the
        # result at the stock site.
        off = exec_helper(None)
        direct = off["_gdn_direct_out_target"](None, SHAPE, DTYPE, DEVICE)
        self.assertIsNone(direct)
        self.assertFalse((None is not None) and (direct is None))

    def test_gate_guards_the_call_site_assignment(self):
        fn = find_def(self.tree, "_forward_core")
        gates = if_containing(fn, "prefill_out_direct")
        self.assertEqual(len(gates), 1)
        self.assertIn("_GDN_DIRECT_OUT_ON", ast.dump(gates[0].test))
        for token in ("gdn_prefill_backend", "spec_sequence_masks",
                      "split_non_spec"):
            self.assertIn(token, ast.dump(gates[0].test))
        # The flag starts as None, so gate-off keeps the stock path.
        none_assigns = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "prefill_out_direct"
                for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ]
        self.assertEqual(len(none_assigns), 1)
        self.assertLess(none_assigns[0].lineno, gates[0].lineno)


class PatcherSource(unittest.TestCase):
    """Docstring statements and fail-closed pins that need no orig."""

    def setUp(self):
        self.src = PATCHER.read_text()
        self.doc = ast.get_docstring(ast.parse(self.src)) or ""

    def test_docstring_names_the_target_and_the_change(self):
        self.assertIn("qwen_gdn_linear_attn.py", self.doc)
        self.assertIn("VLLM_GDN_DIRECT_OUT", self.doc)
        self.assertIn("output=", self.doc)
        self.assertIn("bit-parity", self.doc)
        self.assertIn("Prefill only", self.doc)
        self.assertIn("core_attn_out", self.doc)

    def test_docstring_states_the_eligibility_rule(self):
        self.assertIn("3-D", self.doc)
        self.assertIn("contiguous", self.doc)
        self.assertIn("dtype", self.doc)
        self.assertIn("16-byte", self.doc)
        self.assertIn("warning", self.doc)

    def test_fail_closed_messages(self):
        self.assertIn("already patched", self.src)
        self.assertIn("not unique/missing", self.src)
        self.assertIn("does not parse", self.src)
        self.assertIn("missing", self.src)
        self.assertIn("count != 1", self.src)
        self.assertIn("ast.parse", self.src)

    def test_zero_initialization_is_not_touched(self):
        # The prefill buffer stays a torch.zeros; only the kernel destination
        # changes in this overlay.
        self.assertNotIn("torch.zeros", self.src)
        self.assertNotIn("torch.empty", self.src)

    def test_wiring_pins(self):
        patches = (REPO / "engine" / "patches.sh").read_text()
        self.assertIn("patch_gdn_direct_out.py", patches)
        self.assertIn("-e VLLM_GDN_DIRECT_OUT=1", patches)
        self.assertIn("qwen_gdn_linear_attn_v030.py", patches)
        self.assertIn("qwen_gdn_linear_attn.py", patches)
        config = (REPO / "engine" / "config.sh").read_text()
        self.assertIn('GDN_DIRECT_OUT="${GDN_DIRECT_OUT:-false}"', config)
        self.assertIn("gdn_direct_out is only supported", config)
        recipe = (REPO / "engine" / "recipe.py").read_text()
        self.assertIn('"GDN_DIRECT_OUT"', recipe)
        self.assertIn("patches/gdn_direct_out/", (REPO / ".gitignore").read_text())
        yaml = (REPO / "recipes" / "prod.yaml").read_text()
        self.assertIn("gdn_direct_out: false", yaml)
        self.assertIn("bit-parity", yaml)
        self.assertIn("prefill only", yaml)

    def test_recipe_key_defaults_off(self):
        out = subprocess.run(
            [sys.executable, str(REPO / "engine" / "recipe.py"),
             str(REPO / "recipes"), "prod"],
            capture_output=True, text=True, check=True,
        ).stdout
        self.assertIn("export GDN_DIRECT_OUT='false'", out)

    def test_no_host_identifiers(self):
        # Built at runtime so the name itself is never committed.
        host = "m" + "si"
        for path in TOUCHED:
            self.assertIsNone(
                re.search(rf"\b{host}\b", path.read_text(), re.IGNORECASE),
                f"{path.name} mentions the bench host",
            )


if __name__ == "__main__":
    unittest.main()
