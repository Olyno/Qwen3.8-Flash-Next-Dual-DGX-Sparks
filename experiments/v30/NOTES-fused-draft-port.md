# vllm#58449 → v0.30.0 port notes (QSA fused multi-step draft decode)

Base image: `vllm/vllm-openai@sha256:8a69ffad` (vLLM 0.30.0), container `v30a3g`.
Target file: `/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/qsa_cache.py`
(885 lines pristine, extracted read-only via `docker exec v30a3g cat`).
Patcher: `files/patch_qsa_fused_draft_v030.py` — anchor-exact `(anchor, replacement)` pairs
vendored byte-exact from the PR's qsa_cache.py hunks; `count==1` assert per anchor; refuses on
drift, on double-apply (MARK line guard), and writes only after `ast.parse`.
Output dirs `files/v030_fused/` + `files/v030_fused_draft/` added to the "launch-time generated
overlays" block of the worktree `.gitignore` (lines 64–65), same convention as `files/v030_ple/`
and `files/v030_fp8kv/`.

## Diff vs PR (adaptation audit)

**Zero line-level adaptation was required.** All 8 qsa_cache.py hunks from PR #58449 apply
verbatim to the pristine 0.30 file: every old-side anchor was found exactly once. Cross-checked:
`git apply` of the qsa_cache-only slice of the PR diff onto the pristine file produces output
byte-identical to the patcher's output modulo the MARK comment
(`diff <(grep -v MARK out/qsa_cache.py) gitapply_result` → empty).

The PR's 9th hunk is `tests/models/qwen4_exp/test_qsa_reference.py` (new
`test_qsa_draft_decode_metadata_update_matches_rebuild`) — test-only, not part of the image,
deliberately NOT ported into the mount. Acceptance is proven at boot/bench level instead
(see Bench plan; the PR test's rebuild-equivalence is exactly what ±2pp per-position encodes).

What the patch does, in PR form:

1. `build_qsa_metadata_triton` splits: thin wrapper + new `_launch_qsa_metadata_kernel(...)`
   that takes raw tensors + scalars and fills metadata buffers in place (capture-safe given
   fixed shapes and scalars). Wrapper behavior unchanged.
2. `QSAForwardMetadata` gains two fields: `common_slot_mapping: torch.Tensor`,
   `num_mapped_tokens: int`. Only one constructor exists in the image (`build()`; grep for
   `QSAForwardMetadata(` outside common/qsa_cache.py → only imports/casts, no constructions),
   so the dataclass change is closed.
3. `build()` populates both new fields from `common_attn_metadata`.
4. `QSAMetadataBuilder.supports_draft_decode_metadata_update = HAS_TRITON` (+ our MARK line).
5. New `QSAMetadataBuilder.update_draft_decode_metadata()` re-launches the extracted kernel on
   the metadata's own persistent buffers — seq_lens/query_start_loc/block_table are read
   in place, nothing is reallocated.

## HAS_TRITON semantics

Pristine file line 26: `from vllm.triton_utils import HAS_TRITON, tl, triton`; line 571
already gates `build_qsa_metadata = build_qsa_metadata_triton if HAS_TRITON else _build_qsa_metadata_torch`.
So the flag is load-bearing in-tree, and the PR's `supports_draft_decode_metadata_update = HAS_TRITON`
is the *correct* gate, not an always-on constant: in any environment where HAS_TRITON is False
(vllm/triton_utils/importing.py: triton missing, `triton.backends` unimportable, or ≠1 active
non-CPU driver outside distributed envs → set False with an info/warning log), the fused path
must stay off and the speculator keeps rebuilding metadata between draft steps — and the torch
fallback builder (never receives the flag) is untouched by this port. In `v30a3g` triton 3.7.1
is installed and one CUDA driver is active, so HAS_TRITON is True there. The port hard-codes
`True` nowhere: the flag assignment is literally `= HAS_TRITON`, verbatim from the PR.

## In-place buffer reuse — verification against 0.30 speculator

Read via `docker exec v30a3g sed/cat` (read-only):
`/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`
(plus base `../spec_decode/speculator.py`, `v1/worker/utils.py`, `v1/attention/backend.py`,
`v1/worker/gpu/block_table.py`).

The fused loop (`_generate_fused_drafts` at speculator.py:605, step loop at 623, in-place
update call at 645):

```python
        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(...)
            if (
                step < self.num_speculative_steps - 1
                and attn_metadata is not None
                and self.advance_draft_positions
            ):
                self.block_tables.compute_slot_mappings(
                    idx_mapping, query_start_loc, positions, num_tokens_padded,
                )
                for attn_group in attn_groups:
                    attn_group.update_draft_decode_metadata(attn_metadata)
```

Tensor-identity evidence — every input the QSA update re-reads is a persistent buffer that is
mutated in place, never reallocated, across draft steps:

- `query_start_loc` / `seq_lens` / `positions`: slices of `self.input_buffers` (fixed-size
  speculator buffers; `_generate_fused_drafts` opens with
  `positions = self.input_buffers.positions[:num_reqs]`,
  `query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]`). Between steps they
  are advanced in place by `_update_draft_inputs_kernel` (speculator.py ≈900–971):
  `position = tl.minimum(position + 1, max_model_len - 1); tl.store(positions_ptr + req_idx, position)`
  and `seq_len = tl.minimum(seq_len + 1, max_model_len); tl.store(seq_lens_ptr + req_idx, seq_len)`
  — stores through the same base pointers; the [:num_reqs] slice views keep identity and shape.
- `block_table`: base `_build_attn_metadata` (spec_decode/speculator.py:325–328) does
  `block_tables = [x[:num_reqs_padded] for x in self.block_tables.input_block_tables]` and
  `slot_mappings = self.block_tables.slot_mappings[:, :num_tokens]` — persistent tensors
  re-sliced identically every build; `get_dummy_slot_mappings`/zero helpers carry the comment
  "this method must return the persistent tensor with the same memory address ... rather than
  allocating a new tensor". `compute_slot_mappings` (v1/worker/gpu/block_table.py:191) writes
  into `self.slot_mappings` via `_compute_slot_mappings_kernel`, no allocation.
- metadata is built once per forward: `_fused_multi_step_decode` (def at speculator.py:558)
  calls `_build_uniform_attn_metadata(..., step=1)` before the loop and passes the *same* dict object
  to every `update_draft_decode_metadata`; `AttentionGroup.update_draft_decode_metadata`
  (worker/utils.py:322–327) resolves `attn_metadata[self.layer_names[0]]` → our
  `QSAForwardMetadata` → builder. In-place mutated `seq_lens` is exactly what the rebuild path
  re-reads each step, so update == rebuild iff the kernel is deterministic on those inputs —
  which is what the PR's test asserts (fresh build at seq_lens+1 equals the in-place update).
- ordering: the loop recomputes slot mappings *before* calling the update, and our update reads
  `metadata.common_slot_mapping` (captured at build = the persistent
  `slot_mappings[:, :num_tokens]` view) — so it sees the freshly written mapping of the *next*
  step's tokens, matching what a rebuild would capture. Same object; only contents advance.
- host-side plumbing pre-exists in 0.30: `AttentionMetadataBuilder.supports_draft_decode_metadata_update
  = False` default + `update_draft_decode_metadata` raising NotImplementedError
  (attention/backend.py:591, :720) marked "implementations must emit capture-safe operations and
  keep replayed tensor state in persistent storage"; `_configure_fused_multi_step_decode`
  (autoregressive speculator.py:101–127) enables `use_fused_multi_step_decode` only when every
  attn group's builder advertises the flag. The QSA builder flag is the only gate the patch has
  to flip — confirmed: both QSA cache groups (raw-key + compressed-key) use QSAMetadataBuilder.

Remaining correctness risk (documented, inherited from the PR design):

- The update re-launches the kernel with `num_mapped_tokens` and `num_tokens` **frozen at build
  time**. Safe in the fused path because it is only taken for uniform decode (query_len=1 per
  req) with fixed padded shapes; a token-count-changing step would require a rebuild, which the
  loop structure never does. If a future 0.30.x routes non-uniform batches through the fused
  loop, the frozen scalars would go stale — the guard is `_build_uniform_attn_metadata` being
  the sole producer, which holds in this image.
- Padded `seq_lens` entries are zeroed for cudagraph (padding loops in
  `_prepare_decode_inputs_kernel`/`_update_draft_inputs_kernel`) — the kernel's mapped-token
  bound (`num_mapped_tokens`) skips them exactly as at build time.
- Kernel launch parameters (`log2(num_reqs)`, grid size) are recomputed per launch from tensor
  *shapes*, which are constant across fused steps → CUDA-graph-capture-stable, as the
  backend.py docstring demands.

## Bench plan (arm QSA_FUSED=1)

- Mount: `-v ~/wt-reactivity/files/v030_fused_draft/qsa_cache.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/qsa_cache.py:ro`
  (ride_r2.sh arms the identical file at `files/v030_fused/qsa_cache.py`; both staged copies are
  md5-identical: `f723fa14feea0278df082cccb7733d3c`).
- Env: `QSA_FUSED=1` selects the arm in the driver. Nothing in vLLM reads it — the patched
  builder self-enables via HAS_TRITON; the var only switches boot/mount in the harness.
  Everything else (model, MTP k, seeds, block sizes) unchanged from the QSA baseline arm.
- Workload: decodebench, 600-token prompts, prose + code splits, @1k output tokens, same
  sampling settings as the baseline; k=4 MTP target.
- Acceptance:
  1. Boot log MUST NOT contain the fallback sentence
     `"Fused multi-step draft decode is not supported by attention backend(s) ...; falling back
     to rebuilding attention metadata between draft steps."` (logger.info_once, autoregressive
     speculator.py:121). Its absence + a completed run = fused path armed.
  2. Per-position acceptance within ±2pp of the unfused QSA baseline (the fused path must be
     numerically equivalent; >2pp divergence means in-place update disagrees with rebuild →
     revert arm, investigate).
  3. Performance reference: PR measured +1.2% decode throughput at k=4, c=1. Expect the same
     order on our MTP config; worse-than-baseline with acceptance OK → remeasure (check FULL
     vs PIECEWISE graph mode before believing it).
- GPU note: no GPU work was run for this port. The live `v30a3g` container was touched only via
  read-only `docker exec cat/sed/grep` (plus a version print); nothing was written into it.

## Verification log (this session)

- `[ok] patched, ast.parse OK, MARK once -> out/qsa_cache.py` (local pristine apply; exit 0).
- `grep -c MARK out/qsa_cache.py` → 1.
- git-apply cross-check: patcher output ≡ PR hunks applied by `git apply` (MARK-only delta).
- Negative control: `_cudagraph_support` anchor line perturbed UNIFORM_BATCH→ALWAYS →
  `DRIFT: vllm#58449 anchor 5 not unique/missing (count=0) ... nothing written`, exit 1, no
  output file.
- Double-apply refusal: patcher on its own output → `already carries the vllm#58449 port
  (MARK x1); refusing to double-apply`, exit 1.
- py3.12 parse: `ast.parse(..., feature_version=(3,12))` OK (container python is 3.12).
- msi staging: `~/wt-reactivity/files/patch_qsa_fused_draft_v030.py` +
  `~/wt-reactivity/files/v030_fused_draft/qsa_cache.py` + `~/wt-reactivity/files/v030_fused/qsa_cache.py`
  (both re-run on msi from pristine → same md5 `f723fa14feea0278df082cccb7733d3c`).
