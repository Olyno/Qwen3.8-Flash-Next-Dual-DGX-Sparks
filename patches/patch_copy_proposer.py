#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Copy-based draft proposer for the vLLM v0.32.0 lane (VLLM_COPY_PROPOSER=1).

Target: vllm/v1/worker/gpu/model_runner.py, GPUModelRunner — two sites:

  __init__, after use_dense_all_token_ids is derived from the speculative
  config (stock: True only for use_ngram()): force it True when the gate is
  on, so RequestState.all_token_ids is a dense device tensor. Stock's
  default is a UVA host mapping, and the scan below reads a request row
  every draft step; host pages would fault in on every access. The buffer
  is [max_num_reqs, max_model_len] int32 — 1 MiB at 4 x 65536, 8 MiB at
  8 x 262144.

  sample_tokens, at the draft handoff: after the speculator returns
  draft_tokens and the scheduled-width tail is set to -1, replace the rows
  that have a copy match, before the write to req_states.draft_tokens.

The scan and the gather are the stock n-gram kernels, imported unchanged
from vllm.v1.worker.gpu.spec_decode.ngram.speculator (the ngram_gpu
method's speculator; that file is not overlaid):

  _ngram_scan_kernel[(num_reqs, n_blocks)](
      token_ids [max_num_reqs, max_model_len] int32, stride,
      idx_mapping int64 [B], total_len int32 [max_num_reqs],
      num_sampled int32 [B], scratch int64 [B, n_blocks], stride, L,
      MIN_N, MAX_N, MAX_N_PO2, BLOCK_L, num_warps=4, num_stages=2)
    Scans every start position in [0, seq_len - MIN_N - 1] for every
    suffix length n in [MIN_N, MAX_N], comparing against the last n
    tokens, and packs a candidate as n * (L + 1) + pos + 1. One max over
    the per-block scores therefore yields longest-n first, rightmost-pos
    within that length. The position bound leaves at least one
    continuation token.

  _ngram_finalize_kernel[(num_reqs,)](
      token_ids, stride, idx_mapping, total_len, num_sampled,
      last_sampled int64 [max_num_reqs], scratch, stride,
      drafts int64 [B, K], L, N_BLOCKS, K, K_PO2, N_BLOCKS_PO2,
      num_warps=2, num_stages=1)
    Gathers tokens[best_pos + best_n : ...] into K columns, filling
    columns past the available continuation (and rows without a match)
    with last_sampled, a token the target may reject.

The decision is a pure function, copy_match_continuation(tokens, min_match,
max_match, cap): the rightmost longest suffix match — longest suffix length
first, rightmost start within that length, at least one continuation token,
at most cap tokens returned. The
kernel's packing rule is the same decision and the tests pin the two
against each other on random sequences. Defaults are 8 and 32 tokens;
VLLM_COPY_PROPOSER_MIN_MATCH / VLLM_COPY_PROPOSER_MAX_MATCH narrow or
widen the window inside the container. The scan unrolls
sum(min_match..max_match) loads per block, so a wide window is heavier
than the small windows ngram_gpu is usually configured with; the defaults
target long verbatim spans (copy-style traffic).

Per-request widths: the number of continuation tokens differs per request
and the finalize kernel already masks that (slot_valid), so no per-request
launch or width plumbing is needed. When a row matches, its whole draft
chain is replaced from column 0 — the MTP chain is recurrent, so a copy
prefix followed by MTP continuations would not be a chain. Columns past
the scheduled width keep the stock -1 fill, and rows without a match keep
their MTP drafts untouched. The drafts-per-request count stays the
scheduled width; this lane only changes their values. Under async
scheduling the scheduler takes only that count from the draft handler
(its spec_token_ids are count placeholders unless structured outputs are
active), and the next step's input build reads the values from
req_states.draft_tokens — the same buffer written here — so the copy
drafts are the ones verified, not just a count change.

Lossless: verification is exact rejection sampling, so any draft is safe
and only the acceptance rate moves. The copy tokens are the request's own
committed ids, so the reduced draft vocabulary (mtp_draft_vocab) and the
FP8 draft head neither help nor constrain this lane, and the scan reads
request state, not attention metadata, so qsa_fused_draft is orthogonal.
No guard is needed for either.

Host-side dispatch only: both launches happen in sample_tokens, outside
every cudagraph capture (the runner captures the model forward only), and
the path takes no D2H read — the match mask comes from the scan scratch on
device and the merge is a torch.where. The kernels stay eager.

Chaining: this overlays the same file as patch_spec_trace.py. With both
keys on, engine/patches.sh applies the trace patcher first and feeds its
output here; the trace patch adds `import os` at the top and this one adds
its own inside the module block, so the chained file parses either way.
With only this key on, stock is patched.

Patch time is fail-closed: every anchor must match exactly once, the
result must parse, and nothing is written on a mismatch.

Inputs:  <out_dir>/orig/model_runner.py (extracted from the pinned image,
         or the trace patcher's output when both keys are on)
Outputs: <out_dir>/model_runner_v030.py
argv[0] overrides the output directory (used by the test).
"""
import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_MARK = "_COPY_PROPOSER_ON"


def copy_match_continuation(tokens, min_match=8, max_match=32, cap=None):
    """Continuation for the rightmost longest suffix match, or None.

    Longest suffix length wins; within one length the rightmost start wins;
    a match must leave at least one continuation token. Returns at most cap
    tokens after the match (None = to the end of the sequence).
    """
    n_tokens = len(tokens)
    longest = min(max_match, n_tokens - 1)
    for n in range(longest, min_match - 1, -1):
        suffix = tokens[n_tokens - n:]
        for pos in range(n_tokens - n - 1, -1, -1):
            if tokens[pos:pos + n] == suffix:
                start = pos + n
                end = n_tokens if cap is None else start + cap
                return list(tokens[start:end])
    return None


class _CopyProposer:
    """Per-worker state and the stock-kernel dispatch for the copy lane."""

    def __init__(self, runner):
        try:
            import triton
            from vllm.triton_utils import HAS_TRITON
            from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (
                _ngram_finalize_kernel,
                _ngram_scan_kernel,
            )
        except ImportError as exc:
            raise RuntimeError(f"n-gram kernels unavailable: {exc}") from exc
        if not HAS_TRITON:
            raise RuntimeError("Triton is not available")
        try:
            self.min_match = int(
                os.environ.get("VLLM_COPY_PROPOSER_MIN_MATCH", "8")
            )
            self.max_match = int(
                os.environ.get("VLLM_COPY_PROPOSER_MAX_MATCH", "32")
            )
        except ValueError as exc:
            raise RuntimeError(f"bad match window: {exc}") from exc
        if not 1 <= self.min_match <= self.max_match:
            raise RuntimeError(
                "VLLM_COPY_PROPOSER match window must satisfy 1 <= min <= max "
                f"(got {self.min_match}..{self.max_match})"
            )

        self._scan = _ngram_scan_kernel
        self._finalize = _ngram_finalize_kernel
        self._req_states = runner.req_states
        self.max_num_reqs = runner.max_num_reqs
        self.max_model_len = runner.max_model_len
        self.num_speculative_steps = runner.num_speculative_steps

        # Same block shape the stock ngram speculator uses for this kernel.
        max_model_len = self.max_model_len
        if max_model_len >= 1024:
            self.block_l = 256
        elif max_model_len >= 256:
            self.block_l = 128
        elif max_model_len >= 64:
            self.block_l = 64
        else:
            self.block_l = max(16, triton.next_power_of_2(max(max_model_len, 1)))
        self.n_blocks = triton.cdiv(max_model_len, self.block_l)
        self.max_match_po2 = max(1, triton.next_power_of_2(self.max_match))
        self.k_po2 = max(1, triton.next_power_of_2(self.num_speculative_steps))
        self.n_blocks_po2 = max(1, triton.next_power_of_2(self.n_blocks))

        self.scratch = torch.zeros(
            (self.max_num_reqs, self.n_blocks),
            dtype=torch.int64,
            device=runner.device,
        )
        # Batch-ordered draft output, merged into the speculator's rows.
        self.drafts = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=runner.device,
        )

    def merge_drafts(self, input_batch, num_sampled, draft_tokens, num_spec_tokens):
        """Overlay copy drafts on the batch rows that have a suffix match."""
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return draft_tokens
        req_states = self._req_states
        token_ids = req_states.all_token_ids.gpu
        idx_mapping = input_batch.idx_mapping

        self._scan[(num_reqs, self.n_blocks)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            req_states.total_len.gpu,
            num_sampled,
            self.scratch,
            self.scratch.stride(0),
            self.max_model_len,
            self.min_match,
            self.max_match,
            self.max_match_po2,
            self.block_l,
            num_warps=4,
            num_stages=2,
        )
        self._finalize[(num_reqs,)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            req_states.total_len.gpu,
            num_sampled,
            req_states.last_sampled_tokens.view(-1),
            self.scratch,
            self.scratch.stride(0),
            self.drafts,
            self.max_model_len,
            self.n_blocks,
            self.num_speculative_steps,
            self.k_po2,
            self.n_blocks_po2,
            num_warps=2,
            num_stages=1,
        )

        # A positive block score means this row matched (the packed value is
        # n * (L + 1) + pos + 1) and finalize wrote its continuation from
        # column 0. Rows without a match keep the MTP drafts.
        matched = self.scratch[:num_reqs].max(dim=1).values > 0
        merged = torch.where(matched[:, None], self.drafts[:num_reqs], draft_tokens)
        if num_spec_tokens < self.num_speculative_steps:
            merged[:, num_spec_tokens:] = draft_tokens[:, num_spec_tokens:]
        return merged


_MODULE_COMMENT = '''\
# ---------------------------------------------------------------------------
# Copy-based draft proposer (patches/patch_copy_proposer.py). Off unless
# VLLM_COPY_PROPOSER=1. Enabled: every draft/verify round, each request's
# committed tokens (RequestState.all_token_ids, total_len) are scanned for
# the rightmost longest suffix match of VLLM_COPY_PROPOSER_MIN_MATCH to
# VLLM_COPY_PROPOSER_MAX_MATCH tokens that leaves a continuation; a match
# replaces the whole MTP draft row from column 0 with up to the scheduled
# width of continuation tokens. Host-side dispatch of the stock n-gram
# kernels, eager, no syncs. Target verification is exact, so any draft is
# safe; only the acceptance rate moves.
# ---------------------------------------------------------------------------
'''


def helper_block() -> str:
    """Gate, pure decision and dispatcher, inserted after the module logger."""
    pure = inspect.getsource(copy_match_continuation).strip("\n")
    dispatcher = inspect.getsource(_CopyProposer).strip("\n")
    return (
        _MODULE_COMMENT
        + "import os\n\n\n"
        + "_COPY_PROPOSER_ON = (\n"
        '    os.environ.get("VLLM_COPY_PROPOSER", "").strip() == "1"\n'
        ")\n\n\n"
        + pure
        + "\n\n\n"
        + dispatcher
        + "\n\n\n"
    )


MODULE_OLD = "logger = init_logger(__name__)\n"
MODULE_NEW = "logger = init_logger(__name__)\n\n\n" + helper_block()

DENSE_OLD = """\
        use_dense_all_token_ids = (
            self.speculative_config is not None and self.speculative_config.use_ngram()
        )
"""
DENSE_NEW = DENSE_OLD + """\
        # The copy proposer scans a request row every draft step; keep
        # all_token_ids device-resident instead of the UVA host mapping.
        if _COPY_PROPOSER_ON:
            use_dense_all_token_ids = True
"""

STATE_OLD = """\
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
            use_dense_all_token_ids=use_dense_all_token_ids,
        )
"""
STATE_NEW = STATE_OLD + """
        # Only the last pipeline-parallel rank reaches the draft handoff.
        self._copy_proposer = None
        if (
            _COPY_PROPOSER_ON
            and self.is_last_pp_rank
            and self.speculative_config is not None
        ):
            try:
                self._copy_proposer = _CopyProposer(self)
            except RuntimeError as exc:
                logger.warning(
                    "VLLM_COPY_PROPOSER: standing down (%s); MTP drafts stay stock.",
                    exc,
                )
"""

HANDOFF_OLD = """\
            if num_spec_tokens < self.num_speculative_steps:
                draft_tokens[:, num_spec_tokens:] = -1
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
"""
HANDOFF_NEW = """\
            if num_spec_tokens < self.num_speculative_steps:
                draft_tokens[:, num_spec_tokens:] = -1
            if self._copy_proposer is not None:
                draft_tokens = self._copy_proposer.merge_drafts(
                    input_batch, num_sampled, draft_tokens, num_spec_tokens
                )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
"""

EDITS = (
    (MODULE_OLD, MODULE_NEW),
    (DENSE_OLD, DENSE_NEW),
    (STATE_OLD, STATE_NEW),
    (HANDOFF_OLD, HANDOFF_NEW),
)


def patch(src: str) -> str:
    if PATCH_MARK in src:
        sys.exit("ERROR: copy_proposer orig model_runner.py is already patched")
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"copy_proposer: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"copy_proposer: patched model_runner_v030.py does not parse: {exc}")
    return src


def main(argv) -> int:
    out_dir = argv[0] if argv else os.path.join(HERE, "copy_proposer")
    orig_path = os.path.join(out_dir, "orig", "model_runner.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(engine/patches.sh extracts it from the pinned image)")
    src = patch(open(orig_path).read())
    out_path = os.path.join(out_dir, "model_runner_v030.py")
    with open(out_path + ".tmp", "w") as f:
        f.write(src)
    os.replace(out_path + ".tmp", out_path)
    print("patched model_runner_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
