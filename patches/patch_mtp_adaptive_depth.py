#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Adaptive MTP draft depth for the vLLM 0.30.0 lane (opt-in,
MTP_ADAPTIVE_DEPTH=true).

Targets the V2 model runner's draft loop — the code gpu_worker.py actually
runs ("Using V2 Model Runner"):

  vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
      AutoRegressiveSpeculator (MTPSpeculator inherits it unchanged):
      - sample_draft / _maybe_predict_acceptance overrides record each draft
        token's top-probability into a persistent [max_num_reqs, k] buffer,
        column-indexed by current_draft_step (capture-safe, so the writes
        also happen inside FULL draft cudagraphs).
      - _multi_step_decode / _generate_fused_drafts (the host-driven draft
        loops) break early once the survival product crosses the threshold.
        Under graph capture the check stays out (a D2H sync cannot be
        captured); the fused-FULL path records the full chain and the cutoff
        is applied post-hoc, see below.
      - propose() ends in _adaptive_finalize: one D2H read of the recorded
        columns, the running-product rule below, and the result is published
        as adaptive_num_draft_tokens.
  vllm/v1/worker/gpu/model_runner.py
      The draft handoff slices req_states.draft_tokens to
      speculator.adaptive_num_draft_tokens before DraftTokensHandler sees it
      (the handler takes the width from the tensor's shape, and the scheduler
      schedules len(spec_token_ids) per request — variable chain widths are a
      stock code path). propose() itself keeps returning the full-width
      buffer: req_states.draft_tokens is fixed [max_num_reqs, k], and the
      dropped columns simply never leave the worker.

The survival product is the running product of per-step top-token
probabilities (the draft head's own survival score, DSpark/SVIP-style); the
batch rule is the mean of per-request products vs
VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD (default 0.5, Strata's spec-min-p operating
point). Training-free and exact: target verification is untouched, truncated
drafts are simply never proposed (the engine treats them like a full
rejection suffix, which stock MTP verify already handles).

k is decided per STEP, uniform across the batch (a per-sequence k would break
the uniform-batch cudagraph dispatch). Disabled under data parallelism (DP
ranks would desync the width); TP ranks share the batch and the survival
product is built from gathered/all-reduced values, so every rank cuts
identically.

Confidence source per draft sampling path:
- local argmax reduction (mtp_draft_vocab, the prod path): ids stay the
  model's own get_top_tokens output (bit-identical to stock); the winner's
  softmax mass is recomputed off the reduced/FP8 head — the dropped rows'
  mass is missing from the denominator, an overestimate that truncates less
  than a full-vocab score, never more. One extra skinny [batch, vocab_slice]
  GEMM + one [batch] all-gather/all-reduce pair per draft step.
- greedy full-vocab: exact winner mass off the compute_logits logits that
  sample_draft already materializes (one softmax per draft step).
- probabilistic drafting: scored off the pre-temperature logits — an upper
  bound at temperature > 0, so it truncates less (the safe direction).
The cutoff check is one small D2H sync per propose (plus one per draft step
in the host-driven loops).

Composes with qsa_fused_draft: that overlay patches qsa_cache.py (the
attention-metadata builder the speculator calls), a different file. With both
on, the fused single-graph draft loop runs full length and the truncation is
applied post-hoc to its output; with it off, the host-driven
_multi_step_decode loop additionally stops replaying draft graphs at the cut.

Runtime gate: VLLM_MTP_ADAPTIVE_DEPTH=1 plus VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD,
both passed via OVERLAY_ENV (engine/patches.sh). Requirements enforced there:
v030 lane, MTP_NUM_SPECULATIVE_TOKENS > 1. VLLM_MTP_ADAPTIVE_DEBUG=1 (recipe
mtp_adaptive_debug) adds rate-limited logging of the recorded per-step probs,
survival products, published widths, and a one-shot full-head vs reduced-head
winner-mass comparison.

Inputs:  patches/mtp_adaptive_depth/orig/speculator.py (from the image)
         patches/mtp_adaptive_depth/orig/model_runner.py (from the image)
Outputs: patches/mtp_adaptive_depth/speculator_v030.py
         patches/mtp_adaptive_depth/model_runner_v030.py
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _adaptive_cut(running_prod: float, threshold: float) -> bool:
    """True once the survival product drops strictly below threshold."""
    return running_prod < threshold


def adaptive_draft_length(step_probs, threshold: float = 0.5) -> int:
    """Draft tokens to keep for a per-step top-prob sequence.

    Keeps draft token i, then stops the chain once the running product
    prod(p_1..p_i) falls below threshold; always keeps at least one token.
    Single-chain statement of the cutoff (the batch rule is
    adaptive_batch_keep); the proposer applies the same rule incrementally.
    """
    prod = 1.0
    keep = 0
    for p in step_probs:
        prod *= p
        keep += 1
        if _adaptive_cut(prod, threshold):
            break
    return max(1, keep)


def adaptive_running_survival_means(batch_step_probs):
    """Per-column batch mean of the per-request survival product.

    batch_step_probs: rectangular rows of per-request per-step top-probs.
    Column j of the result is mean_req prod(p_1..p_j) — the quantity the
    cutoff compares against the threshold after draft step j.
    """
    if not batch_step_probs:
        return []
    n = len(batch_step_probs)
    survival = [1.0] * n
    means = []
    for j in range(len(batch_step_probs[0])):
        survival = [s * row[j] for s, row in zip(survival, batch_step_probs)]
        means.append(sum(survival) / n)
    return means


def adaptive_batch_keep(batch_step_probs, threshold: float = 0.5) -> int:
    """Uniform chain length for a batch: keep the crossing token, floor 1."""
    means = adaptive_running_survival_means(batch_step_probs)
    keep = len(means)
    for j, m in enumerate(means):
        if _adaptive_cut(m, threshold):
            keep = j + 1
            break
    return max(1, keep)


# Injected verbatim into the speculator overlay, so the offline test
# exercises the exact code that runs in the container.
PURE_BLOCK = "\n".join(
    inspect.getsource(f)
    for f in (
        _adaptive_cut,
        adaptive_draft_length,
        adaptive_running_survival_means,
        adaptive_batch_keep,
    )
)

# ------------------------------------------------------------- speculator.py
SP_IMPORTS_OLD = "from typing import Any\n\nimport torch\nimport torch.nn as nn\n"
SP_IMPORTS_NEW = (
    "from typing import Any\n\nimport os\n\nimport torch\nimport torch.nn as nn\n"
)

SP_DIST_OLD = "from vllm.config.compilation import CUDAGraphMode\n"
SP_DIST_NEW = """\
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
"""

SP_MODULE_OLD = "logger = init_logger(__name__)\n"
SP_MODULE_NEW = (
    "logger = init_logger(__name__)\n\n\n"
    + """# ---------------------------------------------------------------------------
# Adaptive MTP draft depth (patches/patch_mtp_adaptive_depth.py). Opt-in:
# VLLM_MTP_ADAPTIVE_DEPTH=1, threshold VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD
# (default 0.5). Per-step, batch-uniform draft-chain cutoff on the draft
# head's own survival product; see the patcher docstring.
# ---------------------------------------------------------------------------
_MTP_ADAPTIVE_DEPTH = os.environ.get("VLLM_MTP_ADAPTIVE_DEPTH", "0") == "1"
_MTP_ADAPTIVE_DEPTH_THRESHOLD = float(
    os.environ.get("VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD", "0.5")
)
# Debug: VLLM_MTP_ADAPTIVE_DEBUG=1 logs the recorded per-step probs, the
# survival products, and the published width — first 20 finalize calls per
# boot, then every 200th. The first call also dumps the head config and a
# one-shot full-head vs reduced-head winner-mass comparison (the decisive
# check for reduced-denominator inflation).
_MTP_ADAPTIVE_DEBUG = os.environ.get("VLLM_MTP_ADAPTIVE_DEBUG", "0") == "1"


"""
    + PURE_BLOCK
    + "\n"
)

SP_INIT_OLD = """\
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False
"""
SP_INIT_NEW = """\
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False
        # Adaptive draft depth: per-step draft top-prob record, written from
        # inside (possibly graph-captured) draft sampling and read host-side
        # at the end of propose(). adaptive_num_draft_tokens is the proposal
        # width the model runner hands to the DraftTokensHandler; both stay
        # at num_speculative_steps unless the cutoff trims them.
        self._adaptive_top_probs = (
            torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            if _MTP_ADAPTIVE_DEPTH
            else None
        )
        self._adaptive_produced = self.num_speculative_steps
        self.adaptive_num_draft_tokens = self.num_speculative_steps
        self._adaptive_debug_calls = 0
        self._adaptive_debug_hidden = None
"""

SP_METHODS_OLD = (
    "    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...\n"
)
SP_METHODS_NEW = (
    SP_METHODS_OLD
    + """
    # ------------------------------------------------------------------
    # Adaptive draft depth. The cutoff is the module-level pure rule: per-
    # request survival product over the produced draft columns, batch mean,
    # strict < threshold; keep the crossing token, floor of one.
    # ------------------------------------------------------------------
    def _adaptive_record_top_probs(
        self, top_probs: torch.Tensor, draft_step: torch.Tensor
    ) -> None:
        # Column-indexed write into the persistent record; capture-safe, so
        # it also runs inside FULL draft graphs. draft_step is
        # self.current_draft_step at every call site.
        num_tokens = top_probs.shape[0]
        self._adaptive_top_probs[:num_tokens].scatter_(
            1,
            draft_step.view(1, 1).expand(num_tokens, 1),
            top_probs.unsqueeze(1),
        )

    def _adaptive_reduced_head_top_probs(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        \"\"\"Winner's softmax mass over the reduced/FP8 draft head.

        Ids stay the model's own get_top_tokens output (bit-identical to
        stock); this recomputes only the probability. The dropped rows' mass
        is missing from the denominator — an overestimate that truncates less
        than a full-vocab score, never more. Every TP rank builds the same
        value (gathered max, all-reduced denominator), so the group cuts
        together.
        \"\"\"
        model = self.model
        unwrap = getattr(model, "unwrap", None)
        if callable(unwrap):
            model = unwrap()
        fp8_head = getattr(model, "_draft_head_fp8", None)
        rows = getattr(model, "_draft_lm_head_weight", None)
        if fp8_head is None and rows is None:
            raise RuntimeError(
                "VLLM_MTP_ADAPTIVE_DEPTH: use_local_argmax_reduction is on but "
                "the draft model has neither _draft_lm_head_weight nor "
                "_draft_head_fp8 (provided by the mtp_draft_vocab / "
                "fp8_draft_head overlays); cannot score draft confidence"
            )
        scale = float(getattr(model.logits_processor, "scale", 1.0))
        if fp8_head is not None:
            logits = fp8_head.logits(hidden_states).float() * scale
        else:
            logits = (
                torch.nn.functional.linear(
                    hidden_states.to(rows.dtype), rows
                ).float()
                * scale
            )
        num_tokens = hidden_states.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        empty = logits.shape[1] == 0
        if empty:
            # This rank owns none of the draft vocabulary: losing bid, zero
            # denominator contribution. The collectives still run.
            local_max = torch.full(
                (num_tokens, 1),
                float("-inf"),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        else:
            local_max = logits.max(dim=-1, keepdim=True).values
        global_max = (
            tensor_model_parallel_all_gather(local_max, dim=-1)
            .max(dim=-1, keepdim=True)
            .values
            if tp_size > 1
            else local_max
        )
        if empty:
            sumexp = torch.zeros(
                num_tokens, dtype=torch.float32, device=hidden_states.device
            )
        else:
            sumexp = (logits - global_max).exp().sum(dim=-1)
        if tp_size > 1:
            sumexp = tensor_model_parallel_all_reduce(sumexp)
        return 1.0 / sumexp

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self._adaptive_top_probs is not None
            and self.use_local_argmax_reduction
        ):
            # Local-argmax drafting materializes no logits; score the winner
            # off the reduced/FP8 head. (vLLM rejects local argmax +
            # probabilistic drafting, so draft_logits is None here.)
            ids = self.model.get_top_tokens(hidden_states)
            self._adaptive_record_top_probs(
                self._adaptive_reduced_head_top_probs(hidden_states), draft_step
            )
            if _MTP_ADAPTIVE_DEBUG:
                # Keep the step's hidden states for the host-side full-head
                # probe in _adaptive_finalize (a device copy, capture-safe;
                # replay overwrites it each step, so finalize sees the last
                # step's states).
                self._adaptive_debug_hidden = hidden_states.detach().clone()
            return ids
        return super().sample_draft(
            hidden_states,
            sample_src_positions,
            idx_mapping,
            temperature,
            seeds,
            draft_step,
            draft_logits,
        )

    def _maybe_predict_acceptance(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
    ) -> None:
        super()._maybe_predict_acceptance(logits, idx_mapping, draft_step)
        if self._adaptive_top_probs is None:
            return
        if self.acceptance_estimator is not None:
            raise RuntimeError(
                "VLLM_MTP_ADAPTIVE_DEPTH is incompatible with "
                "enable_adaptive_verification (two confidence consumers)"
            )
        # Greedy full-vocab path: exact winner mass. Probabilistic drafting
        # is scored off the pre-temperature logits — an upper bound at
        # temperature > 0, so it truncates less (the safe direction).
        top_probs = logits.float().softmax(dim=-1).max(dim=-1).values
        self._adaptive_record_top_probs(top_probs, draft_step)

    def _adaptive_should_cut(self, num_reqs: int, step: int) -> bool:
        \"\"\"Stop the chain before drafting column `step`.

        Columns 0..step-1 hold this call's drafts; cut when the survival
        product after column step-1 has crossed the threshold.
        \"\"\"
        if self._adaptive_top_probs is None or self.dp_size > 1 or num_reqs < 1:
            return False
        if torch.cuda.is_current_stream_capturing():
            # Capture records the full fixed-length chain; the cutoff is
            # applied post-hoc to the replayed record in propose() instead.
            return False
        means = adaptive_running_survival_means(
            self._adaptive_top_probs[:num_reqs, :step]
            .to("cpu", torch.float64)
            .tolist()
        )
        if _adaptive_cut(means[-1], _MTP_ADAPTIVE_DEPTH_THRESHOLD):
            self._adaptive_produced = step
            return True
        return False

    def _adaptive_debug_probe_full_head(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        \"\"\"Winner's softmax mass over the FULL draft lm_head (debug only).

        Same collective pattern as _adaptive_reduced_head_top_probs but over
        this rank's full lm_head shard; compared against the reduced-head
        score it measures the dropped rows' missing denominator mass.
        \"\"\"
        model = self.model
        unwrap = getattr(model, "unwrap", None)
        if callable(unwrap):
            model = unwrap()
        weight = model.lm_head.weight
        scale = float(getattr(model.logits_processor, "scale", 1.0))
        logits = (
            torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
            .float()
            * scale
        )
        num_tokens = hidden_states.shape[0]
        local_max = logits.max(dim=-1, keepdim=True).values
        global_max = (
            tensor_model_parallel_all_gather(local_max, dim=-1)
            .max(dim=-1, keepdim=True)
            .values
            if get_tensor_model_parallel_world_size() > 1
            else local_max
        )
        sumexp = (logits - global_max).exp().sum(dim=-1)
        if get_tensor_model_parallel_world_size() > 1:
            sumexp = tensor_model_parallel_all_reduce(sumexp)
        return 1.0 / sumexp

    def _adaptive_debug_log(self, num_reqs: int, keep: int, rows) -> None:
        self._adaptive_debug_calls += 1
        n = self._adaptive_debug_calls
        if n == 1:
            model = self.model
            unwrap = getattr(model, "unwrap", None)
            if callable(unwrap):
                model = unwrap()
            fp8 = getattr(model, "_draft_head_fp8", None)
            head_rows = getattr(model, "_draft_lm_head_weight", None)
            logger.warning(
                "MTP_ADAPTIVE_DEBUG head: fp8=%s rows=%s scale=%s "
                "full_rows=%s",
                getattr(fp8, "kernel", fp8) if fp8 is not None else None,
                tuple(head_rows.shape) if head_rows is not None else None,
                getattr(model.logits_processor, "scale", None),
                tuple(model.lm_head.weight.shape),
            )
        if not (n <= 20 or n % 200 == 0):
            return
        means = [sum(col) / len(col) for col in zip(*rows)]
        mins = [min(col) for col in zip(*rows)]
        surv = adaptive_running_survival_means(rows)
        logger.warning(
            "MTP_ADAPTIVE_DEBUG call=%d reqs=%d produced=%d keep=%d thr=%.3f "
            "p_mean=%s p_min=%s surv=%s",
            n,
            num_reqs,
            self._adaptive_produced,
            keep,
            _MTP_ADAPTIVE_DEPTH_THRESHOLD,
            ["%.4f" % v for v in means],
            ["%.4f" % v for v in mins],
            ["%.4f" % v for v in surv],
        )
        if n <= 3 and self.use_local_argmax_reduction:
            hidden = self._adaptive_debug_hidden
            if hidden is not None and hidden.shape[0] >= num_reqs:
                p_full = self._adaptive_debug_probe_full_head(
                    hidden[:num_reqs]
                ).tolist()
                p_red = [row[-1] for row in rows]
                logger.warning(
                    "MTP_ADAPTIVE_DEBUG fullhead call=%d p_reduced_mean=%.4f "
                    "p_full_mean=%.4f missing_mass=%.4f",
                    n,
                    sum(p_red) / len(p_red),
                    sum(p_full) / len(p_full),
                    1.0 - sum(p_full) / max(sum(p_red), 1e-9),
                )

    def _adaptive_finalize(self, num_reqs: int) -> None:
        \"\"\"Publish this step's proposal width for the model runner.\"\"\"
        keep = self.num_speculative_steps
        rows = None
        if (
            self._adaptive_top_probs is not None
            and self.dp_size == 1
            and num_reqs > 0
        ):
            rows = (
                self._adaptive_top_probs[:num_reqs, : self._adaptive_produced]
                .to("cpu", torch.float64)
                .tolist()
            )
            keep = adaptive_batch_keep(rows, _MTP_ADAPTIVE_DEPTH_THRESHOLD)
        self.adaptive_num_draft_tokens = keep
        if _MTP_ADAPTIVE_DEBUG and rows:
            self._adaptive_debug_log(num_reqs, keep, rows)
"""
)

SP_MSD_LOOP_OLD = """\
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
"""
SP_MSD_LOOP_NEW = """\
        for step in range(1, self.num_speculative_steps):
            # Adaptive draft depth: stop the chain once the survival product
            # crosses the threshold. Host-driven path only — under FULL graph
            # replay each iteration is the same captured step graph, so an
            # early break replays a prefix of the same graphs.
            if self._adaptive_should_cut(num_reqs, step):
                break
            # Rebuild every step when positions advance, or just once
"""

SP_FUSED_LOOP_OLD = """\
        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
"""
SP_FUSED_LOOP_NEW = """\
        for step in range(1, self.num_speculative_steps):
            # Adaptive draft depth: see _multi_step_decode. When this whole
            # loop is captured as one graph (fused FULL mode) the check
            # stays out of the capture and propose() trims post-hoc.
            if self._adaptive_should_cut(num_reqs, step):
                break
            self.current_draft_step.fill_(step)
            self._generate_draft(
"""

SP_PROPOSE_MID_OLD = """\
        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
"""
SP_PROPOSE_MID_NEW = """\
        # Adaptive draft depth: full chain unless a host-driven loop below
        # cuts it short (a fused FULL graph records all steps; the width is
        # then trimmed post-hoc in _adaptive_finalize).
        self._adaptive_produced = self.num_speculative_steps
        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
"""

SP_PROPOSE_TAIL_OLD = """\
        self.on_multi_step_decode_end(num_reqs)

        return self.draft_tokens[:num_reqs]
"""
SP_PROPOSE_TAIL_NEW = """\
        self.on_multi_step_decode_end(num_reqs)

        # Adaptive draft depth: publish the (possibly truncated) proposal
        # width. The returned buffer keeps full width — req_states.draft_
        # tokens is fixed [max_num_reqs, k]; the runner slices the columns.
        self._adaptive_finalize(num_reqs)

        return self.draft_tokens[:num_reqs]
"""

SPECULATOR_HUNKS = (
    (SP_IMPORTS_OLD, SP_IMPORTS_NEW),
    (SP_DIST_OLD, SP_DIST_NEW),
    (SP_MODULE_OLD, SP_MODULE_NEW),
    (SP_INIT_OLD, SP_INIT_NEW),
    (SP_METHODS_OLD, SP_METHODS_NEW),
    (SP_MSD_LOOP_OLD, SP_MSD_LOOP_NEW),
    (SP_FUSED_LOOP_OLD, SP_FUSED_LOOP_NEW),
    (SP_PROPOSE_MID_OLD, SP_PROPOSE_MID_NEW),
    (SP_PROPOSE_TAIL_OLD, SP_PROPOSE_TAIL_NEW),
)

# ------------------------------------------------------------ model_runner.py
MR_IMPORT_OLD = "import gc\n"
MR_IMPORT_NEW = "import gc\nimport os\n"

MR_MODULE_OLD = "logger = init_logger(__name__)\n"
MR_MODULE_NEW = """\
logger = init_logger(__name__)

# Adaptive MTP draft depth debug (patch_mtp_adaptive_depth.py): with
# VLLM_MTP_ADAPTIVE_DEBUG=1, log the proposal width at the two handoff points
# (set_draft_tokens / take_draft_token_ids) — first 20 calls, then every 200th.
_MTP_ADAPTIVE_HANDOFF_DEBUG = os.environ.get("VLLM_MTP_ADAPTIVE_DEBUG", "0") == "1"
_MTP_ADAPTIVE_HANDOFF_CALLS = [0]


def _mtp_adaptive_handoff_log(where: str, num_proposed, width: int) -> None:
    _MTP_ADAPTIVE_HANDOFF_CALLS[0] += 1
    n = _MTP_ADAPTIVE_HANDOFF_CALLS[0]
    if n <= 20 or n % 200 == 0:
        logger.warning(
            "MTP_ADAPTIVE_DEBUG %s call=%d num_proposed=%s width=%d",
            where,
            n,
            num_proposed,
            width,
        )
"""

MR_HANDOFF_OLD = """\
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )
"""
MR_HANDOFF_NEW = """\
            # Adaptive MTP draft depth (patch_mtp_adaptive_depth.py): the
            # speculator may have truncated this step's draft chain; only the
            # first adaptive_num_draft_tokens columns are proposals. Other
            # speculator types lack the attribute and pass through full width.
            num_proposed_drafts = getattr(
                self.speculator, "adaptive_num_draft_tokens", None
            )
            draft_tokens_for_handler = self.req_states.draft_tokens[
                input_batch.idx_mapping
            ]
            if num_proposed_drafts is not None:
                draft_tokens_for_handler = draft_tokens_for_handler[
                    :, :num_proposed_drafts
                ]
            if _MTP_ADAPTIVE_HANDOFF_DEBUG:
                _mtp_adaptive_handoff_log(
                    "handoff",
                    num_proposed_drafts,
                    draft_tokens_for_handler.shape[1],
                )
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                draft_tokens_for_handler,
            )
"""

MR_TAKE_OLD = """\
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()
"""
MR_TAKE_NEW = """\
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if _MTP_ADAPTIVE_HANDOFF_DEBUG:
            _mtp_adaptive_handoff_log(
                "take", None, self.draft_tokens_handler.num_draft_tokens
            )
        return self.draft_tokens_handler.get_draft_tokens()
"""

RUNNER_HUNKS = (
    (MR_IMPORT_OLD, MR_IMPORT_NEW),
    (MR_MODULE_OLD, MR_MODULE_NEW),
    (MR_HANDOFF_OLD, MR_HANDOFF_NEW),
    (MR_TAKE_OLD, MR_TAKE_NEW),
)


def _apply(src: str, hunks, name: str, guard: str) -> str:
    if guard in src:
        sys.exit(f"ERROR: mtp_adaptive_depth {name} orig is already patched")
    for i, (old, new) in enumerate(hunks):
        count = src.count(old)
        if count != 1:
            sys.exit(f"mtp_adaptive_depth: anchor {i} in {name} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"mtp_adaptive_depth: patched {name} does not parse: {exc}")
    return src


def _patch(orig_dir: str, out_dir: str, stem: str, hunks, guard: str) -> None:
    orig = os.path.join(orig_dir, f"{stem}.py")
    if not os.path.isfile(orig):
        sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
    src = _apply(open(orig).read(), hunks, f"{stem}.py", guard)
    out = os.path.join(out_dir, f"{stem}_v030.py")
    os.makedirs(out_dir, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(src)
    os.replace(out + ".tmp", out)
    print(f"patched {out}")


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "mtp_adaptive_depth", "orig")
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, "mtp_adaptive_depth")
    _patch(orig_dir, out_dir, "speculator", SPECULATOR_HUNKS, "_adaptive_top_probs")
    _patch(orig_dir, out_dir, "model_runner", RUNNER_HUNKS, "adaptive_num_draft_tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
