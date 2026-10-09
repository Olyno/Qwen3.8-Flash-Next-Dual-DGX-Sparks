#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Adaptive MTP draft depth for the vLLM 0.30.0 lane (opt-in,
MTP_ADAPTIVE_DEPTH=true).

Patches vllm/v1/spec_decode/llm_base_proposer.py: after each draft step the
running product of the batch-mean top-token softmax probability (the draft
head's own survival score, DSpark/SVIP-style) is compared against
VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD (default 0.5, Strata's spec-min-p operating
point); once it drops below, the chain stops early. Training-free and exact:
target verification is untouched, truncated drafts are simply never proposed.

k is decided per STEP, uniform across the batch. The draft loop replays one
cudagraph per iteration, dispatched on the (loop-invariant) decode batch
size, so stopping early replays a prefix of the same captured graphs —
FULL_DECODE_ONLY capture and the padded drafter batch are untouched, and no
per-sequence padding is needed (a per-sequence k would break exactly that).
The runner already carries a narrower [batch, k'] draft output to the
scheduler (prev_num_spec_tokens / DraftTokenIds), which verifies k' tokens
next step. Disabled under data parallelism (DP ranks would desync the break);
TP ranks share the batch and the survival product is built from all-reduced
values, so every rank breaks identically.

Confidence source: greedy ids stay bit-identical to _greedy_sample's (same
local argmax, same (value, index) all-gather), so threshold 0 reproduces the
stock stream. The prob is the winner's softmax mass over the head that
produced it — the reduced draft-vocab slice or its FP8 copy when
MTP_DRAFT_VOCAB/FP8_DRAFT_HEAD is on (one extra [batch] all-reduce per draft
step; the dropped rows' mass is missing from the denominator, an overestimate
that truncates less than a full-vocab score, never more), the gathered full
head otherwise. k=1 and parallel drafting are no-ops (floor of 1 token). The
cutoff check is one [1] D2H sync per draft step.

Runtime gate: VLLM_MTP_ADAPTIVE_DEPTH=1 plus VLLM_MTP_ADAPTIVE_DEPTH_THRESHOLD,
both passed via OVERLAY_ENV (engine/patches.sh). Requirements enforced there:
v030 lane, MTP_NUM_SPECULATIVE_TOKENS > 1.

Inputs:  patches/mtp_adaptive_depth/orig/llm_base_proposer.py (from the image)
Outputs: patches/mtp_adaptive_depth/llm_base_proposer_v030.py
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
    The proposer loop applies the same rule incrementally (_adaptive_cut on
    the batch-mean survival product), so this is the offline-testable
    statement of the cutoff.
    """
    prod = 1.0
    keep = 0
    for p in step_probs:
        prod *= p
        keep += 1
        if _adaptive_cut(prod, threshold):
            break
    return max(1, keep)


# Injected verbatim into the proposer overlay, so the offline test exercises
# the exact code that runs in the container.
PURE_BLOCK = inspect.getsource(_adaptive_cut) + "\n" + inspect.getsource(adaptive_draft_length)

IMPORTS_OLD = "import dataclasses\nfrom importlib.util import find_spec\n"
IMPORTS_NEW = "import dataclasses\nimport os\nfrom importlib.util import find_spec\n"

DIST_OLD = "from vllm.distributed.parallel_state import get_pp_group\n"
DIST_NEW = """\
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import get_pp_group
"""

MODULE_OLD = "logger = init_logger(__name__)\n\n\nclass SpecDecodeBaseProposer:"
MODULE_NEW = (
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


"""
    + PURE_BLOCK
    + "\nclass SpecDecodeBaseProposer:"
)

METHOD_OLD = """\
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        \"\"\"Greedy-sample draft tokens from hidden states.\"\"\"
"""
METHOD_NEW = """\
    def _adaptive_draft_sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        \"\"\"(draft_token_ids, draft_probs, top_probs) for the adaptive cutoff.

        Greedy ids are bit-identical to _greedy_sample's (same local argmax,
        same (value, index) all-gather), so threshold 0 reproduces the stock
        stream exactly. top_probs is the winner's softmax mass over the head
        that produced it; with a reduced draft vocab the dropped rows' mass
        is missing from the denominator, an overestimate, so the cutoff
        truncates less than a full-vocab score would, never more.
        \"\"\"
        if (
            self._enable_probabilistic_draft_probs
            and not sampling_metadata.all_greedy
        ):
            ids, draft_probs = self._sample_draft_tokens(
                hidden_states, sampling_metadata
            )
            assert draft_probs is not None
            top_probs = draft_probs.gather(-1, ids[:, None].long()).squeeze(-1)
            return ids, draft_probs, top_probs

        model = self.model
        if isinstance(model, BreakableCUDAGraphWrapper):
            model = model.unwrap()
        fp8_head = getattr(model, "_draft_head_fp8", None)
        rows = getattr(model, "_draft_lm_head_weight", None)
        id_map = getattr(model, "_draft_id_to_target_id", None)
        num_tokens = hidden_states.shape[0]

        if (fp8_head is None and rows is None) or id_map is None:
            # Stock full-vocab head; compute_logits applies the logit scale /
            # soft cap and gathers across TP.
            logits = self.model.compute_logits(hidden_states)
            if self.use_heterogeneous_vocab:
                assert self.vocab_mapping is not None
                logits = self.vocab_mapping.constrain_draft_logits(logits)
            top_probs, ids = logits.float().softmax(dim=-1).max(dim=-1)
            if self.use_heterogeneous_vocab:
                ids = self.vocab_mapping.map_draft_to_target_ids(ids)
            return ids, None, top_probs

        # Reduced draft-vocab slice or its FP8 copy: per-rank local logits
        # (unscaled, so apply the logit scale the stock head would).
        scale = float(getattr(model.logits_processor, "scale", 1.0))
        if fp8_head is not None:
            logits = fp8_head.logits(hidden_states).float() * scale
        else:
            logits = (
                torch.nn.functional.linear(hidden_states.to(rows.dtype), rows).float()
                * scale
            )
        if logits.shape[1] == 0:
            # This rank owns none of the draft vocabulary: losing bid.
            local_max_vals = torch.full(
                (num_tokens,),
                float("-inf"),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            global_indices = torch.zeros(
                (num_tokens,), dtype=torch.long, device=hidden_states.device
            )
        else:
            local_max_vals, local_max_indices = logits.max(dim=-1)
            global_indices = id_map[local_max_indices]

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            denom = (logits - local_max_vals[:, None]).exp().sum(dim=-1)
            return global_indices.to(torch.int64), None, 1.0 / denom

        # Same winner reduction as LogitsProcessor.get_top_tokens, then the
        # softmax denominator over the whole slice: one [batch] all-reduce.
        local_pair = torch.stack([local_max_vals, global_indices.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(num_tokens, tp_size, 2)
        winner = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        ids = gathered[:, :, 1].gather(dim=-1, index=winner)
        global_max = gathered[:, :, 0].gather(dim=-1, index=winner)
        if logits.shape[1] == 0:
            sumexp = torch.zeros(
                (num_tokens,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            sumexp = (logits - global_max).exp().sum(dim=-1)
        denom = tensor_model_parallel_all_reduce(sumexp)
        return ids.squeeze(-1).to(torch.int64), None, 1.0 / denom

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        \"\"\"Greedy-sample draft tokens from hidden states.\"\"\"
"""

FIRST_SAMPLE_OLD = """\
        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]
"""
FIRST_SAMPLE_NEW = """\
        adaptive_depth = _MTP_ADAPTIVE_DEPTH and (
            self.vllm_config.parallel_config.data_parallel_size == 1
        )
        if adaptive_depth:
            draft_token_ids, draft_probs, top_probs = self._adaptive_draft_sample(
                sample_hidden_states, sampling_metadata
            )
            survival = top_probs
        else:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
        draft_probs_list = None if draft_probs is None else [draft_probs]
"""

LOOP_HEAD_OLD = """\
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
"""
LOOP_HEAD_NEW = """\
        for token_index in range(self.num_speculative_tokens - 1):
            # Adaptive depth: one [1] D2H sync per draft step on the batch-mean
            # survival product. The product is built from all-reduced values,
            # identical on every TP rank, so the whole group breaks together.
            if adaptive_depth and _adaptive_cut(
                float(survival.mean()), _MTP_ADAPTIVE_DEPTH_THRESHOLD
            ):
                break
            # Update the inputs.
"""

LOOP_SAMPLE_OLD = """\
            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata
            )
            if draft_probs is not None:
"""
LOOP_SAMPLE_NEW = """\
            hidden_states = hidden_states[:batch_size]
            if adaptive_depth:
                draft_token_ids, draft_probs, top_probs = (
                    self._adaptive_draft_sample(
                        last_hidden_states[:batch_size], sampling_metadata
                    )
                )
                survival = survival * top_probs
            else:
                draft_token_ids, draft_probs = self._sample_draft_tokens(
                    last_hidden_states[:batch_size], sampling_metadata
                )
            if draft_probs is not None:
"""

HUNKS = (
    (IMPORTS_OLD, IMPORTS_NEW),
    (DIST_OLD, DIST_NEW),
    (MODULE_OLD, MODULE_NEW),
    (METHOD_OLD, METHOD_NEW),
    (FIRST_SAMPLE_OLD, FIRST_SAMPLE_NEW),
    (LOOP_HEAD_OLD, LOOP_HEAD_NEW),
    (LOOP_SAMPLE_OLD, LOOP_SAMPLE_NEW),
)


def _apply(src: str) -> str:
    if "_adaptive_draft_sample" in src:
        sys.exit("ERROR: mtp_adaptive_depth orig is already patched")
    for i, (old, new) in enumerate(HUNKS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"mtp_adaptive_depth: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    return src


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "mtp_adaptive_depth", "orig")
    out_dir = argv[1] if len(argv) > 1 else os.path.join(HERE, "mtp_adaptive_depth")
    orig = os.path.join(orig_dir, "llm_base_proposer.py")
    if not os.path.isfile(orig):
        sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
    src = _apply(open(orig).read())
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"mtp_adaptive_depth: patched llm_base_proposer_v030.py does not parse: {exc}")
    out = os.path.join(out_dir, "llm_base_proposer_v030.py")
    os.makedirs(out_dir, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(src)
    os.replace(out + ".tmp", out)
    print(f"patched {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
