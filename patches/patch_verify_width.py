#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adaptive speculative-verify width for the vLLM v0.32.0 scheduler
(opt-in: verify_width_adaptive / VLLM_VERIFY_WIDTH_ADAPTIVE).

Target: vllm/v1/core/sched/scheduler.py. The scheduler owns the one scalar
that decides how many draft positions a batch verifies
(num_spec_tokens_to_schedule), pads fresh decode requests to the same width
(1 + k rows under pad_spec_decode), and fills their placeholder draft list
([-1] * k). Stock always takes speculative_config.num_speculative_tokens for
that scalar. This overlay keeps the width batch-uniform but lets it follow the
acceptance rate measured by update_from_output:

    ratio < 0.35 -> k = 1
    ratio < 0.60 -> k = 2
    else         -> k = config k        (never above config k)

The ratio is an exponential moving average (alpha 0.25) of the round's
aggregate accepted / max(1, drafted) draft rows, summed over every request in
the batch; a round with no draft rows carries no signal and is skipped. Width
changes are behind a dwell: at most one change every 4 schedule() calls,
starting from config k. The dwell is the hysteresis — one noisy round cannot
move the width, and every move is held for four scheduling steps. The chosen
width is used in the padding width, the padding consistency check (the Mamba
alignment guard), the placeholder fill and its assert, and the published
num_spec_tokens_to_schedule, so all of them agree within a step.

Lossless: only the number of speculative verify rows changes. The target
model's verification of whatever drafts exist is untouched, and rejected
drafts are discarded exactly as stock does; a narrower width simply proposes
fewer draft positions. Under async scheduling the published width lands in the
next step's proposal (stock behavior: the scheduler publishes the width, the
worker proposes against it), so a change takes effect one step later.

Stand-downs, checked once at scheduler construction:
  - VLLM_VERIFY_WIDTH_ADAPTIVE unset -> the scalar is config k at every site;
    the only cost is one bool check per schedule()/update_from_output call
    and no EMA bookkeeping;
  - data parallelism > 1 -> config k (DP ranks must agree on the width);
  - speculative_config.num_speculative_tokens_per_batch_size set -> config k,
    because stock already turns that table into self.dynamic_sd_lookup and
    the table owns the published width; the two knobs are not combined.

The hard ceiling is stock: the model runner captures its decode graph at
num_speculative_tokens + 1 query rows, so a chosen width is capped at config k
and the V2 runner already trims drafts to the scheduled width when it is
smaller.

Patch time is fail-closed too: every anchor must match exactly once, the
result must parse, and nothing is written on any mismatch.

Inputs:  <out_dir>/orig/scheduler.py (extracted from the pinned image by
         engine/patches.sh)
Outputs: <out_dir>/scheduler_v030.py
argv[0] overrides the output directory (used by the test).
"""
import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_MARK = "_verify_width_init"


def _verify_width_update_ema(ema, accepted, drafted, alpha=0.25):
    """One EMA step from a round's aggregate (accepted, drafted) counts."""
    ratio = accepted / max(1, drafted)
    if ema is None:
        return ratio
    return alpha * ratio + (1.0 - alpha) * ema


def _verify_width_target(ema, config_k):
    """Draft width for an acceptance EMA: 1 / 2 / config k, capped at config."""
    if config_k <= 0 or ema is None:
        return config_k
    if ema < 0.35:
        return min(1, config_k)
    if ema < 0.60:
        return min(2, config_k)
    return config_k


def _verify_width_next(ema, config_k, current_k, steps_since_change, dwell=4):
    """Width for this step: hold current_k until the dwell has expired."""
    if config_k <= 0:
        return config_k
    if steps_since_change < dwell:
        return current_k
    return _verify_width_target(ema, config_k)


_STATE_HELPERS = '''
def _verify_width_init(sched) -> None:
    """Arm the policy; DP>1 or a stock per-batch-size table stands it down."""
    enabled = _VERIFY_WIDTH_ON
    reason = ""
    if not enabled:
        reason = "env off"
    elif sched.num_spec_tokens <= 0:
        reason = "no speculative tokens"
    elif sched.parallel_config.data_parallel_size > 1:
        reason = "data parallel size > 1"
    elif sched.dynamic_sd_lookup is not None:
        reason = "num_speculative_tokens_per_batch_size is set"
    sched._vw_active = not reason
    sched._vw_off_reason = reason
    sched._vw_k = sched.num_spec_tokens
    sched._vw_ema = None
    sched._vw_calls = 0
    sched._vw_last_change = 0
    sched._vw_accepted = 0
    sched._vw_drafted = 0
    if enabled and reason and reason != "no speculative tokens":
        logger.warning(
            "verify width: %s; staying at config k=%d.",
            reason,
            sched.num_spec_tokens,
        )


def _verify_width_step(sched) -> int:
    """Width for this schedule() call; config k whenever the policy is off."""
    if not sched._vw_active:
        return sched.num_spec_tokens
    sched._vw_calls += 1
    picked = _verify_width_next(
        sched._vw_ema,
        sched.num_spec_tokens,
        sched._vw_k,
        sched._vw_calls - sched._vw_last_change,
    )
    if picked != sched._vw_k:
        sched._vw_k = picked
        sched._vw_last_change = sched._vw_calls
    return sched._vw_k


def _verify_width_finish_round(sched) -> None:
    """Fold one update_from_output round into the acceptance EMA."""
    if not sched._vw_active:
        return
    if sched._vw_drafted > 0:
        sched._vw_ema = _verify_width_update_ema(
            sched._vw_ema, sched._vw_accepted, sched._vw_drafted
        )
    sched._vw_accepted = 0
    sched._vw_drafted = 0
'''


def helper_block() -> str:
    """Policy + state helpers inserted before the Scheduler class."""
    pure = "\n\n\n".join(
        inspect.getsource(fn).strip("\n")
        for fn in (
            _verify_width_update_ema,
            _verify_width_target,
            _verify_width_next,
        )
    )
    return (
        "# --- adaptive speculative-verify width (VLLM_VERIFY_WIDTH_ADAPTIVE) ---\n"
        "# patches/patch_verify_width.py. The scheduler publishes one verify\n"
        "# width per step; this policy follows the batch acceptance EMA\n"
        "# (<0.35 -> 1, <0.60 -> 2, else config k) with a four-step dwell.\n"
        "# Only the number of verify rows changes, never target verification.\n"
        "# Inert unless armed via the recipe key / env gate.\n"
        "import os\n"
        "\n\n"
        "_VERIFY_WIDTH_ON = (\n"
        '    os.environ.get("VLLM_VERIFY_WIDTH_ADAPTIVE", "").strip() == "1"\n'
        ")\n"
        "\n\n"
        + pure
        + "\n\n\n"
        + _STATE_HELPERS.strip("\n")
        + "\n\n\n"
    )


CLASS_OLD = "class Scheduler(SchedulerInterface):\n"
CLASS_NEW = helper_block() + CLASS_OLD

INIT_OLD = """\
        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
"""
INIT_NEW = """\
        _verify_width_init(self)

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
"""

SCHEDULE_OLD = """\
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
"""
SCHEDULE_NEW = """\
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        _verify_width_k = _verify_width_step(self)
"""

PAD_OLD = """\
                        padded_num_tokens = 1 + self.num_spec_tokens
"""
PAD_NEW = """\
                        padded_num_tokens = 1 + _verify_width_k
"""

MAMBA_OLD = """\
                        if (
                            pad_spec_decode
                            and num_new_tokens != 1 + self.num_spec_tokens
                        ):
"""
MAMBA_NEW = """\
                        if (
                            pad_spec_decode
                            and num_new_tokens != 1 + _verify_width_k
                        ):
"""

FILL_OLD = """\
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
"""
FILL_NEW = """\
                if pad_spec_decode:
                    assert num_new_tokens == 1 + _verify_width_k
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * _verify_width_k
"""

PUBLISH_OLD = """\
        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
"""
PUBLISH_NEW = """\
        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = _verify_width_k
"""

ACCEPT_OLD = """\
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
"""
ACCEPT_NEW = """\
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                if self._vw_active:
                    self._vw_accepted += num_accepted
                    self._vw_drafted += num_draft_tokens
"""

END_OLD = """\
            eco.scheduler_stats = stats

        return engine_core_outputs
"""
END_NEW = """\
            eco.scheduler_stats = stats

        _verify_width_finish_round(self)
        return engine_core_outputs
"""

EDITS = (
    (CLASS_OLD, CLASS_NEW),
    (INIT_OLD, INIT_NEW),
    (SCHEDULE_OLD, SCHEDULE_NEW),
    (PAD_OLD, PAD_NEW),
    (MAMBA_OLD, MAMBA_NEW),
    (FILL_OLD, FILL_NEW),
    (PUBLISH_OLD, PUBLISH_NEW),
    (ACCEPT_OLD, ACCEPT_NEW),
    (END_OLD, END_NEW),
)


def patch(src: str) -> str:
    if PATCH_MARK in src:
        sys.exit("ERROR: verify_width orig scheduler.py is already patched")
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"verify_width: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"verify_width: patched scheduler_v030.py does not parse: {exc}")
    return src


def main(argv) -> int:
    out_dir = argv[0] if argv else os.path.join(HERE, "verify_width")
    orig_path = os.path.join(out_dir, "orig", "scheduler.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(engine/patches.sh extracts it from the pinned image)")
    src = patch(open(orig_path).read())
    out_path = os.path.join(out_dir, "scheduler_v030.py")
    with open(out_path + ".tmp", "w") as f:
        f.write(src)
    os.replace(out_path + ".tmp", out_path)
    print("patched scheduler_v030.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
