#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Env-gated speculative-decode round trace for the vLLM v0.32.0 lane.

Measurement only: no modeling change, no output change. When VLLM_SPEC_TRACE
is unset (default) nothing is opened, timed, or synchronized — the worker
pays one string check per sample step.

Target: vllm/v1/worker/gpu/model_runner.py, GPUModelRunner.sample_tokens.
The V2 runner verifies the drafts the scheduler scheduled for the step and
then proposes the next round's drafts, all host-side (no captured graph). The
trace stamps a perf_counter at the end of each call and, when the step
scheduled drafts, appends one record there — after the proposal, where both
halves of the round are known:

  alloc       draft positions the scheduler actually scheduled for the round
              (InputBatch.num_draft_tokens — not the draft graph width)
  accepted    alloc - rejected (num_rejected from verification)
  ms          wall time since the previous sample_tokens boundary (the
              engine step that produced and verified these drafts); null on
              the first traced round of a process, and idle gaps between
              requests show up in the first round after them
  round       1-based counter per worker process
  reqs        requests in the batch
  alloc_req   per-request scheduled drafts
  accepted_req  per-request accepted drafts
  rank        torch.distributed rank when initialized

The only device synchronization the trace adds is one D2H read of
num_rejected per traced round. The output file is opened lazily in append
mode and flushed per line, so a crash keeps every completed line. A failed
open disables the trace after one warning instead of failing serving.

Inputs:  <out_dir>/orig/model_runner.py (extracted from the pinned image)
Outputs: <out_dir>/model_runner_v030.py
argv[0] overrides the output directory (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Injected after `logger = init_logger(__name__)`. Kept as a module constant
# so the offline test executes the exact helper code that runs in the
# container.
MODULE_BLOCK = '''
# ---------------------------------------------------------------------------
# Speculative-decode round trace (patches/patch_spec_trace.py). Off unless
# VLLM_SPEC_TRACE names a JSONL file. Enabled: one record per draft/verify
# round at the end of sample_tokens, flushed per line; ms is the wall time
# since the previous sample_tokens boundary. alloc is the number of
# draft positions the scheduler actually scheduled for the round (not the
# draft graph width); accepted = alloc - rejected. The trace's only device
# sync is the D2H read of num_rejected on a traced round.
# ---------------------------------------------------------------------------
_SPEC_TRACE_PATH = os.environ.get("VLLM_SPEC_TRACE", "")
# round, last perf_counter, file handle, rank, failed-open flag
_SPEC_TRACE_STATE = [0, None, None, None, False]


def _spec_trace_mark() -> None:
    import time

    _SPEC_TRACE_STATE[1] = time.perf_counter()


def _spec_trace_emit(input_batch, num_rejected) -> None:
    if not _SPEC_TRACE_PATH or num_rejected is None or _SPEC_TRACE_STATE[4]:
        return
    import json
    import time

    if _SPEC_TRACE_STATE[2] is None:
        try:
            _SPEC_TRACE_STATE[2] = open(_SPEC_TRACE_PATH, "a")
        except OSError as exc:
            _SPEC_TRACE_STATE[4] = True
            logger.warning(
                "VLLM_SPEC_TRACE: cannot open %s: %s", _SPEC_TRACE_PATH, exc
            )
            return
    if _SPEC_TRACE_STATE[3] is None:
        rank = 0
        try:
            from torch import distributed as dist
        except ImportError:
            dist = None
        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        _SPEC_TRACE_STATE[3] = rank
    num_reqs = input_batch.num_reqs
    # The trace's one D2H read: per-request rejected draft counts.
    rejected = [int(x) for x in num_rejected[:num_reqs].tolist()]
    alloc_req = input_batch.num_draft_tokens_per_req
    alloc_req = (
        [int(x) for x in alloc_req[:num_reqs]] if alloc_req is not None else []
    )
    accepted_req = (
        [a - r for a, r in zip(alloc_req, rejected)] if alloc_req else []
    )
    alloc = int(input_batch.num_draft_tokens)
    accepted = sum(accepted_req) if accepted_req else alloc - sum(rejected)
    mark = _SPEC_TRACE_STATE[1]
    _SPEC_TRACE_STATE[0] += 1
    record = {
        "round": _SPEC_TRACE_STATE[0],
        "ms": (
            round((time.perf_counter() - mark) * 1000.0, 3)
            if mark is not None
            else None
        ),
        "alloc": alloc,
        "accepted": accepted,
        "reqs": num_reqs,
        "alloc_req": alloc_req,
        "accepted_req": accepted_req,
        "rank": _SPEC_TRACE_STATE[3],
    }
    _SPEC_TRACE_STATE[2].write(json.dumps(record, separators=(",", ":")) + "\\n")
    _SPEC_TRACE_STATE[2].flush()
'''

IMPORT_OLD = "import gc\n"
IMPORT_NEW = "import gc\nimport os\n"

MODULE_OLD = "logger = init_logger(__name__)\n"
MODULE_NEW = "logger = init_logger(__name__)\n\n\n" + MODULE_BLOCK.strip("\n") + "\n"

EMIT_OLD = """\
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        return async_output
"""
EMIT_NEW = """\
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        # Speculative-decode round trace: the target's verdict on the drafts
        # scheduled for this round is known and the next round's drafts were
        # just proposed. One record per round that scheduled drafts; the
        # boundary below feeds the next round's wall time. Unset
        # VLLM_SPEC_TRACE means one string check here: no file, no sync.
        if _SPEC_TRACE_PATH and input_batch.num_draft_tokens:
            _spec_trace_emit(input_batch, num_rejected)
        if _SPEC_TRACE_PATH:
            _spec_trace_mark()

        return async_output
"""

EDITS = (
    (IMPORT_OLD, IMPORT_NEW),
    (MODULE_OLD, MODULE_NEW),
    (EMIT_OLD, EMIT_NEW),
)


def patch(src: str) -> str:
    if "_spec_trace_emit" in src:
        sys.exit("ERROR: spec_trace orig model_runner.py is already patched")
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"spec_trace: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"spec_trace: patched model_runner_v030.py does not parse: {exc}")
    return src


def main(argv) -> int:
    out_dir = argv[0] if argv else os.path.join(HERE, "spec_trace")
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
