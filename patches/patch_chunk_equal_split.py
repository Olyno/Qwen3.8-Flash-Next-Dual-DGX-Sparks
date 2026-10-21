#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Equal-width chunked prefill for the v0.32 lane (opt-in: chunk_equal_split /
VLLM_CHUNK_EQUAL_SPLIT).

Idea-level credit, no code copied: Niko1221/Strata (MIT), chunk policy #41 as
changed after its 0.1.38 — one chunk if it fits, else equal chunks, no stub
tail. Strata enforces that inside its own C++ scheduler; this is an
independent Python implementation of the stated policy against vLLM's v1
scheduler, written from the policy and not from its source.

Stock v0.32 caps one prefill chunk at the configured
long_prefill_token_threshold and then takes everything below the cap, so a
prompt just past the cap is left with a stub-tail chunk. With the prod cap of
4800 a 5,000-token prompt is split 4800+200: both splits need two steps, but
the second step carries 200 rows while still paying the fixed cost of a step
(KV block allocation, attention-metadata build, the Mamba/GDN state
checkpoint, the MTP drafter's metadata pass), and the prompt's next chunk would
re-read the prefix state that stub step wrote. Splitting into
k = ceil(remaining / cap) chunks of ceil(remaining / k) rows keeps the cap and
the step count identical and makes the tail as full as the head — 2500+2500 for
that prompt. Chunk count is never worse than stock's (both hit the
ceil(remaining / cap) minimum); what goes away is the near-empty step.

Both clamp sites get the policy: the running-request one (an in-flight prefill
chunk) and the waiting one (where a fresh long prompt takes its first cut, and
the site that actually produces the 4800+200 split). Downstream clamps are
untouched, so the token budget, the Mamba block-aligned split and the prefill
lookahead reserve still win over the equal split when they apply — under GDN
align mode the first chunk is floored to a block boundary and the residue lands
in the next chunk, still without a stub.

Fail closed, both ends:
  - runtime: with VLLM_CHUNK_EQUAL_SPLIT unset the patched scheduler takes the
    stock branch verbatim (the gate is read once at module import; the call
    sites differ from stock only by the wrapped ternary), so the overlay is
    inert until engine/patches.sh arms it via the recipe key;
  - patch time: the env/gate block and both clamp sites must each match exactly
    once and the result must parse, or nothing is written.

--self-check verifies the policy without a boot or an extracted orig: the
documented cases, a grid over caps and totals (no stub, spread <= 1 row,
nothing above the cap, step count == ceil(total / cap)), and that the generated
runtime helper agrees with the reference policy. The real verification is a
boot with chunk_equal_split: true and bench/longctx.py TTFT at two or more
concurrent prompts — a single eligible request already bypasses the cap (stock
zeroes the threshold when nothing else wants the budget), so a lone prompt
looks unchanged either way.

Inputs:  patches/chunk_equal_split/orig/scheduler.py — vllm/v1/core/sched/
         scheduler.py from the pinned image (engine/patches.sh extracts it)
Outputs: patches/chunk_equal_split/scheduler_v032.py
"""
import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_ENV = "VLLM_CHUNK_EQUAL_SPLIT"
GATE_VAR = "_CHUNK_EQUAL_SPLIT"
HELPER = "_chunk_equal_split"


def chunk_rows(remaining: int, cap: int) -> int:
    """Rows to run this step for a prefill with `remaining` rows left under a
    per-step cap of `cap`: one chunk if it fits, else ceil(remaining / k) with
    k = ceil(remaining / cap) equal chunks, so no chunk is a stub tail.

    Pure integer math. The result never exceeds `cap` (k >= remaining / cap)
    and never exceeds `remaining`.
    """
    if cap <= 0 or remaining <= cap:
        return remaining
    k = -(-remaining // cap)
    return -(-remaining // k)


def plan_chunks(total: int, cap: int) -> list[int]:
    """Every chunk the policy produces for a `total`-row prompt.

    The scheduler re-derives the split each step from what is left, so replaying
    chunk_rows is exactly what it does; the last chunk always fits and closes
    the prompt. Test-side view of the policy, not shipped into the image file.
    """
    plan, left = [], total
    while left > 0:
        n = chunk_rows(left, cap)
        assert 0 < n <= left, (total, cap, left, n)
        plan.append(n)
        left -= n
    return plan


def helper_block() -> str:
    """Gate + policy function inserted into scheduler.py. The function body is
    generated from chunk_rows so the overlay and the patcher cannot drift."""
    src = inspect.getsource(chunk_rows)
    assert src.startswith("def chunk_rows("), src[:32]
    return (
        "\n"
        f"# --- equal-width chunked prefill ({GATE_ENV}) ---\n"
        "# patches/patch_chunk_equal_split.py. Stock caps a chunk at\n"
        "# long_prefill_token_threshold, so a prompt just past it lands on a\n"
        "# stub-tail chunk (5000 at cap 4800 -> 4800+200: a whole step for 200\n"
        "# rows). Same cap, same step count, no stub: cut ceil(remaining / cap)\n"
        "# equal chunks instead. Inert unless armed.\n"
        "import os\n"
        "\n"
        f'{GATE_VAR} = os.environ.get(\n'
        f'    "{GATE_ENV}", "0"\n'
        ').strip().lower() in ("1", "true", "yes", "on")\n'
        "\n"
        "\n"
        + src.replace("def chunk_rows(", f"def {HELPER}(", 1)
        + "\n"
    )


def site(indent: str) -> tuple[str, str]:
    """(anchor, replacement) for one clamp site; the two sites differ only by
    indentation, so one template covers both."""
    old = (
        f"{indent}if 0 < long_prefill_token_threshold < num_new_tokens:\n"
        f"{indent}    num_new_tokens = long_prefill_token_threshold\n"
    )
    new = (
        f"{indent}if 0 < long_prefill_token_threshold < num_new_tokens:\n"
        f"{indent}    num_new_tokens = (\n"
        f"{indent}        {HELPER}(\n"
        f"{indent}            num_new_tokens, long_prefill_token_threshold\n"
        f"{indent}        )\n"
        f"{indent}        if {GATE_VAR}\n"
        f"{indent}        else long_prefill_token_threshold\n"
        f"{indent}    )\n"
    )
    return old, new


ANCHOR_LOGGER = "logger = init_logger(__name__)\n"
EDITS = [
    (ANCHOR_LOGGER, ANCHOR_LOGGER + helper_block()),
    site(" " * 12),
    site(" " * 20),
]


def patch(src: str) -> str:
    if GATE_VAR in src:
        sys.exit("ERROR: chunk_equal_split orig scheduler.py is already patched")
    for i, (old, new) in enumerate(EDITS):
        count = src.count(old)
        if count != 1:
            sys.exit(f"chunk_equal_split: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"chunk_equal_split: patched scheduler_v032.py does not parse: {exc}")
    return src


def self_check() -> int:
    """Policy + generated-source check; no image, no orig, no engine needed."""
    assert chunk_rows(4800, 4800) == 4800, "exact cap fits one chunk"
    assert chunk_rows(4799, 4800) == 4799, "under the cap is one chunk"
    assert chunk_rows(5000, 4800) == 2500, "5000 at 4800 is two 2500-row chunks"
    assert plan_chunks(5000, 4800) == [2500, 2500]
    assert plan_chunks(9600, 4800) == [4800, 4800], "exact multiple: no split"
    assert plan_chunks(9700, 4800) == [3234, 3233, 3233]
    for cap in (1, 2, 7, 128, 4800, 8192):
        for total in range(1, 4 * cap + 4):
            plan = plan_chunks(total, cap)
            assert sum(plan) == total, (total, cap, plan)
            assert len(plan) == -(-total // cap), (total, cap, plan)
            assert max(plan) <= cap, (total, cap, plan)
            first = chunk_rows(total, cap)
            assert min(plan) >= first - 1, f"stub tail: {total} {cap} {plan}"
            assert all(a - b <= 1 for a, b in zip(plan, plan[1:])), (total, cap, plan)
    ns: dict = {}
    exec(compile(ast.parse(helper_block()), "<gate>", "exec"), {}, ns)
    fn = ns[HELPER]
    for cap in (1, 128, 4800):
        for total in (1, 4799, 4800, 5000, 9600, 9700, 20000):
            assert fn(total, cap) == chunk_rows(total, cap), (total, cap)
    print(f"self-check ok: {len(EDITS)} anchors, policy grid clean")
    return 0


def main(argv) -> int:
    if argv and argv[0] == "--self-check":
        return self_check()
    out_dir = argv[0] if argv else os.path.join(HERE, "chunk_equal_split")
    orig_path = os.path.join(out_dir, "orig", "scheduler.py")
    if not os.path.isfile(orig_path):
        sys.exit(f"ERROR: missing {orig_path} "
                 "(engine/patches.sh extracts it from the pinned image)")
    src = patch(open(orig_path).read())
    open(os.path.join(out_dir, "scheduler_v032.py"), "w").write(src)
    print("patched scheduler_v032.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
