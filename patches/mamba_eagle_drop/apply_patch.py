#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""MambaManager drop_eagle_block cache-hit fix for the vLLM 0.30.0 lane
(unconditional).

Backports vllm-project/vllm#57128 onto the stock v0.30.0 files:

Upstream: https://github.com/vllm-project/vllm/pull/57128 (limwand,
unmerged), Apache-2.0.

    vllm/v1/core/single_type_kv_cache_manager.py
    - MambaManager.find_longest_cache_hit accepted drop_eagle_block but
      never acted on it: with GDN + MTP/EAGLE + prefix caching, the most
      recent matched block — which may hold recurrent state written over
      rejected draft positions — stayed reachable through the prefix cache
      and got reused by later requests sharing that prefix (silent
      corruption). Both the fine-grained and the coarse branch now scan the
      full, unrestricted window and skip only the first (most recent)
      checkpoint actually found when drop_eagle_block is set, then keep
      scanning for the next, older, already-committed one. The PR's
      shared_prefix_checkpoint -> fine_grained_prefix_cache rename hunks
      are already in v0.30.0 and are not re-applied.

    vllm/v1/core/kv_cache_coordinator.py
    - Comment-only hunk: the coordinator already (correctly) skips the
      eagle_margin for MambaSpec groups; the comment now documents the
      right reason (the mamba finder drops in place instead of searching
      past the candidate length).

Correctness fix (the unhandled drop read stale GDN state), so no toggle.

Inputs:  patches/mamba_eagle_drop/orig/*.py (extracted from the image)
Outputs: patches/mamba_eagle_drop/*_v030.py (mounted by engine/patches.sh)
argv[0]/argv[1] override the orig/output directories (used by the test).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# vllm/v1/core/single_type_kv_cache_manager.py (MambaManager)
# ---------------------------------------------------------------------------
FINE_CEIL_OLD = """\
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
"""
FINE_CEIL_NEW = """\
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            # In "align" mode, real Mamba state checkpoints only exist at
            # sparse positions (block_size / scheduler-step boundaries), not
            # at every fine-grained hash unit. So we must not pre-shrink the
            # search ceiling by a fixed hash_block_size (that reliably lands
            # in a gap between checkpoints and yields no hit at all). Instead,
            # search the full, unrestricted window and -- only for the first
            # (most recent) checkpoint we actually find -- skip it once when
            # drop_eagle_block is set, then keep scanning for the next
            # (necessarily older, already-committed) checkpoint below it.
            # This excludes exactly the one block that may hold unverified
            # MTP/EAGLE draft state, instead of blanking out the whole tail
            # of the search window.
            skip_next_hit = drop_eagle_block
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
"""
FINE_SKIP_OLD = """\
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    block_idx = fine_idx // scale_factor
"""
FINE_SKIP_NEW = """\
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    if skip_next_hit:
                        skip_next_hit = False
                        continue
                    block_idx = fine_idx // scale_factor
"""
COARSE_CEIL_OLD = """\
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
"""
COARSE_CEIL_NEW = """\
        max_num_blocks = max_length // block_size
        # See the fine-grained branch above for why we don't pre-shrink the
        # ceiling: skip only the first real match we find when
        # drop_eagle_block is set, then keep scanning for the next one.
        skip_next_hit = drop_eagle_block
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
"""
COARSE_SKIP_OLD = """\
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
"""
COARSE_SKIP_NEW = """\
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                if skip_next_hit:
                    skip_next_hit = False
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
"""
MGR_HUNKS = (
    (FINE_CEIL_OLD, FINE_CEIL_NEW),
    (FINE_SKIP_OLD, FINE_SKIP_NEW),
    (COARSE_CEIL_OLD, COARSE_CEIL_NEW),
    (COARSE_SKIP_OLD, COARSE_SKIP_NEW),
)

# ---------------------------------------------------------------------------
# vllm/v1/core/kv_cache_coordinator.py (comment only)
# ---------------------------------------------------------------------------
COORD_OLD = """\
                # Eagle matches one extra drop unit (one hash unit for
                # fine-grained managers, else one cache block) and then drops
                # it, landing back at the candidate length. No margin for
                # mamba: its finder never drops (draft models have no mamba
                # layers), so the hit would grow past the candidate.
"""
COORD_NEW = """\
                # Eagle matches one extra drop unit (one hash unit for
                # fine-grained managers, else one cache block) and then drops
                # it, landing back at the candidate length. No margin for
                # mamba: unlike the other managers, MambaManager's finder
                # never needs to search *beyond* the candidate length to
                # honor drop_eagle_block -- it skips only the first (most
                # recent) checkpoint it finds within [0, curr_hit_length) and
                # keeps scanning for the next, older one, so the result never
                # exceeds curr_hit_length in the first place.
"""
COORD_HUNKS = ((COORD_OLD, COORD_NEW),)

# (orig name, hunks, output name)
TARGETS = (
    ("single_type_kv_cache_manager.py", MGR_HUNKS,
     "single_type_kv_cache_manager_v030.py"),
    ("kv_cache_coordinator.py", COORD_HUNKS, "kv_cache_coordinator_v030.py"),
)


def main(argv) -> int:
    orig_dir = argv[0] if argv else os.path.join(HERE, "orig")
    out_dir = argv[1] if len(argv) > 1 else HERE
    for orig_name, hunks, out_name in TARGETS:
        orig_path = os.path.join(orig_dir, orig_name)
        if not os.path.isfile(orig_path):
            sys.exit(f"ERROR: missing {orig_path} "
                     "(start.sh extracts it from the image)")
        src = open(orig_path).read()
        if "skip_next_hit" in src:
            sys.exit(f"ERROR: mamba_eagle_drop orig {orig_name} is already patched")
        for i, (old, new) in enumerate(hunks):
            count = src.count(old)
            if count != 1:
                sys.exit(f"mamba_eagle_drop: anchor {i} in {orig_name} not "
                         f"unique/missing (count={count}):\n{old[:200]}")
            src = src.replace(old, new)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"mamba_eagle_drop: patched {out_name} does not parse: {exc}")
        open(os.path.join(out_dir, out_name), "w").write(src)
        print(f"patched {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
