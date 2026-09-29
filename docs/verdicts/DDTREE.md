# DDTree: harvesting the draft's candidate tree (killed, day 0)

Follow-up to `DFLASH.md`. After the DFlash kill, we proposed reopening on
*tree verification*, on the strength of the packaged draft's own validation
metric:

- Linear draft (one token per step): expected accepted length 2.31 in the
  draft packager's validation; 1.37–1.46 measured live on prose/code/reasoning.
- Full selector tree: `unary_top_16_oracle_accepted_length = 4.77` in the
  shipped `val_metrics.json` — if the target model verified the draft's 16-way
  candidate tree instead of a single chain, it could accept up to ~4.8 tokens
  per verification step.

The reopen was scoped as integration (2 days), on the belief that this serving
stack already carries the tree-verification path (`adaptive_verification.py`).
This document records why that belief was wrong and closes the proposal.

## Method

Static analysis of the serving stack as built into the experiment image
(`qwen38-dflash2-solve64-w4s2`): the worker's spec-decode sources
(`adaptive_verification.py`, `spec_decode/{dflash,dflash2,dspark}/*.py`,
`rejection_sampler.py`, `model_runner.py` sampling path) plus the draft model's
selector config. The GPU was not used; no new measurements.

## Findings

1. **The tree exists — in the draft only.** The draft's candidate selector
   scores every node conditional on its parent (`scores[step, parent, child]`,
   16 children per node; see `_selector_walk_kernel` in the dflash2
   speculator). Its output is a single path: the walk already picks the
   highest-probability chain. Nothing beyond that chain reaches the target.

2. **The target verifies chains.** The sampling path in this build is one
   contiguous run of bonus + draft tokens per request (`rejection_sampler.py`,
   `_iter_request_chunks`; batch layout `query_start_loc`/`cu_num_logits`).
   There is no parent index, no tree mask, and no branch-capable attention
   metadata anywhere in the worker or the attention backends of this image.

3. **`adaptive_verification.py` is not tree verification.** It allocates a
   global per-step budget of draft tokens across requests — ranking each
   request's chain positions by the running product of draft confidences — and
   trims chain length. It prunes; it never branches. The reopen's scoping was
   a misread of this file, caught before any measurement time was spent.

4. **The blocker is the model, not only the code.** 36 of this checkpoint's 60
   layers are gated delta net (linear attention). Tree verification requires
   every layer to read per-branch states; the GDN kernels here carry one state
   per request. Per-node state branching across 36 layers has no existing
   implementation in this stack — kernel research, not wiring.
   *(Inference from the kernel interfaces; nothing in this build to point at.)*

## Verdict

Killed at day 0 of the reopen: the "wire the existing components" path does not
exist. The 4.77 oracle remains what it always was — computed offline from the
draft's own scores — and is unreachable on this stack without implementing
tree attention with per-node GDN state on the target: a fork-level kernel
project measured in weeks.

The DFlash conclusion stands and absorbs this result. Speculative decoding on
this hardware is capped at the chain quality the model's own draft head already
produces; MTP (built-in, no extra weights) sits at the top of that cap:
τ ≈ 2.9 prose measured on the dual-Spark production server, ~2.3 on mixed live
traffic. Every alternative drafter we measured (ngram, packaged DFlash2) lands
below it.

## What would change this verdict

- Upstream ships target-side tree verification (tree attention masks **and**
  branch-capable linear-attention state) for hybrid models of this class. The
  draft's selector scores are already shaped to feed it.
- A drafter producing better single chains on reasoning prose. That lane is
  measured dead here: τ 1.37–1.46 live, versus the paper's numbers on code.
