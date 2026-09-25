# DFlash-class drafter + DDTree — experiment plan & log

Goal: raise batch-1 decode speed above this cluster's measured MTP=3
baseline **without changing a single emitted token** (speculative
decoding verifies every draft token against the target model, so output
is byte-identical by construction — the only cost is wasted draft work).

Status: **planned, day-1 spike pending.** This document is the working
log; every claim below marked *measured* has a command and raw output
recorded in `results/` (this repo, branch `experiments/dflash`).

## Baseline (measured, this kit — TP2+EP, MTP=3, GMU 0.835)

| metric | value | source |
|---|---|---|
| batch-1 greedy, MTP off | 24.5 tok/s | README "MTP measured on this kit" |
| batch-1 greedy, MTP=3 | 52.1 tok/s (2.13×) | same |
| draft acceptance | 72.8% (823/1131); 89 / 74.5 / 60 by position | same |
| with 47k reduced draft vocab (default) | code 71.5 / prose 56.8 tok/s @1 stream | README "Reduced-vocabulary MTP drafting" |
| tokens per step | code 3.85 / prose ~2.9 | same |
| single Spark (TP1), MTP=3 | prose ~24.9 / code ~35.1 tok/s | README tp1 section |

The prose/code gap (2.9 vs 3.85 tok/step) is the target: a drafter that
closes it is the whole thesis of this experiment.

## Method (external evidence, honestly labeled)

- **DFlash** (z-lab, open recipe): a small (5-layer) *block-diffusion*
  drafter conditioned on the target model's hidden states drafts 16
  tokens per block in one pass, reusing the target's embeddings and
  lm_head. Published τ (accepted tokens per verification step) 3.38–4.61
  on structured text vs 2.1–2.5 for EAGLE-3-class drafting.
- **DFlash-2 existence proof**: Atlas Cybernetics reports 66.6 tok/s code
  (4.10×) and 24.7 tok/s prose (τ≈1.6) for Qwen3.8-27B on one DGX Spark,
  byte-identical output. *Their* prose number is our caution flag and
  our differentiation: low-structure text is where drafting is hardest.
- **DDTree** (arXiv 2604.12989): builds a best-first verification tree
  from the same single draft pass; claimed +~10% acceptance over linear
  drafting at bounded verify cost. External number → re-measured here.

Note our baseline drafter class: this checkpoint's built-in MTP head
already exceeds the EAGLE-3 numbers DFlash's paper compares against, so
expected gain is *bounded* — the gate below is set against OUR numbers,
not the paper's.

## Plan (6 working days on the experiment Spark; production cluster
untouched until a ship decision)

| day | work | gate / output |
|---|---|---|
| 1 (+½) | **Kill-gate spike, zero new training.** Serve the working-copy checkpoint; measure per-position acceptance → tokens/step with two free mechanisms: vLLM `ngram` proposer (corpus-mined prompt lookup over post-bake reasoning traces) and existing MTP=3, on code + prose + GPQA-style reasoning. | **> 2.9 tok/step prose**, else project killed (days spent: ~1.5) and negative result documented here |
| 2–3 | **Train the drafter.** 5-layer block-diffusion head conditioned on target hidden states (extracted offline via the calibration dump path from the bake tooling); shares target embeddings + lm_head; corpus = ~170M tokens of post-bake traces (code + reasoning prose), 20% held out. | held-out τ; per-position decay vs the 89/74.5/60 MTP curve |
| 4 | **Integrate.** Custom proposer in the serving vLLM build (plugin path alongside `method: mtp`); reduced draft-vocab trick reused if applicable. Fallback: benchmark under SGLang where DFlash exists. | **byte-identity check** vs non-speculative decode (the lossless proof) |
| 5 | **DDTree** on top; full matrix. | tok/s {MTP3, DFlash, DFlash+DDTree} × {code, prose, math} × batch {1,2,4,8} |
| 6 | **Ship or scrap.** Ship = merge to main + `start-dflash.sh` + this log finalized. Scrap = archive as negative result, keep corpus & curves. | ship gate: **prose ≥65 or code ≥80 @1 stream** (vs 56.8 / 71.5) |

## Artifacts & locations

- Working copy of the baked checkpoint (124 GB, never edited; hidden
  states and acceptance traces land separately):
  `$HOME/models/Qwen3.8-Flash-Next-NVFP4-dflash-target` — the published
  `-lean` checkpoint is untouched by every experiment below.
- Training/serving host: the experiment box; the 2-node production
  cluster keeps serving throughout.
- Raw benchmark outputs: `results/dflash/` (this branch).

## Decision log

- 2026-09-25 — branch `experiments/dflash` cut from `main` @ `e80340f`;
  plan approved as above; baseline anchor corrected to the cluster's own
  measured MTP=3 numbers (52.1 / 71.5 code / 56.8 prose) after review —
  the original draft anchored on external blog numbers.

## Results

_(filled by the days above)_
