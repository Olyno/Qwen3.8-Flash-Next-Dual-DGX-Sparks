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

# Day-1 spike results (2026-09-25, msi experiment Spark, checkpoint = lean working copy)

## Measured

| arm | prose | code | reasoning | entropy |
|---|---|---|---|---|
| batch-1 decode, NO speculative decoding | 16.2 / 16.3 tok/s | 16.2 / 16.6 | 16.5 / 16.8 | 16.3 / 16.7 |
| ngram lookup (S<=4, no corpus) tau (accepted/draft) | 1.39 / **2.69** | 1.53 / 1.71 | 1.99 / 1.99 | 1.36 / 3.77 (degenerate repetition — see decodebench.py docstring) |
| MTP=3 live production (user traffic, cumulative /metrics) | tau = 1 + 1,076,588/822,814 = **2.31** | | | |
| MTP=3 harness reference (README, same tooling) | prose ~2.9 tok/step | code ~3.85 | | |
| Packaged DFlash2 draft v0.1.0 (Sept 18, trained for the STOCK checkpoint) | val accept_len = **2.31** tokens/step (val_metrics.json) | | | |

## Raw data

`bench/acceptance_sweep.py --tag ngram4` wrote per-cell drafts/accepted;
reproduced verbatim in the console log above (first run in history to use
per-task acceptance deltas — the tool is now committed to this branch).

## Verdict: KILL the DFlash-2 drafter project (gate fired, day 1.5)

Pre-registered gate: a candidate advances past day 1 only if it beats
**2.9 tokens/step on prose** (this cluster's measured MTP=3 prose number).

- ngram lookup, the strongest free mechanism, best prose cell 2.69 < 2.9.
- The packaged DFlash2 drafter — an actually-trained block-diffusion head
  on a sibling Qwen3.8-Flash checkpoint — validates at 2.31 accept_len.
  The paper's 3.4-4.6 tau band is structured text; the external 24.7
  tok/s prose datapoint (Atlas, tau ~1.6) confirms the shape of the curve.
- Independent measurement agrees: live production MTP=3 on the baked
  checkpoint runs tau 2.31 on mixed traffic — and MTP drafting is FREE
  (the head ships in the checkpoint).

Conclusion: on low-structure reasoning prose, this generation's drafting
mechanisms (ngram, MTP, DFlash-class block diffusion) cluster at tau
1.4-2.9 and none clears the bar that would justify the training+integration
budget. The project's differentiated bet (beat MTP on prose) is answered by
evidence inside the first day and a half. Salvage: the sweep tool, the
working-copy procedure, and this negative result — re-openable if a
drafter with prose tau > 3 appears.

## Side findings (durable, non-obvious)

1. **Directory-name substring bug**: naming a served checkpoint dir with
   the word "dflash" anywhere in the path makes vLLM's spec-config
   auto-detection ("dflash" in model.lower()) classify method:mtp requests
   as dflash drafts -> EAGLEConfig wrap -> AttributeError on composite
   text_config. Avoid draft-brand substrings in checkpoint paths.
2. **TP1 + nvidia-derived checkpoint + MTP is broken in both local images**:
   old fc120: mtp.layers.48 FP8 scale param missing at ~85% weight load
   (the layer-index alias from start.sh's overlay path does not fix it on
   this loader; on TP2 the same checkpoint+MTP works). new a9c416: same
   crash pre-load. Production TP2 unaffected.
3. **Canonical image drift**: gx10 re-pulled vllm/vllm-openai:qwen38-flash-next
   (now a9c416…) after msi's fc120… build; msi retagged old one as
   `-msi-old`. TP1 experiments should pin the tag they were validated on.
4. **The lean working copy carries the MTP layer alias in-place**
   (config.json + hf_quant_config.json now declare mtp.layers.48);
   documented here, weights otherwise untouched.

## Day-1 addendum: head-to-head measurement of the packaged DFlash2 draft (same day)

Gate fired on validation-set evidence; this closes the "definitely" by
measuring draft v0.1.0 live, in place of the target's MTP head, on the
identical bench prompts (msi, image qwen38-dflash2-solve64-w4s2, the draft's
own lineage build; greedy, ignore_eos, K=1, 8-block drafts):

| task | tokens/step (tau) | decode tok/s | MTP=3 bar |
|---|---|---|---|
| prose | 1.371 | 15.6-18.3 | 2.9 (harness) / ~2.9-3.85 range |
| code | 1.412 | 16.4-18.8 | |
| reasoning | 1.455 | 17.1-19.4 | |

Reference points from the same machine, same day: no-speculative decode
16.2-16.8 tok/s; free ngram lookup tau 1.4-2.7. The trained drafter sits
AT the no-spec speed band (drafting overhead cancels its acceptance), and
its tau is BELOW the free mechanisms' best cells. Its validation accept_len
of 2.31 does not transfer to this traffic — the structured-text assumption
of the recipe (and of Atlas's 66.6-tok/s code result) does not hold on
long-horizon reasoning prose.

Reproduction: ~/dflash_spike/dfix_launch.sh (msi); metrics deltas via the
scrape in bench/acceptance_sweep.py (same formulas).

Verdict unchanged and now head-to-head confirmed: killed. If a drafter with
prose tau > 3 ever ships, the launcher and sweep tool revive this in
under an hour.

