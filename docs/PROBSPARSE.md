# Reducing routed experts per token (ProbSparse-style) — measured on one Spark

Third experiment in the decode-speed program (after the drafter tests in
`DFLASH.md` / `DDTREE.md`). Idea from
"You Only Need 2/3 of the Chosen Experts" (arXiv 2609.25809, 2026-09-22): in
fine-grained MoE models the router's lowest-weight experts per token contribute
little, and *uniformly* serving fewer of them (the paper's "one-integer
change") keeps ~98.8 % of quality while serving 1.2–1.7× faster. Dynamic
per-token rules added <1 % over uniform truncation at that budget — so this
spike tests the uniform variant, which needs no code patch at all, only a
config field.

This checkpoint routes 10 of 512 experts per token per MoE layer (48 layers).
At batch-1 decode, expert weights are ~18 % of the bytes read per token
(byte ledger, `DFLASH.md` program notes), so cutting k 10→6 removes 40 % of
that slice: the step-time model predicted +3.6 % (k8) to +12 % (k6).

## Method

- One Spark (TP1), no drafter, working server from the day-1 recipe:
  stock `nvidia/Qwen3.8-Flash-Next-NVFP4` via the fork's FP8-PLE CPU-offload
  image (`serve/files/` patch set, GMU 0.735, ctx 131k) — the exact stack that
  served the 16–17 tok/s no-spec baseline in `DFLASH.md`.
- Expert count changed with `--hf-overrides
  {"text_config": {"num_experts_per_tok": K}}` — vLLM merges dict overrides
  recursively into the nested text config (`config/model.py
  _apply_dict_overrides`), so no checkpoint copy is needed. The serving code
  reads `hf_text_config.num_experts_per_tok` at MoE-block construction
  (`model_executor/models/qwen3_next.py`), i.e. the override is the real
  activation switch. K=10 reproduces the stock server exactly (control).
- Speed: `bench/decodebench.py` (greedy-style ignore_eos, 600 tokens, temp
  0.6), contexts 1k and 100k, tasks prose/code/entropy/copy; one pass per
  config, servers restarted between arms (single variable: K).
- Quality: GPQA-Diamond (198) + MATH-500 (500) + GSM8K (100) through the same
  runner/scorer as the lean bake study (temp 0.6 / top-p 0.95 / per-problem
  seed), stock-checkpoint baseline rows already on disk
  (GPQA 74.2 % / MATH 88.8 % / GSM8K 96.0 %). Gate: ≤ 1 pp total drop vs
  baseline on each. K=6 is the worst case: if it passes, K=8 passes by the
  paper's monotonicity; if it fails, K=8 is tested to find the usable edge.

## Speed result (measured, this box, tok/s)

| content (ctx 1k) | K=10 | K=8 | K=6 | K=6 vs K=10 |
|---|---|---|---|---|
| prose   | 16.0 | 16.4 | 16.7 | **+4.4 %** |
| code    | 16.1 | 16.4 | 16.9 | **+5.0 %** |
| entropy | 16.0 | 16.5 | 16.9 | +5.6 % |
| copy    | 17.4 | 17.6 | 18.2 | +4.6 % |

100k-context rows agree (+2.5–7.7 %). Monotone K10 < K8 < K6 across every
cell — the effect is real and directionally as the byte ledger predicted
(half the optimistic band: this serving stack spends more per step than pure
expert weights, and kernel-level costs don't scale linearly with count).

Raw rows: `spike_ps/results_k{10,8,6}.txt`; table: `spike_ps/ps_analyze.py`.

## Quality result (measured, this box)

(quality gate runs after the speed table — see below)

## Findings during setup (worth recording)

- The fork-lean patch set (`serve/fork-lean/files/`) is for the **RadixArk**
  checkpoint (NVFP4 PLE table); serving the nvidia checkpoint's FP8-PLE path
  with it makes the PLE offload worker die at CUDA context creation. The
  nvidia patch set (`serve/files/`) is the correct mount for stock/baked-lean
  on TP1; `start-tp1.sh` in this repo selects via `PLE_GIB`, direct-launch
  recipes must pick the mount set deliberately.
- The 124 GiB lean working copy (wk1) **cannot fit TP1** under any GMU:
  96.8 GiB non-PLE weights vs ~89 GiB budget. All TP1 work on this box uses
  the 98.6 GiB stock checkpoint (with FP8-PLE override).
- Day-1's 16.2–16.8 tok/s no-spec band is reproduced by the K=10 control
  (16.0–16.2 here; same stack, fresh server) — the measurement rig is stable
  across days to ±0.3 tok/s.
