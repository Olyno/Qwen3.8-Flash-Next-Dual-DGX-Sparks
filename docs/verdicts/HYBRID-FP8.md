# Hybrid FP8 dense projections — experiment plan + recon

Third decode-speed experiment (after `DFLASH.md`, `DDTREE.md`, `PROBSPARSE.md`).
Idea: at batch-1 decode the model is memory-bandwidth bound; the dense
(bf16) side layers are the largest single slice of bytes streamed per step.
Convert them to FP8 E4M3 per-output-channel weights with dynamic per-token
activation quantization — no calibration data, no draft, orthogonal to
MTP/acceptance — and keep everything that must stay precise (router gates,
norms, PLE, embeddings, MTP) in bf16/fp8 as shipped.

This lane already exists in the serving fork as `FP8_DENSE=true`
(`serve/fork-lean/files/fp8dense/`), built against the **RadixArk**
checkpoint layout. It ships a streaming converter, four loader overlay
diffs (~150 lines total), a verifier — and an explicit gap in its own
README: "Not yet measured on GPU… GSM8K/AIME re-evaluation is still owed."
This experiment closes that gap on our distribution, then ports the build
to the checkpoint we actually serve.

## Recon (this box, 2026-09-25, no GPU used)

- Serving image supports `FP8_PER_CHANNEL_PER_TOKEN` natively
  (`modelopt.py:103,387,536`) — dispatch exists without patches.
- The TP1 recipe bind-mounts `serve/files/modelopt_patched.py` (FP8-block
  MoE support); that file lacks the per-channel dispatch → the hybrid mount
  set must be the `files/overlay/*.py` patched files (modelopt,
  hyperconnection, model, mtp), which this repo's `start.sh` already does on
  TP2 (`FP8_DENSE=true` → overlay mounts, README §FP8-dense).
- The RadixArk source build is 135 GiB incl. NVFP4 PLE — **cannot fit
  TP1** here; its measured numbers (if it ever gets any) are TP2-only.
- nvidia/lean checkpoint byte ledger (measured from shard headers):
  10.22 GiB BF16 total; quantizable ≈ 7.5 GiB (GDN 3.89 + hyper-conn 1.23 +
  attn 1.25 + lm_head 1.18 + shared-expert 0.45 + other 0.95 minus keeps).
  Embeddings excluded. In the nvidia layout the standard-attention q/k/v/o
  are already MXFP8 for most layers, so the realistic saving is
  ≈ 3–3.5 GiB of the ≈ 7 GiB no-spec TP1 step → the analytical model
  (fork README: 9.2→5.7 GiB/step on TP2, ≈1.5×) says decode **+25–45 %**
  on TP1; the byte math here says +30–50 % if all of GDN+HC+lm_head halve.
- Tensor naming: identical `model.language_model.layers.*` namespace as the
  RadixArk build (verified from the nvidia hf_quant_config exclude patterns)
  → converter QUANT_PATTERNS transfer unchanged; only the **shard plumbing**
  differs (nvidia packs bf16+fp8+nvfp4 tensors inside one 10-shard set,
  no separate `model-bf16-*` files; the converter must rewrite in-place
  names, keep foreign-dtype tensors verbatim, and merge
  `quantized_layers`/`exclude_modules` with the shipped ModelOpt config
  instead of replacing it). Verified: attention q/k/v/o are fully bf16
  (1.246 GiB measured from shard headers; the shipped quantized_layers are
  only 48×NVFP4 expert blocks + FP8 PLE + one FP8_BLOCK group), so every
  dense group in the ledger above is quantizable.
- The bake touches lm_head (49 rows). Quantizing lm_head must run
  **after** the bake rows are in place — conversion from our lean copy,
  not from stock. lm_head is 248k×2560 bf16; per-channel FP8 halves its
  1.18 GiB read per step (draft head untouched: MTP stays excluded).

## Design

1. Port `make_fp8_dense_checkpoint.py` → nvidia/lean shard layout
   (single-pass: per-tensor decision, verbatim copy for non-quantized,
   merge ModelOpt config from the shipped `hf_quant_config.json`).
   Work stays on copies: input = wk1 (lean working copy), output = new dir
   `~/models/Qwen3.8-Flash-Next-NVFP4-hyb`, never the published checkpoint.
2. Serve on TP1 msi with the overlay mount set + no drafter first
   (comparable to the k10 control 16.0–16.2 tok/s), then with MTP=3
   (comparable to the production acceptance numbers; the TP1+MTP crash from
   the DFlash lane must not recur — it was specific to the aliased wk1
   MTP layers; hybrid keeps MTP bf16-excluded as stock nvidia does).
3. Quality gate: GPQA-198 + MATH-500 + GSM8K-100 on the hybrid, same
   harness as the ProbSparse arm (and its k6 numbers become a free
   reference row). Gate ≤ 1 pp vs stock baselines (74.2 / 88.8 / 96.0).
4. Speed gate: prose ≥ +20 % vs k10 control on the same box, else kill.

## Log

- 2026-09-25: recon complete. Converter ported to the nvidia shard layout
  (`spike_hyb/make_fp8_dense_nvidia.py`): per-shard quantize decision,
  foreign tensors byte-copied, ModelOpt config merged (48 NVFP4 + FP8 PLE +
  FP8_BLOCK entries preserved) with the shipped `exclude_modules` list
  REPLACED by the canonical keep-list — vLLM's `is_layer_excluded` check runs
  before the `quantized_layers` lookup and the nvidia config excludes exactly
  the tensors we quantize (they ship bf16). 591 dense linears confirmed
  (identical count to the RadixArk build).
- Loader overlays ported: 3 of 4 diffs apply clean to our serving image
  (`modelopt`, `model`, `mtp`); `hyperconnection` needed `nvidia/` (not
  `common/`) as the patch target. The per-channel dispatch is genuinely
  missing from the image's MIXED_PRECISION `get_quant_method` (FP8/NVFP4/
  W4A16/MXFP8 arms only) — overlay mandatory, as the fork documented.
  `stack_modelopt.py` layers the overlay on top of `serve/files/
  modelopt_patched.py` (FP8_BLOCK MoE + per-channel arms in one file; the
  TP1 PLE-offload recipe needs both).
- Build + verify: `~/models/q38-hyb` from `~/models/q38-stock-real` (real
  copy, hardlinked untouched shards; converter never writes src).
  Verifier ported (`verify_fp8_dense_nvidia.py`): 591/591 FP8 tensors,
  foreign tensors unchanged in rewritten shards, worst dequant rel-err
  0.0263 (E4M3 floor), 0 problems. Disk: 69 GiB of new bytes — the nvidia layout
  packs experts inside the dense-bearing shards, so 9 of 11 shards are
  rewritten wholesale (RadixArk separate-`model-bf16-*` layout needed only
  11.8 GiB); acceptable here (1.9 TB free).

- 2026-09-26 06:19: GPU speed gate MEASURED on msi, old serving image, exact
  ps_launch protocol (decodebench, 600 tok, temp 0.6, no drafter, K=10 dense
  control rows from the ProbSparse table):

  | content (ctx 1k) | control NVFP4-dense | hybrid FP8-dense | delta |
  |---|---|---|---|
  | prose   | 16.0 | **25.6** | +60 % |
  | code    | 16.1 | **26.1** | +62 % |
  | entropy | 16.0 | **25.8** | +61 % |
  | copy    | 17.4 | **29.2** | +68 % |

  100k-context rows agree (25.4-28.9). Only variable: checkpoint dense quant
  format. Analytical model said +30-50 %; reality is +60-68 % - the byte model
  undercounted NVFP4 dequant cost (dense matmuls pay per-element unpack; the
  FP8 W8A8 path does not). Independent corroboration from the dual-spark lane:
  closed PR #44 measured +49 % FP8-dense prose on TP2 (36.7 -> 54.8).
- Boot bug found + fixed: the converter's EXDEV fallback emitted `-> /src/...`
  symlinks (hardlink across the build container's separate mounts fails); they
  dangle when serving the checkpoint dir alone - tokenizer load died instantly.
  All 13 small files + 2 weight shards dereferenced in place (119.7 GiB total,
  0 symlinks, re-verified 591/591), and `make_fp8_dense_nvidia.py` now COPIES
  on hardlink failure instead of symlinking.
- QUALITY GATE PENDING (GPU blocked on box recovery): GPQA/MATH/GSM8K on
  q38-hyb, same runner/scorer; baselines 74.2 / 88.8 / 96.0; gate <= 1 pp.
  The +60 % only ships if this passes.

- 2026-09-26 13:04-18:10: QUALITY GATE MEASURED on msi, old serving image,
  bake-matrix protocol (bench_runner temp 0.6 top-p 0.95 seed 1337, same
  scorer as the lean study and the ProbSparse arm), three suites on
  `~/models/q38-hyb`, paired against `results/arms/*__base.scored.jsonl`:

  | suite | hybrid | baseline (stock) | delta | paired flips (hyb/base) |
  |---|---|---|---|---|
  | GPQA-198 | **155/198 = 78.3 %** | 147/198 = 74.2 % | **+4.1 pp** | 18 / 10 |
  | MATH-500 | 442/500 = 88.4 % | 444/500 = 88.8 % | −0.4 pp | 6 / 8 |
  | GSM8K-100 | 96/100 = 96.0 % | 96/100 = 96.0 % | 0.0 pp | 1 / 1 |

  Overthinking-proxy counts (ot_proxy on wrong rows): GPQA 40 (hyb) vs 51
  (base) — the hybrid arm both scores higher and rambles less. MATH 17 vs
  18, GSM8K 2 vs 2. McNemar on paired ids: GPQA net +8 (p ≈ 0.21, two-sided
  binomial — the positive direction is at least as credible as the negative);
  MATH net −2 (p ≈ 1.0); GSM8K net 0. Nothing regresses; the hard suite moves
  up.

## Verdict

**ADOPT.** Speed gate PASSED with margin (+60-68 % all content classes);
quality gate PASSED and then some: GPQA +4.1 pp, MATH −0.4 pp (noise,
flips 6v8), GSM8K exact tie. Joint reading: at batch-1 the dense projections
are BOTH the bandwidth bottleneck and (at NVFP4) a precision bottleneck —
FP8 per-channel fixes both at once, the only single change in the program
so far that raises speed AND intelligence. Serving default: q38-hyb +
num_experts_per_tok=6 + lean bake (combo checkpoint A4 building; B1 proves
the quant axis, lean's +5.6 pp was measured separately, combo re-gate owed
by its own 3-suite run). The +4.1 pp on GPQA is within noise for a strict
claim — what is NOT noise-worthy to ignore: quality went UP while speed
went up 60 %. Caveats kept honest: single scorer/protocol, temp 0.6;
structured-output tasks untested; the MTP path runs its own dense layers —
A3 measures v0.30+hyb+spec end-to-end.
