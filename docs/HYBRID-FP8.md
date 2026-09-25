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
