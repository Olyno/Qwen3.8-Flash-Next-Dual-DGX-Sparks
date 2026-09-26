# vLLM v0.30.0 upgrade — port + deprecation inventory (analysis day, 2026-09-26)

Target image: `vllm/vllm-openai:v0.30.0` (stable, published 2026-09-22). Chosen over
the Aug-31 nightly cached here after diffing: qwen4_exp model files are byte-identical
nightly vs stable; only docstring/format drift in modelopt.py. Newest nightly (Sep 25)
exists but stable is the "keep one specific image" candidate.

The model lives at `vllm/models/qwen4_exp/` in v0.30 (serve image: qwen3_8_flash_next);
registry maps `Qwen4ExpForConditionalGeneration` natively — no model-registry patch.

## Now NATIVE in v0.30 → our old patches are DEAD CODE
1. **modelopt MIXED_PRECISION arms**: `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_PB_WO` and
   `FP8_BLOCK_SCALES` MoE are first-class (`LINEAR_ALGOS` table + `resolve()`).
   Our `files/modelopt_patched.py` + `hyb_spike/stack_modelopt.py` exist only for
   this → DELETE after bench.
2. **Prefix candidates**: `_quantized_layer_prefix_candidates` handles
   `model.language_model.* ↔ language_model.model.* ↔ model.*` and bare `lm_head`
   (exactly the stacker's item 3). Also `exclude_modules` wildcard matching
   (`*.ple.*`) is native.
3. **PLE FP8 dispatch**: `ple_embedding_dtype=="float8_e4m3fn"` →
   `Qwen4ExpPLEFp8EmbeddingMethod` in stock code. Our serve patch for the FP8 arm
   is superseded (the NVFP4-PLE arm in patch_ple_layer.py was never needed for the
   wk1/stock checkpoints — PLE is FP8 there).
4. **PLE CPU offload**: native `EngramConfig` (`--engram-config`, defaults to
   env `VLLM_PLE_CPU_OFFLOAD=1`) → `Qwen4ExpPLEPinnedHostEmbedding`: pinned CPU
   table + UVA triton lookup + side-stream prefetch + graph-safe wiring.
   Our 4-file `files/ple_offload/` bind-mount stack (protocol/worker/connector/
   ple_offload_layer, incl. the cuStreamWaitValue32 GB10 workaround) can go —
   pending the GB10 capture behaviour check on first live boot.
5. **MTP draft quant config**: `get_draft_quant_config` + `configure_quant_config`
   already applied in `qwen4_exp/nvidia/mtp.py` (serve needed our mtp.py overlay
   AND it missed the HC mixer — see below).

## Still OURS (ported `files/`, marker `[fp8dense overlay]`)
- Stock `GatedResidual` hardcodes `quant_config=None` for its 3 Linears; the
  checkpoint declares **290 HC tensors FP8_PER_CHANNEL_PER_TOKEN** (measured in
  q38-hyb config: NVFP4 48, FP8 1, FP8_BLOCK_SCALES 1, PCPT 591). Passing the
  resolved mixed-precision config through is still required, same as serve.
- Same for both final `hyper_connection_mixer` GatedResiduals (target + MTP) and
  both `ParallelLMHead`s (FP8 per the checkpoint; v0.30 `get_quant_method` handles
  `ParallelLMHead` → ModelOptFp8PcPt, so the arm works).
- **NEW vs serve era**: the MTP mixer now gets `draft_vllm_config.quant_config` —
  v0.30 quantizes `mtp.*.hyper_connection_mixer` (serve checkpoint excluded mtp.*).
  Missing this = FP8 weights loaded into an unquantized bf16 Linear → load fail.
- `_PlePackedTableEmbedding` (ngram_embedding.py overlay): mmap-backed pinned PLE
  from `VLLM_PLE_PACKED_TABLE_DIR` (47.7 GiB table; page-cache on GB10 unified
  memory instead of anonymous pinned RAM). Reuses native UVA lookup; shard copies
  flow through the normal loader. Bench against native pinned before deciding.

## Files
- `port_v30.py` — anchor-checked port script (re-run against future images; refuses
  on drift). Run from this dir with `~/upgrade/v30/src` populated from the image.
- `files/{model,mtp,hyperconnection,ngram_embedding}.py` — ported outputs,
  bind-mounted by `launch_v30.sh` (exact ps_launch.sh flags + engram defaults).
- `imp_check.py` — no-GPU wiring test; PASSED inside v0.30 image (2026-09-26):
  `IMPORT-OK all four ported files wired`.

## GPU queue (after ProbSparse quality + Hybrid-FP8 bench free msi)
1. stock wk1 on v0.30 (K=10, MTP off) → decodebench prose/code vs the 16.0-16.2
   tok/s old-image baseline → gain/loss number.
2. same boot → GPQA smoke (lossless sanity of the engine swap).
3. if ≥ parity: q38-hyb + MTP arms ride the same overlay; retire dead patches.
