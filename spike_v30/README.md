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


## First live boot (2026-09-26, chain): crash triage + fix set

All four v0.30 boots (stock/MTP3/hybrid on the new image) died identically:
engine initialized (modelopt_mixed, FlashInfer CUTLASS NVFP4 MoE, FlashInfer
GDN prefill + CUDA decode, ported PLE overlay reported the FP8 CPU table),
then the EngineCore worker was SIGKILLed (exit 137 = cgroup kill) during
post-load profiling. Hypothesis space (NOT yet proven on box; note the old
image ran fine WITH the same 100g cap, so the cap only bites via some
v0.30-specific memory behaviour):

1. cgroup cap: `--memory 100g --memory-swap 100g` copied from the old-image
   launcher; v0.30's engram/pinned PLE path + CUDA unified allocations +
   host weight pages exceed 100 GB during the profiling dummy-forward.
   The dual lane's feat/vllm-030 start.sh runs the SAME image + model for a
   1-hour soak, stable — and carries NO --memory flag.
2. packed-table mmap specifics: the UVA dummy-forward touches all 47.7 GiB
   of the page-cache-backed map (non-evictable once GPU-touched on unified
   memory); also O_RDWR open on a ro-mounted source can EROFS.
3. GB10 page-cache class the lane already knows: their start.sh runs
   evict_page_cache.py (fadvise DONTNEED) before launch — without it,
   "weight loading can die partway with CUDA OOM on an otherwise idle box"
   (their words, #35/#61). Our launcher did neither.

Fix set applied in launch_v30.sh (each neutralizes one suspect; together
they cover the space): cap removed, evict_page_cache.py before boot,
VLLM_PLE_PACKED_TABLE_DIR demoted to opt-in (native pinned path becomes the
default — the mmap overlay must earn its place in a later A/B, if ever).

Also verified from the upstream v0.30.0 tree (GitHub contents API):
`vllm/v1/worker/gpu/spec_decode/dflash2/` EXISTS (speculator.py) — the newer
image carries the DFlash2 machinery our serving fork lacks. qwen4_exp
nvidia/ files confirmed byte-identical to what we ported against.

Note: v0.30 defaults VLLM_PLE_CPU_OFFLOAD to 1 (fork PR #68) — we keep it
explicit.

## Second crash (2026-09-26 ~16:10, post-B1): box hard-hang, driver bug found
After the quality gate finished (B1 PASS — see docs/HYBRID-FP8.md), the A1
boot was launched while the A4 lean->hybrid converter was still copying
(~120 GB of EXDEF fallback copies through a 1.5 GB-capped container).
sshd went unresponsive while Tailscale still answered pings — same hang
signature as the morning's chain; the box needed a power-cycle. Root cause
class: two concurrent unified-memory hogs (GPU boot + bulk page-cache
writer). RULE: never boot a vLLM server while a bulk copy runs; serialize
heavy jobs. resume.sh encodes this (phases are single-purpose; A4 build
is a separate manual step run ONLY between phases).

The boot-time audit also caught a launcher bug worth its own line:
**launch_v30.sh mounted "$MODEL" but passed the hard-coded
`nvidia/Qwen3.8-Flash-Next-NVFP4` to `vllm serve`** — every A-phase would
have silently served the HF-cache stock model and mislabeled the hybrid
rows as v0.30 gains. Fixed: serve "$MODEL". (The mount made the dir
available; the argument never used it — classic bind-mount camouflage.)

## GPU queue (adopted recipes; box recovery is the only blocker)
1. A1: stock wk1, K=6, no drafter (launch_v30.sh defaults) -> decodebench +
   concbench ladder vs 16.0-16.2 old-image baseline and vs the +60% hybrid
   rows (same protocol).
2. A2: + speculative-config {method mtp, K=5, draft_sample_method
   probabilistic, rejection_sample_method block} — hibrid48 serving recipe
   on our checkpoint; vocab-reduction overlay (MTP_DRAFT_VOCAB) must NOT be
   combined: use_local_argmax_reduction is rejected with sampled drafting
   (team PR #71 finding).
3. A3: q38-hyb + A2 spec (the combo the B1 verdict greenlights).
4. A4: lean x hybrid combo checkpoint (converter running when box returns)
   -> its own 3-suite gate before serving adoption.
Bench rows land here + in docs/HYBRID-FP8.md; P1 profiler = same boot with
VLLM_TORCH_PROFILER_DIR set (launcher opt-in mount).
