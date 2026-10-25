# Serving experiments

A running log of changes to the serving path: what was measured, what was kept, and what was rejected. Negative results are kept on purpose so the same experiment is not repeated.

## Method

- `bench/decodebench.py`, 200 generated tokens per run, contexts 1k / 32k / 65k, temperatures 0 and 0.8, four content types (prose, code, entropy, copy).
- A fresh context nonce per run, so prefill is not served from the prefix cache.
- Server otherwise idle. `tok/s = engine steps/s × tokens/step`; acceptance comes from vLLM's spec-decode counters.
- Hardware for every entry below: one DGX-class box, `tensor_parallel_size = 1`, FP8 KV, MTP with 4 drafts, FP8 draft head, adaptive verify width, `gpu_memory_utilization = 0.72`, `max_model_len = 65536`.

The number that matters for an agent is not the best case. Copy-heavy output reaches 60+ tok/s because the draft head predicts quoted text almost perfectly; prose is the floor. Report both.

## Baseline, committed configuration

Temperature 0, 200 decoded tokens:

| context | content | TTFT (s) | decode (tok/s) | steps/s | tokens/step | acceptance |
|---|---|---|---|---|---|---|
| 1,000 | prose | 0.53 | 27.8 | 13.3 | 2.09 | 44.1% |
| 1,000 | code | 0.52 | 37.0 | 13.2 | 2.83 | 60.2% |
| 1,000 | entropy | 0.57 | 38.4 | 12.5 | 3.11 | 67.2% |
| 1,000 | copy | 0.59 | 61.7 | 13.9 | 4.42 | 89.5% |
| 32,000 | prose | 11.34 | 30.2 | 14.3 | 2.12 | 46.7% |
| 32,000 | code | 11.34 | 39.7 | 13.8 | 2.93 | 62.1% |
| 32,000 | entropy | 11.34 | 42.6 | 14.6 | 2.97 | 63.2% |
| 32,000 | copy | 11.36 | 65.7 | 13.9 | 4.81 | 95.2% |
| 65,000 | prose | 22.92 | 28.1 | 14.1 | 1.99 | 43.4% |
| 65,000 | code | 22.85 | 38.6 | 14.2 | 2.73 | 63.0% |
| 65,000 | entropy | 22.87 | 41.5 | 14.2 | 2.94 | 59.5% |
| 65,000 | copy | 22.91 | 62.0 | 13.7 | 4.55 | 92.9% |

Temperature 0.8 moves individual rows by a few points (prose 27-29, code 36-44, copy 59-66) without changing the shape.

What the table says:

- **Decode is flat in context.** Prose 27.8 / 30.2 / 28.1 and code 37.0 / 39.7 / 38.6 across 1k to 65k. The 64k spread is within run-to-run noise, which confirms the earlier gate measurement.
- **TTFT is linear.** 0.53 s, 11.3 s, 22.9 s for 1k, 32k, 65k, i.e. about 0.35 ms per 1k tokens.
- **Speed is acceptance, not step time.** Steps/s is stable at 13-14; tokens/step ranges from 1.99 (prose) to 4.55 (copy). The acceptance column is the lever.

## Comparison with the published single-node reference

The upstream single-node project reports, on the same class of hardware: **48.7 tok/s, 61.5 ms per engine step, 3.00 tokens/step** for prose at one stream, with MTP 3, `MAX_NUM_SEQS=8`, BF16 recurrent state and a FULL decode graph.

Ours on the same content type: **27.8 tok/s, 75.2 ms per step, 2.09 tokens/step**.

| Factor | Reference | Ours | Difference |
|---|---|---|---|
| Step time | 61.5 ms | 75.2 ms | +22% slower |
| Tokens per step | 3.00 | 2.09 | −30% |
| Decode | 48.7 tok/s | 27.8 tok/s | −43% |

The larger term is tokens/step, not step time. That is acceptance: the draft head agrees with the target on 44% of prose positions here. The reference also runs a shallower draft (3 vs 4), a larger `MAX_NUM_SEQS`, BF16 recurrent state and a different graph-capture set, so the comparison is a hypothesis, not a controlled A/B. The prompt sets also differ, and this run uses 200 tokens against their 600.

**Do not quote ">50 tok/s" for this box until that gap is explained.** On the dual-node production deployment the arithmetic is different; this log measures the single-node box only.

## What was already checked

- **CUDA-graph capture sizes.** The boot log shows `cudagraph_capture_sizes = [1, 2, 4, 5, 8, 10, 16, 20, 24, 32, 40]`, so the MTP batch of 5 rows already has an exact graph. The 5-to-8 padding fix from the DeepInfer audit does not apply here. No experiment needed.
- **Loop-guard scan.** The proxy's repetition detector scanned 662 candidate periods per chunk. A one-byte comparison before building each slice cut it from 161 to 85 µs per call, with identical verdicts on the regression corpus. Kept.
- **Pinned policy skill.** The proxy now inlines a pinned skill's body instead of a path. Functional, not a speed change. Kept.

## Next experiments, in order

1. **Draft depth 3 versus 4** on identical texts. The reference runs 3 and reports 3.00 tokens/step; we run 4 and see 2.09. If depth 3 wins on prose, the extra verify row is not paying for itself.
2. **`MAX_NUM_SEQS = 8`** with the full graph capture set the reference uses. Step time and acceptance both move with the capture configuration.
3. **BF16 recurrent state** (`mamba_ssm_cache_dtype`) against the current setting; the reference measures +8.5% at eight streams.
4. **`gpu_memory_utilization` 0.72 versus 0.80.** This changes the KV pool, not the decode math, but the reference runs 0.80 and the comparison should not carry a second variable.
5. **Prefix-cache hashing with CBOR** (`--prefix-caching-hash-algo sha256_cbor`). Expected 3.5 ms per 64k request, zero decode effect. Config-only; needs a restart.
6. **Proxy micro-optimizations:** one batched embedding call per request, `math.sumprod` cosine with cached norms, and skipping the request JSON parse when no proxy feature is enabled. Measured CPU savings of 1.5 ms, 0.2 ms and 1.2 ms per request respectively.

## Reproduction

```bash
cd ~/Qwen3.8-Flash-Next-Dual-DGX-Sparks
BENCH_BASE=http://localhost:8888 BENCH_MODEL=Qwen3.8-Flash-Next-NVFP4 \
  python3 bench/decodebench.py --decode 200 --contexts 1000,32000,65000 \
  --jsonl bench/data/exp_baseline2.jsonl
```

Raw rows are appended to `bench/data/exp_baseline2.jsonl` on the host. The boot log for the configuration above is the `vllm-fn` container from 2026-10-24, with `tensor_parallel_size=1`, `kv_cache_dtype=fp8`, and `speculative_config` at 4 drafts with `disable_eagle_block_drop=true`.
