# Serving experiments

A running log of changes to the serving path: what was measured, what was kept, and what was rejected. Negative results are kept on purpose so the same experiment is not repeated.

## Scope of these numbers

Every row in this file was measured on one box with `tensor_parallel_size = 1`. Production runs across two boxes with tensor parallelism, so the numbers divide into two kinds and only one of them transfers:

- **Transfers.** Anything counted in tokens: acceptance per draft position, tokens per verify step, prompt tokens, completion tokens, thinking tokens, turns per task. These are properties of the model, the draft head, the prompt and the proxy; splitting the model across two ranks does not change which tokens the model would have produced.
- **Does not transfer.** Anything counted in seconds: decode tok/s, engine step time, TTFT, the cost of one extra verify row, concurrency scaling. A step at two ranks pays an all-reduce per layer per row, so a row that costs 4% of a single-box step can cost more on the pair, and the KV pool is per-rank instead of unified.

So a single-box measurement can accept a token-count change and reject a *performance* change, but it cannot set production performance expectations. Where a verdict below depends on step time, it is a single-box verdict and is labelled as one.

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

## Draft depth 3 instead of 4 (rejected)

The reference runs three drafts and reports 3.00 tokens/step; this build runs four and reports 2.09. If the fourth row does not pay for itself, a shallower chain should win. It does not.

Arms: identical recipe, identical texts, identical context nonces, one stream, temperature 0, 200 decoded tokens. The only difference is `mtp_num_speculative_tokens`, 4 on the baseline above and 3 here.

| context | content | TTFT (s) | decode (tok/s) | steps/s | tokens/step | acceptance |
|---|---|---|---|---|---|---|
| 1,000 | prose | 0.55 | 27.4 | 14.7 | 1.86 | 44.9% |
| 1,000 | code | 0.52 | 39.1 | 14.9 | 2.63 | 67.0% |
| 1,000 | entropy | 0.51 | 30.1 | 15.3 | 1.97 | 46.9% |
| 1,000 | copy | 0.52 | 53.8 | 15.1 | 3.59 | 94.2% |
| 32,000 | prose | 11.25 | 32.5 | 14.0 | 2.33 | 56.2% |
| 32,000 | code | 11.22 | 37.5 | 14.7 | 2.58 | 63.4% |
| 32,000 | entropy | 11.26 | 38.9 | 15.4 | 2.54 | 61.9% |
| 32,000 | copy | 11.27 | 58.8 | 15.1 | 3.92 | 97.4% |
| 65,000 | prose | 22.73 | 29.8 | 14.1 | 2.12 | 50.0% |
| 65,000 | code | 22.74 | 39.8 | 14.6 | 2.74 | 66.1% |
| 65,000 | entropy | 22.73 | 43.1 | 15.4 | 2.85 | 68.9% |
| 65,000 | copy | 22.75 | 55.4 | 14.7 | 3.81 | 96.1% |

What it shows:

- **One fewer verify row buys 5-8% of step rate**, uniformly: 13-14 steps/s becomes 14-15.4 at every context.
- **Acceptance per row goes up** when the fourth draft is removed (32k prose 46.7% → 56.2%), which confirms the fourth position is the weakest one: most of those rows end in a rejection.
- **But the cap binds on predictable content.** Copy loses 13-15% at every context and entropy loses 22% at 1k, because those tasks were spending the fourth row well (copy ran 4.42 tokens/step on the baseline).
- Prose gains at long context (+7.6% at 32k, +6.0% at 65k) and is flat at 1k (−1.4%); code is mixed (−5.5% at 32k, +5.7% at 1k, +3.1% at 65k), which is the shape most agent traffic has.

Mean over the twelve cells: **40.5 tok/s against 42.8 on the baseline**, so the change loses on one box. `mtp_num_speculative_tokens` stays at 4.

Read this verdict with the scope note above in mind. What transfers is the token count: the fourth draft is accepted often enough to add 0.23 tokens per step on prose and 0.83 on copy, and no topology changes that. What does not transfer is what the extra row costs. One more verify row is one more all-reduce per layer on a two-rank deployment, so the row is relatively more expensive there and the trade could flip. This is the one experiment in this file that deserves a repeat on the dual-node deployment before the question is closed.

## Concurrent streams on one rank

Same build as the baseline above, with `max_num_seqs` raised from 4 to 8 and nothing else changed. Each stream asks for the same prose task, 200 tokens, temperature 0.

| streams | aggregate (tok/s) | per stream (tok/s) |
|---|---|---|
| 4 | 59.9 | 15.0 – 19.9 |
| 8 | 95.1 | 11.9 – 12.9 |

Four streams were measured twice on the depth-3 arm and once here and land in the same 51-62 range, so treat the four-stream row as ±10%. Eight streams add 59% aggregate while per-stream rate falls 21%, which says the box is not saturated at four.

Single-stream rows are unchanged by the higher limit: 1k copy 61.2 against 61.7, 65k prose 28.2 against 28.1, 65k copy 65.4 against 62.0. One row moved a lot (32k code 34.3 against 39.7) with no mechanism behind it, so it is recorded as unexplained rather than as a regression.

This is a capacity measurement, and capacity is the part that does not transfer: at two ranks the KV pool is per-rank and each step pays an all-reduce, so eight streams may scale differently. It is not evidence for the deployment.

## Prompt-cache reuse, measured

The point of the prompt-layout work is that an injected block must not invalidate the cached prefix. `cached_tokens` in the response usage measures exactly that: send the same conversation twice and read how much of the second prompt the server did not have to prefill.

| path | prompt tokens | cached on the repeat | share |
|---|---|---|---|
| straight to the server | 6,072 | 5,040 | 83.0% |
| through the proxy, pinned skill | 6,794 | 6,720 | 98.9% |

The proxy adds 722 prompt tokens for the pinned skill body, and the repeat still prefills only 74 of 6,794 tokens. Two things follow: the injection is byte-identical across requests, so it is paid once and cached thereafter, and a 6k prompt costs about 17% of one prefill per turn instead of 100%.

This number does transfer. Prefix caching is per-rank, but the tokens that hit are the same tokens on either topology; a byte that changes in the head costs a full re-prefill per rank, so the saving if anything doubles with two ranks.

## What a five-turn session actually costs

Prefix caching helps only if the session grows by appending. This measures the real curve: a 4k-token document, then four turns that each append a short answer, a ~2k-token tool result and the next question. Direct to the server, no proxy, temperature 0.

| turn | prompt tokens | served from cache | new prefill | completion |
|---|---|---|---|---|
| 1 | 4,077 | 0 (0.0%) | 4,077 | 64 |
| 2 | 6,756 | 3,360 (49.7%) | 3,396 | 64 |
| 3 | 9,430 | 6,720 (71.3%) | 2,710 | 64 |
| 4 | 12,104 | 8,400 (69.4%) | 3,704 | 64 |
| 5 | 14,801 | 11,760 (79.5%) | 3,041 | 64 |
| **total** | **47,168** | **30,240 (64.1%)** | **16,928** | **320** |

Two numbers matter downstream:

- **Prompt tokens outnumber generated tokens 147 to 1** over the session. Any work on the prompt side moves more than any work on the decode side, and unlike decode it behaves the same on one rank or two.
- **The marginal prefill is 2.7-3.7k tokens per turn**, and about 2k of that is the new tool output. That is the target: a tool result is paid for once when it arrives and again in every later turn's prompt, so shrinking or de-duplicating it attacks the only part of the prompt the cache cannot absorb.

## Running the open question on the pair

The depth question is the only entry here that needs the two-rank deployment. It is a sequential A/B, one recipe change, no code:

1. Boot the committed recipe and run the reference arm. Snapshot the counters, run the workload, then diff them, so the acceptance is a delta and not the server's lifetime average:

   ```
   python3 bench/mtp_accept.py --save /tmp/acc_before.json
   BENCH_BASE=http://localhost:8888 BENCH_MODEL=Qwen3.8-Flash-Next-NVFP4 \
     python3 bench/decodebench.py --decode 200 --contexts 1000,32000,65000 --temps 0.0
   python3 bench/mtp_accept.py --since /tmp/acc_before.json
   ```

2. Pin `mtp_num_speculative_tokens: 3` in the recipe, boot, repeat step 1 into a second file.
3. Boot the committed recipe again and repeat within the hour. Per `docs/bench-gates.md`, a delta under 2% is not a win until the baseline has been re-run next to it.

Compare **ms/step and tokens/step**, not tok/s: acceptance moves with the text, so tok/s compares texts rather than configurations. The decision rule that matters is the same one this file used on one rank — does the fourth row buy more tokens per step than the step time it costs. On the pair the row also costs an all-reduce per layer, so the bar it has to clear is higher.

The before/after matrix runs on the same deployment for the same reason, and its commands are unchanged from the ones above; only the deployment is different.

## Duplicate tool output, and per-request accounting

A tool result that repeats verbatim is the one part of an agent transcript that is pure waste: it is paid for in every later turn's prompt. Measured through the proxy with two byte-identical ~30k-token tool results in one conversation:

| arm | prompt tokens |
|---|---|
| default (off) | 60,172 |
| `PROXY_DEDUPE_TOOL_OUTPUT=1` | 30,195 |

The later copy becomes `[identical to the tool result above at message N; B bytes]`, so the model still learns that a repeat happened and how large it was without re-reading it. Only exact bytes, only `role: "tool"`, only above 512 bytes; the first occurrence and every non-tool message stay byte-identical, and the replacement is idempotent across turns.

It stays opt-in. That is a meaning change, and this benchmark has no repeated tool results, so nothing here answers the quality question — it needs a task set where the duplicate actually occurs.

Every completed request now also logs its own cost:

```
score proxy: usage prompt_tokens=60172 cached_tokens=0 completion_tokens=16 reasoning_tokens=16
```

That line is the attribution mechanism for everything above: it separates "the prompt grew" from "the model thought longer", which is otherwise indistinguishable from the outside.

## Reasoning and long-context baseline

`bench/reasoning_check.py`, ten short reasoning tasks plus two long-context recall probes, temperature as served, 2,048-token generation cap:

| | |
|---|---|
| score | **12 / 12** |
| prompt tokens | 77,522 (11,760 served from cache) |
| completion tokens | 2,407 |
| of which reasoning | 1,659 (69%) |
| long-context probe | fitted to 63,482 prompt tokens, PASS in 24.8 s |

Three things this fixes for later comparisons:

- **Thinking is 69% of everything the model generates here.** A budget on the reasoning trace is the only lever that moves completion tokens by a large fraction, and this is the number any such change has to be measured against.
- **The score is 12/12 on the current build**, so a later change that drops a task is a regression, not noise.
- **The long-context probe now fits the served window before it is sent.** It previously asked for a prompt above `max_model_len` and recorded a 400, which read like a model failure in the score.

## Two request-shape findings

Measured against the live server while preparing the prompt-side work:

- **A prompt plus `max_tokens` over the window is not an error.** 57,563 prompt tokens with `max_tokens: 8192` asks for 65,755 against a 65,536 window and succeeds; generation stops at the window with `finish_reason: length`. A pre-send clamp that lowers `max_tokens` to fit therefore changes nothing — the answer ends at the same token either way — so there is no guard to add here.
- **An over-long prompt is an error, and it is immediate.** A prompt above `max_model_len` returns 400 before any prefill, so there is no wasted work to recover either. The only way to make such a request succeed would be to drop conversation content, which is not lossless.

Together these remove a whole class of "fit check" work from the plan: the server already does the right thing in both directions.

## Already settled by configuration

- **BF16 recurrent state.** `mamba_ssm_cache_dtype: bfloat16` is in the committed recipe and visible in the live container arguments, so the reference's +8.5% is already in the baseline above. No A/B left to run.
- **`gpu_memory_utilization` 0.80.** Rejected on record: 0.80 fails the start-up free-memory check on this class of box, and 0.75 lets the driver fail allocations under load. 0.72 is the highest value that boots and stays up, so the comparison with a 0.80 reference carries a genuine capacity difference that cannot be removed.
- **Prefix-cache hashing with CBOR.** The flag exists in this vLLM build (`prefix_caching_hash_algo` accepts `sha256_cbor`) and `cbor2` is importable in the image, so wiring `--prefix-caching-hash-algo sha256_cbor` would work. Its ceiling is about 4 ms per 64k request against a 23 s prefill, which is below run-to-run noise. Not wired; revisit only if prefix hashing ever shows up in a profile.
- **`--performance-mode interactivity`.** Read the v0.32 source: only `throughput` changes anything (it doubles `max_num_batched_tokens` and `max_num_seqs`); `balanced` and `interactivity` are no-ops. Not a lever.

## Next experiments

Single-rank performance work stops here: step time, decode rate and concurrency scaling on one box do not predict the deployment, so every remaining performance change belongs on the dual-node box. What one box can still decide is anything counted in tokens, and that is where the next work goes:

1. **`max_num_seqs`** stays derived from the node count (4 on one box, 8 on two). The single-box A/B above shows no single-stream effect, so there is nothing to pin.
2. **Prompt and completion tokens per task.** `bench/reasoning_check.py --out ... --compare ...` now totals `prompt_tokens`, `cached_tokens`, `completion_tokens` and `reasoning_tokens` and prints the delta against a baseline file. Run it before and after every token-side change.
3. **Proxy micro-optimizations:** one batched embedding call per request, `math.sumprod` cosine with cached norms, and skipping the request JSON parse when no proxy feature is enabled. Measured CPU savings of 1.5 ms, 0.2 ms and 1.2 ms per request respectively.

## Reproduction

```bash
cd ~/Qwen3.8-Flash-Next-Dual-DGX-Sparks
BENCH_BASE=http://localhost:8888 BENCH_MODEL=Qwen3.8-Flash-Next-NVFP4 \
  python3 bench/decodebench.py --decode 200 --contexts 1000,32000,65000 \
  --jsonl bench/data/exp_baseline2.jsonl
```

Raw rows are appended to `bench/data/exp_baseline2.jsonl` on the host. The boot log for the configuration above is the `vllm-fn` container from 2026-10-24, with `tensor_parallel_size=1`, `kv_cache_dtype=fp8`, and `speculative_config` at 4 drafts with `disable_eagle_block_drop=true`.
