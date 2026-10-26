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

## After the loader change

The same matrix on the boot that uses `runai_streamer` (see `docs/load-time.md`;
271 s load, 368,009 KV tokens against 180-240k before). TTFT and step rate are
unchanged — 0.56 / 11.4 / 23.0 s at 1k / 32k / 65k, about 13 steps/s — so the
loader moved bytes, not the decode loop.

| content | 1k temp 0 | 32k temp 0 | 65k temp 0 |
|---|---|---|---|
| prose | 27.1 | 25.5 | 27.9 |
| code | 37.3 | 39.2 | 35.7 |
| copy | 65.4 | 59.8 | 64.7 |

Those are inside the spread of the earlier boot, so the loader did not change
decode throughput.

**Pool size is not a measurement.** The figure quoted above is what a container
restart reports, and a restart also skips the launch-time checkpoint page-cache
release. But clean `./start.sh` boots of the *same* recipe are no more stable:
they have reported 414,389, 422,455, 444,636, 462,784, 499,081 and 578,733 tokens
(6.32x to 8.83x at 65,536). A 40% swing on identical configuration means pool
size cannot rank two configurations, and no claim in this file rests on it.

**The `entropy` row is not a comparable measurement.** Its content is regenerated
per run, and three runs on one boot gave 30.0, 44.1 and 38.6 tok/s at 1k
(acceptance 47.1%, 66.2%, 57.1%). A single sample differs from another by more
than any change worth measuring, so entropy rows from single runs — including
the ones in the table above it — should not be used to compare configurations.
Copy repeats to within 2% and is the row to judge a small change on; prose does
not repeat as tightly as the sentence this replaces assumed — see below.

## Which rows actually repeat

The corpus is stable across runs: `--salt` only prepends a one-line nonce to a
shared filler body, so every run serves the same text and the same n-grams. The
rows are nevertheless not equally repeatable. Three runs of the same 1k matrix on
one boot, nothing else running:

| run | prose | code | copy |
|---|---|---|---|
| 1 | 15.5 | 28.9 | 67.4 |
| 2 | 19.0 | 31.0 | 64.3 |
| 3 (inside the 1k/48k pass) | 21.6 | 31.2 | 66.0 |

Copy repeats to about 2%, code to about 4%, prose to about 16% — and the prose
step rate moved with it, 8.4 to 11.5 steps/s at a fixed acceptance near 40%, so
this is whole-engine jitter rather than text. A single prose row cannot decide a
configuration; copy can. That also bounds the numbers above: three samples on
this boot span 15.5-21.6 and all sit below the earlier boot's single 27.1, which
is either a real regression or a high draw on the other side. Copy shows no such
change (67.4 / 64.3 / 66.0 against 65.4 recorded), which argues against a general
decode regression.

Baseline on the boot that runs `runai_streamer`, 1k and 48k, both temperatures
(`--temps 0 0.8`, 600 decode tokens):

| context | content | temp 0 | temp 0.8 |
|---|---|---|---|
| 1k | prose | 21.6 | 19.4 |
| 1k | code | 31.2 | 16.7 |
| 1k | copy | 66.0 | 59.6 |
| 48k | prose | 14.5 | 15.6 |
| 48k | code | 22.0 | 22.7 |
| 48k | copy | 62.8 | 63.7 |

TTFT was 0.52-1.17 s at 1k and 16.9-41.8 s at 48k; acceptance held at 39-40%
(prose), 58-64% (code) and 97-99% (copy) in every bucket, so the acceptance lever
moved the same way as before. The 1k code row at temp 0.8 (16.7 against 31.2) is
one sample each and falls in the jitter measured above; repeat it before reading
anything into that pair.

## A 131,072-token cap on one node

`MAX_MODEL_LEN` is env-overridable and is not a recipe key, so this gate is a
single boot: `MAX_MODEL_LEN=131072 RECIPE=prod ./start.sh --launch
--no-download` — 514 s to healthy, with `--max-model-len 131072` confirmed on the
live process.

| boot | cap | KV pool | concurrency at the cap |
|---|---|---|---|
| prod | 65,536 | 414,389 | 6.32x |
| gate | 131,072 | 701,991 | 5.36x |

Both pool figures are single readings, and the same recipe alone swings 414-579k
across boots, so the larger pool here is **not** attributable to the cap and is
not claimed as one. What the gate establishes is that a 100k-token request is
served:

| run | prompt tokens | TTFT | prefill | decode | needles found |
|---|---|---|---|---|---|
| 100k, 16 output tokens | 100,117 | 39.26 s | 2,550 tok/s | 32.5 tok/s | 0 of 3 |
| 100k, 96 output tokens | 100,116 | 35.65 s | 2,808 tok/s | 61.7 tok/s | 2 of 3 |

The first row is not a result: a three-item answer cannot fit in 16 tokens. At 96
tokens the model found two of the three needles, so the context is served but
retrieval inside it is not reliable. The committed default stays 65,536; 131,072
is available to a caller that needs it and accepts both the 35-39 s TTFT and that
retrieval rate.

## The FP8 target head, gated

The head is the full 248,320-row projection that feeds sampling *and* target
verification, so a lossy copy of it must pass the distribution gate before it
earns a place in the default. An E4M3 copy with per-row scales cuts the per-step
head read from about 4.6 ms to 2.3 ms of a ~77 ms step at ~273 GB/s.

| arm | p99 KL | floor p99 KL | top-1 | scaled top-1 gate | perplexity |
|---|---|---|---|---|---|
| stock (capture) | — | 1.730 | 90.09% | — | 3.16 |
| FP8 target head | 1.660 | 1.730 | 89.60% | 88.09% | 3.16 (+0.005) |

Decode at 1k, temperature 0, three runs per arm, reported as steps/s because
ms/step is its reciprocal and steps/s is what the engine emits:

| row | stock | FP8 head | delta |
|---|---|---|---|
| copy | 13.5 / 13.7 / 13.7 | 14.2 / 14.2 / 14.2 | **+4.2%** |
| prose | 12.7 / 12.8 / 12.9 | 12.8 / 12.7 / 13.4 | +1.3% |

Copy is the row that repeats, and it moved by the same amount in every run with
no overlap between the arms: **-4.0% ms/step**. The prose row is inside its own
16% spread and cannot corroborate anything. The arm's boot reported KV 444,636
tokens, but pool size swings 414-579k across clean boots of one recipe, so no
pool gain is claimed. On the boot served by the committed recipe the same replay
gives p99 KL 1.671 and top-1 89.72%, and copy runs 14.0 / 14.2 steps/s, so the
shipped boot reproduces the arm at +3.4 to +4.2% against the stock 13.5-13.7.

Both conditions met, so `target_head_fp8: true` is committed. The arm was booted
from a one-key variant of the same recipe, and `engine/recipe.py` resolves the
committed recipe to a byte-identical environment to that variant, so the boot
that passed is the boot that ships.

## The FP8-dense hybrid (`recipes/fp8.yaml`)

A second checkpoint, `~/models/q38-lean-hyb`: NVFP4 routed experts with FP8
per-channel dense projections, served from its own pinned image, MTP 3 instead of
4. Boot 863 s, no errors. Pool size is not comparable across boots (see above), so
the hybrid is ranked on the gates alone.

| gate | shipped default | q38 |
|---|---|---|
| p99 KL (self-noise floor 1.939) | — | **1.577 PASS** |
| top-1 (floor 88.98%, relaxed 86.98%) | — | 90.45% PASS |
| mean KL, against a 0.1215 floor | — | 0.1515 |
| perplexity delta | — | +2.6% |
| decode copy, steps/s | 14.0 / 14.2 | 17.3 / 17.6 |
| decode copy, tokens per step | 4.79 / 4.79 | 3.69 / 3.76 |
| decode copy, tok/s | 66.9 / 68.1 | 63.7 / 65.8 |
| 18-task ruler | 15/18 | 16/18 |

The hybrid buys step rate and gives it back in tokens per step — the shallower
draft costs a token per step — so aggregate throughput is flat to slightly worse.
Quality sits inside the distribution floor on p99 KL and moves the wrong way on
mean KL and perplexity, and the +1 task is inside a ruler that flips five of
eighteen tasks between repeats. It stays a measured alternative, not the default:
it also needs a second 120 GB checkpoint and a pinned image.

One instrument note. The KL gate could not run at first because this recipe
never declared `max_logprobs`, and the engine's default of 20 refuses the
yardstick's 64-wide request — prod carries that declaration for the same reason;
`recipes/fp8.yaml` now does too. The fourth gate (replay of a recorded session)
is measured on the default above; the hybrid has no replay row yet.

### Composed draft vocabulary at one node: rejected by the memory envelope

The 47k draft vocabulary could not be combined with the FP8-dense draft head
before: both rewrite the same `mtp.py`, and the slice assumed BF16 rows. The
composition now works — the slice reads the quantized head and carries its
per-row scale, and the arm's own log confirms it:

```
MTP draft vocab: torch.float8_e4m3fn rows dequantized with their scale for the draft head.
MTP draft vocab: 47149 of 248320 tokens (19.0%); draft lm_head shard 0.59 -> 0.22 GiB per draft step
```

It has no numbers, because it did not finish. After the weights loaded (73.2 GiB
in 624.7 s) the engine core logged `No available shared memory broadcast block
found in 60 seconds` five times over four minutes, then the driver returned
`NV_ERR_NO_MEMORY` sixteen times in twenty-six seconds, the container exited 255,
and the host rebooted. The log never reached `GPU KV cache size`, so neither
decode nor acceptance could be measured, and the boot after the reboot restored
the default.

This is a rejection on the envelope, not on quality: the hybrid needs a second
120 GB checkpoint and is built for the pair, and adding the slice changes the
draft head's residency at TP=1 without buying back the dense-projection memory.
A repeat needs either two nodes or a lower `gpu_memory_utilization`, and it costs
another reboot if it is wrong — so it waits for the pair.

## Real-session replay, the gate whose unit is a turn

Three of the four gates score a bench. This one replays the prompts of a recorded
agent session unchanged, so the unit is a turn of real work at real prompt sizes.

Source: a 3,842-request agent session, extracted with
`bench/session_replay.py extract --window-tokens 12000 --sample 24`, which keeps
the start, the middle and the end rather than the first 24 turns whose prompts all
look alike. Prompts total 301,344 tokens (p50 13,543, range 669-16,245); every
reply is capped at 128 tokens.

| arm | turns | wall | wall per turn | completion | reasoning |
|---|---|---|---|---|---|
| shipped default | 24 | 559.5 s | 23.3 s (5.5-35.3) | 3,049 | 1,931 |

That is 539 prompt tokens per wall-second once the decode is included, so at these
sizes a turn is dominated by prefill: the 669-token turn finishes in 5.5 s, the
16k one takes 35 s. This is the row a change has to move, and the row the other
gates' ms/step deltas get translated through.

### Where the 23 s goes

`send` now streams each reply, so a turn splits where the engine splits it: the
first token marks the end of prefill, the rest is decode, and the per-request
`cached_tokens` says how much of the prompt the prefix cache served.

| | per turn | share | rate |
|---|---|---|---|
| prefill (time to first token) | 19.4 s | 84% | 647 prompt tok/s over 465.6 s |
| decode | 3.6 s | 16% | 34.5 tok/s on 128 tokens |
| wall | 23.0 s | | 552.4 s for 24 turns |
| prefix cache | 0 of 301,344 tokens | 0% | |

Prefill is the cost, and it is 4-6x slower than the same engine's measured bench
prefill (2,550-3,850 tok/s at 100k): a 669-token turn still costs 2.1 s of prefill,
and the per-turn prefill grows with the prompt at roughly 550 tok/s. Since decode
lands where the arithmetic says it should, the gap is entirely in what happens
before the first token. Named candidates, none yet separated: no prefix reuse, the
cold PLE n-gram gathers (the 158x first-pass penalty documented under concurrency,
which applies to every genuinely new prompt), and the 4,800-token chunk boundary
with its per-chunk host gather.

The zero cache share is a property of **this sample**, not of the server: the 24
requests are spread across a 3,842-turn session, so they share only the system
prefix. A consecutive window is the measurement that separates "prefill is slow"
from "these prompts never reuse anything", and it is the next run.

Sampling is even, not random (`--sample 24`), so the mix of prompt sizes is
representative while any adjacency effect is not.

### Locality: the 23 s is a cold-prefix upper bound

Eight consecutive turns of the same session (prompts 12,848-14,121 tokens),
sent three times: in order cold, then the identical eight reversed, then in order
again. Same prompts in all three, so content is fixed and only residency changes.

| run (8 turns, 108,142 prompt tokens) | cache share | mean prefill | mean decode | wall per turn |
|---|---|---|---|---|
| consecutive, first pass (cold) | 57% | 4.85 s | 1.97 s | 6.8 s |
| the same eight, reversed (second) | 95% | 0.49 s | 2.08 s | 2.6 s |
| consecutive, second pass | 95% | 0.47 s | 2.01 s | 2.5 s |

The clean paired delta is the first and third rows — identical order, identical
content, one rerun apart: prefill **-90.3%**, 10.3x. The reversed run is not an
adjacency test in the end: it ran second, so every prefix was already resident,
and it found 95% cached regardless of order. Residency, not order, is what the
cache rewards.

So the 23.0 s spread row and this 2.5 s consecutive row are not in conflict: the
spread sample sends 24 turns whose histories the server has never seen, so every
one pays a cold prefill. A real session sends its turns in order, and the previous
turn's prefix is already in the pool.

The cold cost itself is not flat. Within the first pass, one turn with a cold
prefix (12,848 tokens) took 19.4 s — 662 tok/s — while two later cold turns of
13.3k and 13.9k took 6.2 s and 6.4 s — about 2,150 tok/s. The first cold prefill
of a boot therefore carries roughly 13 s that later cold prefills do not, which is
the shape the cold PLE-ngram gather predicts: it is paid once per content set, not
once per token.

Three of the eight consecutive turns still reported 0 cached (turns 1900, 1903,
1906) even on the second pass, so their prefixes genuinely differ from their
predecessor's — a new tool result or a compaction boundary. Those are the turns a
cache cannot help.

What this means for the goal: a warm consecutive session already runs at 2.5 s per
turn, well inside the 12 s target; the spread replay is an upper bound for
never-seen content. The remaining measurable lever is the cold path — the ~13 s
one-off first-prefill penalty and the ~2.1k tok/s steady-state cold prefill
against the engine's 3.8k bench rate.

### The cold penalty is per content region, not per boot

Five regions, four far-apart blocks plus a single probe, all sent once in four and
a half minutes on a boot whose warmup had finished nine minutes earlier, with
`running` and `waiting` at 0 at every boundary:

| block | first turn | first prefill | first cold tok/s | later cold turns in the block |
|---|---|---|---|---|
| A, turns 200-207 | 200 | 19.62 s | 643 | 2,191 / 2,611 / 2,471 |
| B, turns 1000-1007 | 1000 | 20.45 s | 714 | 2,271 / 2,327 / 2,509 / 2,336 / 2,144 / 1,817 |
| C, turns 2400-2407 | 2400 | 30.68 s | 490 | 2,391 / 1,919 / 1,937 / 1,972 |
| D, turns 3300-3307 | 3300 | 18.65 s | 540 | none further cold |
| E, turn 1500 alone | 1500 | 19.41 s | 690 | — |

A boot-time warmup can only be paid once; this was paid five times. Across the 13
non-first cold turns the median is **2,271 tok/s** (range 1,817-2,611), so each new
region carries **+13.5 to +24 s** over the steady rate. The probe rules out
"first request of a process": it was alone in its invocation and still cost 690
tok/s. A control sending four *different* regions inside one invocation gives
656 / 787 / 740 / 712 tok/s — four payments, one process.

The mechanism that fits is the PLE n-gram table: 51.2 GB, mmapped with
`MADV_RANDOM`, roughly a 4 KB fault per miss, ~19.8 MB/s at queue depth 1. A new
region touches rows nothing has read since boot; that region's later turns reuse
them, so they are fast even when their prefix cache misses (turns 202-204 report
`cached 0` and still run at 2,191-2,611 tok/s). This run has no fault counters, so
the mechanism is the leading explanation, not a measurement.

That suggested a real in-order session would pay the penalty once, at its first
prompt. It does not.

### A real 40-turn stretch crosses the boundary six times

Window 1100-1139 sent once, in order, on a live boot: 382.8 s of wall, 9.57 s per
turn (median 7.51), mean prefill 7.63 s, mean decode 1.93 s, cache share 32.5%.

| turn | prompt | cached | prefill | cold tok/s |
|---|---|---|---|---|
| 1100 | 13,575 | 0 | 22.81 s | 595 |
| 1117 | 11,275 | 0 | 20.31 s | 555 |
| 1118 | 11,826 | 0 | 16.30 s | 726 |
| 1127 | 11,841 | 0 | 21.88 s | 541 |
| 1132 | 5,802 | 0 | 6.75 s | 860 |
| 1136 | 12,136 | 0 | **49.31 s** | 246 |

Six of forty turns (15%) paid the cold penalty, and they carry **108.1 s — 28.2%
of the wall clock**. So it is not once per session: an agent session diverges
whenever a turn brings back new text (a file read, a command output, a compaction
boundary), and each divergence starts a new region. Two of the six are adjacent
(1117, 1118), which rules out any per-request or per-block explanation.

Turn 1136 is the worst single prefill measured: 49.3 s for a 12,136-token prompt at
246 tok/s, 4.6x the steady-rate prediction. Cause not identified; it is the same
shape as the others and no larger.

The ceiling this sets: if the cold path ran at the steady 2,271 tok/s, the same
forty turns would cost about **275 s instead of 383 s**. The threshold used here
(prefill > 2x the steady-rate cost of the uncached remainder) flags 14 turns, but
eight of those are warm turns whose small uncached remainder cannot absorb fixed
per-request overhead — the six fully cold turns are the real set.

### Prefetching the cold rows: the tail, not the plateau

`ple_prefill_prefetch` (default on) has a host thread pool pread a prefill
chunk's distinct n-gram rows before the GPU gather reads them. It is the
capture-safe half of the staged lookup: prefill is eager, so there is no pinned
allocation inside a device context, no side-stream join and no graph to upset. It
joins before returning, so the pages are resident when the gather is enqueued, and
it fails closed to the serial path on any exception.

Same window (turns 1140-1179, 490,751 prompt tokens), both runs opened on a
dropped page cache:

| | off | on |
|---|---|---|
| wall | 284.1 s | **233.3 s** |
| prefill total | 198.1 s | 153.6 s (−22.5%) |
| mean prefill | 4.95 s | 3.84 s |
| fully-cold turns | 23 | 24 |
| cold tokens/s, whole set | 1,889 | 2,333 (1.235x) |
| worst prefill | 17.09 s | **7.93 s** |
| major faults during the run | +121,360 | **+40,026** |

The gain is not uniform, and the split is the interesting part. Three turns that
had fallen to 250-1,000 tok/s (1149, 1166, 1168) go from 45.64 s to 20.72 s —
**2.20x**. The other twenty cold-in-both turns go from 2,266 to 2,445 tok/s —
**1.079x**. So the mechanism removes the pathological tail and leaves the steady
cold rate where it was: after this, no cold turn exceeds 7.9 s, and the
"six cold turns carry 28% of the wall" shape is gone.

The mechanism is confirmed by the fault counters rather than by the timings: the
same content, same cold start, 121,360 major faults without the prefetch against
40,026 with it. Prefetching into a buffer does not take major faults; the gather
then finds the pages resident.

Confounds, stated because they matter: the two runs are different boots, and
decode moved 29.8 to 32.1 tok/s (−6.3 s of the 50.8 s wall gain), so the
attributable figure is the **prefill delta of −44.5 s**. The two arms' starting
page-cache floors differ (7.4 GB against 1.2 GB, both dropped the same way), and
one turn had a 3,360-token prefix hit in the off arm against none in the on arm —
both of which work against the prefetch. One run per arm, no error bars; per-turn
jitter is about ±0.5 s, well under the effects claimed.

The extractor was the blocker until now. It rebuilt the transcript with
`json.loads(json.dumps(history))` at every assistant turn — quadratic — so a 22 MB
session ran for minutes and produced nothing. With a shallow snapshot of a history
whose entries are never mutated afterwards, the same extraction takes 1.3 s, and
`tests/test_session_replay.py` pins both the output shape and the absence of that
serialisation.

## Concurrency: the stream cap, and what a long prefill does to every stream

### The cap is `max_num_seqs`, not KV

`max_num_seqs` is derived from the topology — 4 on one node, 8 on two — because
the PLE offload that makes the model fit at TP=1 halves the per-rank footprint
when a second node arrives. It is not a KV ceiling:

| observation | value |
|---|---|
| single node, 8 concurrent requests | 4 running, the rest queued |
| single node, 4 streams x 32k, KV peak | 0.45-0.49 of the pool |
| two nodes, 8 running + 7 waiting | 26% of a 2.7M-token pool |

Throughput at the cap is not the problem either: eight runs of four streams at
32k prompt landed between 91 and 154 tok/s aggregate, against 27.8 tok/s for a
single prose stream. Batching buys about 4.5x aggregate for about 1.1x per
stream, which is what a weights-bound decode loop should do.

### A long prefill starves decode (systematic)

Three decoders each 32k into their turn, plus one 60k-token prefill arriving
three seconds later. Decoder-only throughput, per second:

`22 83 154 48` then `0 0 0 20 0 0 0 16 ...` for about 250 s, then `66 80 94 95
78 76 83 118` once the prefill drains.

The arithmetic is exact. A step spends its whole `max_num_batched_tokens`
budget (8,192) on prefill, and prefill runs at about 3,850 tok/s, so that step
takes about 2.1 s. Each decoding stream advances only its verify tokens (1 + 4
drafts) in that step. Eight streams therefore produce roughly 40 tokens per
2.1 s — about 19 tok/s, which is the "drops to 15 tok/s" that a busy session
shows. A `/compact` submits exactly this shape, and on two nodes
`max_model_len` is 262,144, so one compaction can queue a prefill much larger
than the one measured here.

### A full stall: cold PLE n-gram rows faulted one page at a time

Twice, a scenario that normally finishes in 7-10 s ran for 512 s and 315 s.
Both times the per-second series was mostly zeros with a single small burst
every few seconds — one decode step taking about 5 s instead of 77 ms. The
engine state during it:

- `num_requests_running` 4, `num_requests_waiting` 0, `num_preemptions_total` 0
- `kv_cache_usage_perc` frozen at 0.4917 across the whole window
- worker at 67% CPU in state R, major faults ~2,500/s against ~490/s in a healthy run
- `MemAvailable` about 5.2 GB and falling

That rules out KV exhaustion, preemption, admission queueing and a CUDA-graph
fallback — none of them move. What does move is host paging, and specifically
the PLE n-gram table.

The table is a 51.2 GB file mapped over ATS (`VLLM_PLE_MMAP_DIR`, with
`MADV_RANDOM` + `FADV_RANDOM` when `VLLM_PLE_MMAP_ADVICE=1`). The GPU
dereferences that mapping during the step, so a miss is not background work: the
step waits for the page. Readahead is deliberately off — the patch's own comment
says readahead comes out of the pool the GPU allocates from on GB10 — so each
miss is a single 4 KB read. Measured on the table file: 19.8 MB/s and 4,836 IOPS
at queue depth 1, 144.1 MB/s at 8 threads, 358.9 MB/s and 87,621 IOPS at 32. The
stalled runs read 30-44 MB/s, which is the single-page path, not the disk's
limit.

The same scenario, same boot, same content, four passes back to back — four
streams, 32,000-token prompt, 192 tokens each:

| pass | wall | aggregate | major faults | longest decoder blackout |
|---|---|---|---|---|
| 1st, cold | 949.4 s | 0.7 tok/s | 2,083,194 | 528 s |
| 2nd | 177.5 s | 6.5 tok/s | 427,385 | 89 s |
| 3rd | 6.8 s | 152.6 tok/s | 2,554 | 0 s |
| 4th | 6.0 s | 172.7 tok/s | 139 | 0 s |

2.08 M faults is about 8 GB of pages touched; by the fourth pass the same work
is 158× cheaper. Engine state never moved: `running` 4, `waiting` 0, preemptions
0, `kv_cache_usage_perc` flat. A prompt whose rows were already resident ran at
98.6 / 115.7 / 145.0 tok/s on the same boot.

So the stall is not intermittent and not a scheduling artefact: it is a function
of how cold the prompt's n-gram rows are. That also rescues the runs that were
fine with 21k major faults — 21k is two orders of magnitude below the cold case,
and those rows were already resident. `/compact` should be the reliable trigger
because it submits a large body of never-seen n-grams at once; any first pass
over new content does the same, which is what an agent does constantly. That
link rests on the content-locality measurement above rather than on an injected
compaction — the injected run was not completed.

This is the one-node PLE-offload lane. At two nodes `ple_offload` is off and the
table is not routed through the mmap, so the same dashboard signature there
still needs its own attribution — the prefill-budget arithmetic above is the
candidate, and that lane keeps `waiting` at 0 here where the dashboard shows 7.

Two fixes remain unmeasured against these numbers: `ple_staged_lookup` (host
gather through a pinned staging buffer, so the GPU never touches the mapping;
documented as failing three capture paths) and more page-cache headroom. The
latter is not available on one node — `gpu_memory_utilization` 0.72 is already
what it takes to fit weights plus pool, which is why the table is offloaded at
all; at two nodes the halved per-rank weights leave room for it. The measured
mitigation available today is content locality: a first pass over 32k tokens of
new text costs roughly 8 GB of page faults at 20-40 MB/s.


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

## Where the wall clock goes, from the engine's own counters

The harness times a task from outside; the engine's Prometheus counters say what
it spent. One full pass of the 18-task ruler through the proxy, with a snapshot
before and after (`bench/engine_trace.py snapshot` / `report`):

| stage | mean per request | share of end to end |
|---|---|---|
| queue | 0.0 ms | 0.0% |
| prefill | 520.7 ms | 21.1% |
| decode | 1,897.7 ms | 76.8% |
| TTFT | 575.0 ms | 23.3% |
| inter-token | 96.7 ms | 3.9% |
| end to end | 2,472.3 ms | 100% |

55 requests, 1,079 generated tokens, 48,726 prompt tokens. The window adds up:
55 x 2.47 s = 136 s against a harness wall time of 137.1 s with 0.2 s in tools,
and TTFT equals queue plus prefill (575 against 520.7) to within a sample.

Two things follow. Nothing waited — queue time is zero at one stream, so on this
path the wall clock is the engine, and decode is three quarters of it; acceptance
and step time are the levers, not scheduling. And the prompt side pays in prefill
at about a fifth of each request even when the calls are short, which is the
reason the per-turn marginal prefill above is worth attacking.

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

## Tool use, measured: twelve tasks, three arms

`bench/tool_tasks.py` plays the harness: it offers `list_dir`, `read_file`, `write_file`, `search` and `run_python`, executes what the model asks for in a scratch directory, feeds results back, and scores each task in code. Twelve tasks, temperature as served, turn bounds 4-7. Same set, three arms:

| arm | score | prompt tokens | failures |
|---|---|---|---|
| direct to the server | **10/12** | 34,892 | `edit_value` (wrong bytes), `large_output` (turn cap) |
| proxy, tool nudge on | **10/12** | 37,396 | `large_output`, **`count_matches`** (turn cap) |
| proxy, nudge aimed at tool choice | **10/12** | 37,408 | `edit_value`, `large_output` |
| proxy, tools pruned to two | **9/12** | 38,719 | `edit_value`, `large_output`, `count_matches` |
| proxy, policy skill pinned | **11/12** | 62,947 | `large_output` |

What each arm actually says:

- **The nudge does not help, and it costs.** Its premise is that the model answers without calling a tool; on this checkpoint it calls 30 tools in 37 turns, so there is nothing to nudge. The arm gained no task, lost `count_matches` to a turn cap, and added 2,504 prompt tokens. Rejected as a default.
- **The pinned policy skill buys one task for 28,055 prompt tokens.** It is charged per *turn*, not per task — 38 turns at 722 tokens each — which is the cache finding above in practice. One run, so the gain is not separable from run-to-run drift; the cost is large and certain. It stays opt-in.
- **`large_output` fails in every arm, and the trace shows why.** Three runs, same behaviour: `list_dir`, then `run_python` on `big.log` itself, then `run_python` on a script that does not exist yet, then `write_file` for that script, then the turn cap. It never calls `read_file`. The model does not avoid tools here; it picks the wrong one. That is a different problem from the one the nudge was written for, and it is the first measured target worth chasing.

`edit_value` fails because the write dropped the file's trailing newline while every other byte matched. That is a real defect, not a checker artefact, and it is recorded as such rather than relaxed.

The set has since grown to eighteen tasks: six of them isolate selection (`updated_file`, `pick_read_over_run`, `pick_search_over_read`, `pick_list_over_guess`, `no_tool_needed`, `write_then_verify`), and `--repeat N` reports a flaky task as flaky instead of as a pass or a fail. Baseline on the committed build: **13/18**, 60 turns, 56,710 prompt tokens, failing `edit_value` (wrong bytes), `large_output` and `pick_search_over_read` (turn cap on a large file), `count_matches` and `write_then_verify` (turn cap). `no_tool_needed` passes, so the model does not over-call on a question the conversation already answers.

Two more arms were run against the same hypothesis, both negative:

- **A nudge that names the narrowest tool** ("read a file with the file-reading tool instead of executing it, search instead of reading everything") changed nothing: same score, same failure, and the trace opens identically with `run_python` pointed at a path that is a directory.
- **Pruning the tool list to two choices** made it worse — 9/12, losing `count_matches` as well — and `large_output` still starts with `run_python`.

The conclusion the four arms support: this is not a prompting problem that a proxy can reach. The model calls tools readily and in volume; it picks the wrong one on a large file, and neither describing the right tool in the system message nor removing tools from the list changes that. The levers that remain are the tool *definitions* the client sends (names and descriptions, which the harness owns) and the model itself. Spending more proxy tokens on this is not supportable by the measurements.

## The turn contract is inert on short tasks

A short behavioural preamble about call batching, re-reading evidence, re-running passed checks and scripting deterministic transforms was measured on two scratch tasks, one run per arm, identical inputs, the only difference being whether the preamble was present in the project's `AGENTS.md`:

| bed | arm off | arm on |
|---|---|---|
| task states "write convert.py" | 7 turns, 6 calls, 30 s, pass | 6 turns, 6 calls, 26 s, pass |
| task states only the goal | 5 turns, 4 calls, 17 s, pass | 5 turns, 4 calls, 16 s, pass |

Nothing to gain: on a five-turn task the model already writes one script and never revisits a passed check, with or without being told. The first bed was also malformed as an experiment — its wording prescribed the single-script method, which is the behaviour the preamble asks for.

So the block is not refuted, it is unaddressed: the session that motivated it ran 95 turns with 45 of them reconnaissance or re-verification, and nothing at five turns can distinguish. The long-horizon bed has since run; it also shows no effect:

| arm | turns (3 runs) | reasoning |
|---|---|---|
| without the block | 21, 35, 21 | 5,904 / 16,497 / 9,371 |
| with the block | 23, 30, 22 | 11,489 / 31,532 / 9,503 |

The converter-and-resume task produces 21–35 turns either way, and the spread is larger than the difference. Verdict: the block does not reduce turns, and should not be shipped as a token saving. What the trace comparison did find is a narrower target — verification placement on diagnosis — recorded in `docs/model-comparison.md`.

## Prefix caching is quantised to 1,680 tokens

The engine sets its attention block size to **1,680** tokens on this configuration (`Setting attention block size to 1680` in the boot log), because one block has to hold one GDN page and the page geometry follows the draft depth (`scripts/mtp_block.py`). Prefix caching works in whole blocks, so the hit is always an integer multiple of 1,680:

| prompt | cached | floor(prompt / 1,680) × 1,680 |
|---|---|---|
| 6,072 | 5,040 | 5,040 |
| 6,794 | 6,720 | 6,720 |
| 792 | **0** | 0 |

The first two rows are the measurements from the cache section above and they match the arithmetic exactly. The third is the one that matters: **a prompt shorter than one block is never cached at all**, even when the same 792-token system message is sent three times in a row.

How much that costs depends entirely on the prompt size, because the loss is at most the last partial block:

| prompt tokens | cached | share |
|---|---|---|
| 1,024 | 0 | 0% |
| 4,096 | 3,360 | 82% |
| 16,384 | 15,120 | 92% |
| 32,768 | 31,920 | 97% |
| 65,536 | 65,520 | 99.98% |

So block quantisation is a **short-prompt problem only**. At the context sizes a real agent session reaches, the cache absorbs everything except the last fraction of a block, and the earlier five-turn measurement (80% at 15k tokens) sits exactly where this table says it should. Any statement about "the cache does not work" is wrong above about 4k tokens.

Three consequences:

- **Short requests pay full price every time.** A system prompt, a skill block or a policy preamble only becomes free once the conversation is longer than one block. Below that it is a straight cost, repeated per request.
- **The injected policy skill costs 722 tokens per request on short traffic.** A batch of twelve unrelated requests through the proxy with the skill pinned moved prompt tokens from 77,522 to 86,186 — exactly 722 each — with **zero** of it cached, because 722 is less than one block. In a growing conversation the same block lands inside the first 1,680 tokens and is cached from the second turn on, which is why the five-turn session above reaches 80%.
- **The block size cannot be lowered.** The engine only ever raises a requested block size, and the derived value is forced by the MTP draft depth plus the GDN page geometry. A smaller draft depth (3) would give 1,664 instead of 1,680 — a rounding difference, not a lever.

So the rule for anything the proxy injects is: it is free only if it stays byte-identical *and* the request is already past a block boundary. Otherwise it is 722 tokens of pure overhead, and the decision to pin a skill has to be made on that basis rather than on the assumption that the cache absorbs it.

## The thinking budget is native, and it does not bind here

This build accepts `thinking_token_budget` on the chat request and enforces it in the sampler (`v1/worker/gpu/sample/thinking_budget.py`), so a trace cap needs no proxy code and no patch. Measured against the baseline on the same twelve tasks:

| arm | score | completion tokens | reasoning tokens |
|---|---|---|---|
| no budget | 12/12 | 2,407 | 1,659 |
| 512 | 12/12 | 2,333 | 1,665 |
| 1024 | 12/12 | 2,191 | 1,572 |

The arms are indistinguishable because nothing reaches the cap: the mean trace is **138 tokens** and the longest is **241** (`negation`), so a 512 budget is never hit. That is the prior baked into the checkpoint doing its job.

**This measurement covers one-shot tasks only.** Every task here asks a self-contained question whose answer fits in a sentence, which is not what an agent turn looks like. A planning or debugging turn over a large tool result can plausibly run a trace many times longer, and this set cannot show it. The tool-use harness reports `reasoning_tokens` per task for exactly that reason; until an agent-shaped trace is measured above the budget, the trace-budget question stays open rather than settled in either direction.

The ceiling is small anyway. Generated tokens are **3% of the tokens a session spends** — 2,407 against 77,522 prompt tokens in this run, and 320 against 47,168 across the five-turn session above. Removing the trace entirely would move the total by about 3%, while the marginal prefill of one tool result moves it by 2,000.

Verdict: no default to set, and the planned trace-budget work is dropped rather than shipped unmeasured. Reopen only if a real agent trace is observed above the budget, which would show up directly in the accounting line.

That condition has now been met on agent-shaped traces — see the next section. Note the two knobs are different things: `thinking_token_budget` is a sampler cap, while the template also reads `reasoning_effort` (xhigh by default, medium, low) as an instruction.

## Reasoning effort does move an agent trace

The same 95 prompt prefixes from a recorded agent session, replayed against the template with two effort settings. Each request is a bounded window of the recorded history (the client's compaction state is not in the transcript, so this is a reproduction, not a byte-for-byte replay); the only variable is `reasoning_effort`.

| arm | completion | reasoning | wall (95 turns) |
|---|---|---|---|
| `low` | 65,067 | 37,981 | 2,036 s |
| `xhigh` (template default) | 73,149 | 48,594 | 2,261 s |
| difference | +12.4% | +27.9% | +11.1% |

- **The trace shrinks by 28%** on the same prompts, and total output by 12%. On the recorded session, reasoning was 76% of all generated tokens (85,610 of 111,949), so this is the first knob that moves the dominant term rather than a rounding error.
- **Behaviour is unchanged at the resolution measured.** Both arms chose the same tool at all 95 turns. That is a weak quality signal — it compares which tool, not whether the task succeeded — so it accepts the lever, it does not certify it.
- **The seconds column is single-box and serial**, with the request stream waiting on each reply. It is not comparable to the decode table above, and it does not transfer to the pair.

Verdict on tokens: a 28% trace reduction for no measured change in tool choice looked worth carrying. **The agent ruler then contradicted it**, and the correction is below rather than in a footnote — the replay measured mid-session continuations, not a tool loop.

## What the effort setting actually buys, on the agent ruler

Same boot, same proxy, 18 tasks, five passes per arm. The three arms turn out to be three different *system prompts*, because the chat template only writes an instruction for two of the three values:

| effort | system block the template emits |
|---|---|
| unset (or `xhigh`) | "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives…" |
| `medium` | **none** — the rendered prompt is the user turn only |
| `low` | "Reasoning effort is set to low. Keep your thinking brief and focused…" |

| arm | scores (5 passes) | mean | turns | reasoning |
|---|---|---|---|---|
| unset (xhigh text) | 15, 15, 14, 13, 13 | 14.0 | 57–60 | 1,031–1,725 |
| `medium` (no text) | 17, 16, 15, 15, 15 | **15.6** | 54–57 | 1,236–1,507 |
| `low` (brief text) | 13, 13 | 13.0 | 60–62 | 1,682–2,246 |

- **`medium` won every paired pass**, +1 to +2 tasks, and it used the fewest turns. The xhigh instruction tells the model to validate assumptions and weigh alternatives; these tasks fail on turn caps, so that instruction spends exactly the turns the cap withholds.
- **`low` is the worst arm on every measure**: fewer tasks than the default, and *more* reasoning than either other arm (1,682–2,246 against 1,031–1,725). "Move directly to the conclusion" produces answers that need repair, and the repair costs more than the brevity saved.
- The 28% saving measured on replayed prompts did **not** transfer: with tools in the loop the brief instruction lengthens traces. A prompt-level effect measured without a tool loop is not a result about agents.

Verdict: the instruction text is a cost, not a control. `medium` is at least as
good as the unset default and cheaper in turns, and any default must be set by
the proxy (`PROXY_REASONING_EFFORT=medium`) because neither the client nor the
server can express it — but it is not promoted to a default. A `--repeat 2` run
of the same arm scored **29 of 36 task runs, 14.5 of 18 per run**, against the
unset default's 14.0 mean, and **five of eighteen tasks flipped between the two
repeats** (29 = 2x12 resolved plus 5 half-passing). The 1.6-task margin from
single passes sits inside the ruler's own repeat variance, so the defensible
claim is "no worse, fewer turns", not "better by a measured amount".

Open, and worth saying plainly: the earlier twelve-task result used a cap that never bound (mean trace 138 tokens). The recorded agent session has a **mean trace of 901 tokens**, so a 512 `thinking_token_budget` would bind on roughly half its turns. The sampler cap and the template instruction have not been compared against each other on agent traces.

## A changed re-read costs a diff, not the file

A coding harness does not re-send a file after an edit; it sends a diff. The proxy can do the same, because it knows the earlier result of the same call. `PROXY_DIFF_TOOL_OUTPUT=1` sends a unified diff against the first result of that call when the content changed, and only when the diff is under half the payload.

Measured on a ~30k-token file read, then read again, three arms:

| case | off | dedupe only | diff |
|---|---|---|---|
| second read unchanged | 60,161 | **30,191** | **30,191** |
| one line edited | 60,165 | 60,165 | **30,375** |
| whole file rewritten | 61,361 | 61,361 | 61,361 |

- **A one-line edit costs 49.5% less.** Plain duplicate collapse saves nothing here, because the bytes differ; the diff saves the file.
- **The diff itself is 184 tokens** beyond the collapsed case (30,375 against 30,191) — the patch for one line with two lines of context.
- **A rewrite falls back to the full payload**, by design: above half the payload a diff stops being a saving and starts being a worse description of the file.
- The reference is always the *first* result of that call, which keeps its full content, so the header always points at a message the model can still read.

Off by default, because it changes what the model reads. The `updated_file` task scores exactly that: write, re-read, answer from the new content — and it fails a `stale_read` where the model answers from a marker or a collapsed copy.

## Duplicate tool output, and per-request accounting

A tool result that repeats verbatim is the one part of an agent transcript that is pure waste: it is paid for in every later turn's prompt. The rule that decides what counts as a repeat matters more than the saving, because two cases look identical byte for byte and must be treated differently.

A result is collapsed only when **the call itself repeats** — same tool, same arguments — *and* the payload is byte for byte the same. Measured through the proxy on three conversations that each carry two ~30k-token payloads:

| case | gate off | `PROXY_DEDUPE_TOOL_OUTPUT=1` | collapsed |
|---|---|---|---|
| same call, file unchanged | 60,161 | **30,191** | yes |
| same call, file updated between reads | 60,165 | 60,165 | **no** |
| different path, equal bytes | 60,161 | 60,161 | **no** |

- **An updated file is always delivered in full.** Different bytes are a different payload, so the model never sees a marker standing in for content it has not read. This is the case that would be dangerous if the rule keyed on the path alone.
- **Two files that happen to hold the same bytes are both delivered.** They are different calls; the model asked for the second path and is not told it already has it. An earlier version of this rule keyed on bytes only and did collapse that case, which is why the measured saving used to read as 49.8% on a conversation that should not have collapsed at all.
- Only `role: "tool"` results above 512 bytes participate, the first occurrence is never touched, and a result whose originating call cannot be resolved is left alone because sameness cannot be proved.

It stays opt-in. That is still a meaning change, and this benchmark set has no repeated tool results, so nothing here answers the quality question.

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
