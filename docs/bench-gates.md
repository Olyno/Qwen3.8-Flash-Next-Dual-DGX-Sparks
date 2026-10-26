# Benchmark gates — no measured configuration gets slower

Standing convention for every serving-config change (patch, flag, recipe knob)
that lands in this repo. The gate is directional, not aspirational: a config
may be *not measurably slower* and land; a measurable regression on any
measured shape does not land without a written trade-off note in the commit.

Ideas credited (methodology only, no code copied): Strata (MIT), XeStrata
(LGPL, idea-only), flashrt (Apache-2.0), ds4-dfm-rs (MIT).

## The matrix

`bench/gate_report.sh` runs the suite and prints one row per gate:

| gate              | metric                                   | threshold                       |
|-------------------|------------------------------------------|---------------------------------|
| prefill, each bucket | TTFT-normalized ms per 1k prompt tokens | regression ≤ **1%**            |
| decode, each bucket  | ms per engine step                      | regression ≤ **5%**            |
| kl-quality           | p99 KL vs self-noise floor, top-1 agreement | p99 ≤ 2× floor; top-1 ≥ 99.5% |

Context buckets are **1k / 32k / 65k** prompt tokens — one row per bucket, not
an average; a config that is 1% faster at 1k and 12% slower at 65k is a
regression. `--record` stores the current numbers as the comparison baseline
(default `bench/data/gate_perf.json`, keyed by endpoint URL + bucket); every
later run gates against it. Exit 1 on any FAIL; a SKIP is never a pass — an
offline host yields an all-SKIP matrix and exit 2, and the run does not count.

## Measured arms

Each opt-in path is measured alone against the committed default (adaptive
verify width on) with the same matrix; a failing row keeps the key off, and the
recipe comment next to the key carries the boot details. Baseline at the
configuration the arms were measured on: prefill `0.758 / 0.363 / 0.351` ms per
1k tokens and decode `76.28 / 76.83 / 77.35` ms/step at 1k / 32k / 65k. A clean
re-record on the committed budget measured `0.743 / 0.383 / 0.352` and
`80.34 / 78.47 / 75.79`; the 1k decode bucket drifts by several percent between
boots, so the arm verdicts lean on the within-run A/B and on every bucket
moving the same way, not on that one row.

| key | decode Δ ms/step (1k / 32k / 64k) | prefill Δ (1k / 32k / 64k) | verdict |
|-----|-----------------------------------|----------------------------|---------|
| `verify_width_adaptive` | −3.1 to −4.1% in every bucket | neutral at 32k/64k | ON |
| `fp8_draft_head` | −9.3..−10.1 / −4.9..−9.3 / −3.1..−5.7% (vs the 0.72 record) | TTFT rows −30.8 / −7.5 / +0.4%, not attributable | ON (KV 223K, 3.40×) |
| `mtp_skip_prefix` | +6.9 to +9.1% in every bucket | neutral/positive, noisy | OFF (parity also failed) |
| `copy_proposer` | +0.3 to +4.1% in every bucket | +2.3% at 32k | OFF |
| `gdn_direct_out` | +13.3 / +6.4 / +3.2% | +3.4 / +2.1 / −0.7% | OFF |
| `ple_staged_lookup` | not gated | not gated | OFF (three boot failures) |
| `spec_trace` | 0% (diagnostic) | 0% | OFF |

`ple_staged_lookup` never reached a measurement: the staged forward collapsed
the profiled KV memory, so the matrix does not exist and a SKIP is not a pass.

## Each arm must prove which proxy it hit

An arm that cannot name its configuration is not an arm. A stale proxy still
holding the port, or a restart that kept the previous environment, produces two
identical arms and a null delta that reads like evidence — this happened once
already, when a guard arm's log was never written and its setting could not be
confirmed after the fact.

`scripts/ab_proxy_arm.sh` makes that loud. It writes the requested environment
to `/tmp/ab/<arm>.log` **before** starting anything, starts the proxy with
exactly that environment, then reads the environment of the process that owns
the port (`/proc/<pid>/environ`, resolved via `ss`) and fails with exit 2 if any
requested `KEY=VALUE` is absent or different. `--assert-only` performs the check
against a running proxy without restarting it, which is the cheap way to
validate an arm before a long run:

```
./scripts/ab_proxy_arm.sh guard PROXY_EDIT_GUARD=1 PROXY_REASONING_EFFORT=medium \
    -- python3 bench/tool_tasks.py --repeat 2
./scripts/ab_proxy_arm.sh guard PROXY_EDIT_GUARD=1 --port 8899 --assert-only
```

The negative paths are covered by `tests/test_ab_proxy_arm.py`: an unset key and
a wrong value must both exit 2, and the log must exist with its header before
the assertion runs.

## Report ms/step, not tok/s (A/B under speculation)

Under MTP/spec decode, `tok/s = steps/s × tok/step`, and `tok/step` moves with
the *text*, not the configuration. Two arms generating different words are
comparing texts, not setups. The delta that attributes to the configuration is
**decode-window / engine-steps = ms/step** (steps = `spec_decode_num_drafts`
under speculation, one step per generation token on a draft-free arm — both
read from `/metrics` deltas around the run, `bench/mtp_accept.py` counters).
Any A/B comparison under speculation — `bench/ab_pair.py`, `bench/sweep.py`,
`bench/gate_report.sh` — reports ms/step; tok/s may be shown next to it as
context, never as the verdict.

For the comparison to mean anything both arms must process identical text:
use `bench/ab_pair.py` (teacher-forced chunk replay of captured completions)
rather than free-running generations.

## Re-run the baseline in the hour

A delta under **2%** on this stack is inside baseline noise: prefix-cache
warmth, thermals and queue order move ms/step that much between passes. Rule:
if |Δ ms/step| < 2%, re-run the baseline arm within the hour before quoting
the delta; if the baseline's own re-run drift is ≥ the claimed delta, the
result is inconclusive, not a win. `bench/ab_pair.py` enforces the shape of
this automatically (immediate baseline re-run + self-drift printout).

## Every path gets a kill-switch

No patch, overlay, or serving flag lands without its own off-switch: one env
var / recipe boolean that disables exactly that path with no rebuild, verified
by launching once with it off. A gate failure mid-session must be reversible
by one flag, not by a re-deploy — same rule for chained patches (switches
compose independently, or the interaction is documented as one unit).

## Memory accounting is in bytes

Memory regressions are read from byte counters — device allocation reported by
the server, `/proc/meminfo` MemAvailable minima (`bench/sweep.py` sampler),
KV pool bytes — never from derived percentages, rounded GiB figures, or
"looks fine". A change that trades speed for memory states both numbers in
bytes at the shapes they were measured. Positive bytes delta = it goes in the
commit message like the qkvg-merge precedent, and the gate doc row it moves.

## Quality gate: the p99-KL yardstick

`bench/kl_yardstick.py` compares baseline vs candidate on five byte-frozen
texts (code / English prose / German / agent transcript / math, pinned in
`bench/data/kl_corpus.jsonl` with the position spine in
`bench/data/kl_corpus_positions.json`) via teacher-forced
`prompt_logprobs=64` on `/v1/completions` — both arms score identical token
positions, nothing is sampled. It always runs the baseline twice first and
prints the self-noise floor before any verdict: **a delta inside your own
floor is not a difference.** Pass requires p99 KL ≤ 2× the floor's p99 KL and
top-1 agreement ≥ 99.5% (flags `--mult`, `--min-top1`). Once `freeze` has
pinned the tokenized position list from a healthy endpoint, every later run
refuses to score if the live tokenization moved. Generation gates
(`output_gate.py`, `quality_gate.py`) stay as breakage detectors; the yardstick
is the measure for subtle quality calls they cannot resolve.

A corpus item may also carry `"scored_token_spans": [[start, end], ...]` —
inclusive-start, exclusive-end token indices that restrict which positions are
pooled into KL, top-1 and perplexity (the full text is still scored). The spans
are token indices pinned by whoever builds the corpus, since they follow that
item's tokenizer, and items without the field score every position.
`--decision-ids` adds a diagnostic row for stop-decision token ids without
moving the verdict, while `--batch N` / `--shape-floor N` fold the server's
batch-composition noise into the floor: the verdict uses the larger of the two
p99 floors, and both are printed.

## Running it

```
bench/gate_report.sh --record                        # after a verified-good boot
bench/gate_report.sh                                 # every subsequent config
bench/gate_report.sh --candidate <cand-endpoint>     # + the KL gate
python3 bench/ab_pair.py --corpus <captured.jsonl> --arm-a <base> --arm-b <cand>
python3 bench/kl_yardstick.py compare --url <base> --candidate <cand>
```

Server must be otherwise idle (foreign traffic lands in every counter and
window). `bench/data/gate_perf.json` is recorded per host, not committed.
