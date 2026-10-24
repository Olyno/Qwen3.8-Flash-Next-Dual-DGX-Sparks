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
