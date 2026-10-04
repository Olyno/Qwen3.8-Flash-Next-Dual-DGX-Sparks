# Decision scoring: don't generate what you can score

For structured-decision requests — classify, route, pick one of N options,
yes/no gates, moderation — autoregressive generation is the wrong tool. The
answer is one of a handful of known strings, so a single prefill pass that
*scores* each option beats decoding tokens and parsing them back out.

Inspiration: [Cloudflare/clef](https://huggingface.co/Cloudflare/clef)
(joint schema head on a Qwen3.8 backbone, one pass scores every option) and
[fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
(340M specialist classifier, label sets at call time). Neither is something
to adopt; the pattern is.

## The pattern

One batched `/v1/completions` call: N prompts, each = shared prefix + one
option, `echo: true, logprobs: 0, max_tokens: 1`. Sum the option-token
logprobs, softmax, done. Prefix caching (on by default in prod) makes the
shared prefix cost one prefill total, and the N option tails batch into the
same step.

Measured on a DGX Spark (prefill ~2,600 tok/s): a 1K-token prompt with 8 short
options scores in **~0.5–1 s**. Generating the same answer costs the same
prefill plus 10–50 decode steps at ~22 tok/s — **1.5–3 s**, plus a parse
failure mode that scoring doesn't have. The gap widens with concurrency:
scoring is pure prefill (batches perfectly), generation is decode-bound.

Use generation when the output is open-ended. Use scoring whenever the
output is one of a known set.

## Try it

```
python3 bench/option_scoring.py --host localhost:8888
```

Demo: routes a support message into 8 intents and prints probabilities.
Flags for your own question/options; see `--help`.

## When the volume justifies it: a specialist router

Clef/GLiNER show a purpose head beats general models on routing and
classification at a fraction of the latency (GLiNER2.5-Decide: 340M, runs
on CPU). If a product has high-volume routing/moderation/triage traffic,
the system-level win is a small specialist model in front of the Spark that
only forwards real generation to the big model. That's a product decision,
not an engine one — but the scoring pattern above is the zero-cost version
of the same idea on the server we already run.
