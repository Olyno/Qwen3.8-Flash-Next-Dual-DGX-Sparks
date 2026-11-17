# FastMTP: not pursued on current equipment

FastMTP = retraining the model's MTP (multi-token prediction) draft head so
it drafts better: higher acceptance rate per speculative step, i.e. more
tokens decoded per target-model forward pass. On paper it's one of the
biggest single levers on decode speed (acceptance rate is the multiplier in
tokens/step).

## Why it's parked (decided 2026-10)

1. **No training-stack support for this architecture.** FastMTP training
   goes through NVIDIA's Speculators library, which has no `qwen4_exp`
   (Qwen3.8-Flash-Next hybrid GDN) support. We'd have to write and validate
   that model support ourselves first.
2. **GPU time we don't have.** Rough estimate from the training recipe:
   ~16 GPU-hours per 5k samples, and a useful corpus is several × 5k.
   The only GPU is the single DGX Spark that also serves the model —
   training means taking the server down for days.
3. **Unified-memory ceiling.** 121 GiB shared between system and GPU is
   tight for training with optimizer states, even for a draft head.

## Revisit when

- Speculators ships `qwen4_exp` support (or someone publishes a FastMTP
  head for this checkpoint we can just load), **and**
- a spare GPU window exists (second Spark, cloud burst, or a scheduled
  multi-day serving outage).

Until then the acceptance-rate levers we *can* pull without training are:
draft-vocab restriction (shipped), draft-k tuning (shipped), and adaptive MTP
draft depth (`mtp_adaptive_depth`, shipped opt-in but INERT 2026-10-10: the
v0.30 lane resolves async scheduling ON, and AsyncScheduler schedules a fixed
k from config without ever reading the worker's proposed width — the cutoff
computes correctly but the truncation is a no-op; see the patcher docstring).
On the v0.32.0 image the overlay's target path is gone (the V2 autoregressive
speculator moved); the flag is DEFERRED pending a decision between a re-port
and the native dynamic-K work (vllm#57053/#60945, `enable_adaptive_verification`
+ `adaptive_verification.py`).
Dead ends, measured:
adaptive verification / MRV2 (in the v0.30 image; boot-probed 2026-10-10 —
conflicts with `use_local_argmax_reduction` which `mtp_draft_vocab` requires,
and `GDNAttentionBackend` is an SSM backend that rejects on-device query
trimming; needs real engine surgery, not worth it) and prefix-cache-under-MTP
(vllm#52244 — inert without `--prefix-match-unit`, which we don't set).

## Async scheduling configuration

`async_scheduling` is a recipe boolean, not an auto/default selector:
`true` emits `--async-scheduling`; `false` (also the default when unset)
emits `--no-async-scheduling`. The existing boolean aliases (`yes`/`no`,
`1`/`0`) still normalize to `true`/`false`. `auto` is not accepted.
Both flags are supported by vLLM v0.32.0 and the audited day-0
`qwen38-flash-next` image used by `mia`; no legacy flag omission is needed.

Previously `false` omitted the flag, letting vLLM auto-enable async for
MTP with the `mp` executor. `prod.yaml` now explicitly sets `true` to preserve
that intended ON behavior. Other recipes or `.env` settings that used
`false` to mean auto must use `true` if they want async enabled; otherwise
they now really disable it. A recipe value still wins over `.env`.
`EXTRA_VLLM_ARGS` remains last, so an explicit scheduling flag there wins
over the recipe's generated flag.

This corrects configuration semantics only: it does not re-port adaptive
draft depth or establish a runtime quality/performance benefit for either
scheduler.
