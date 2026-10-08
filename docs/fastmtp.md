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
draft-vocab restriction (shipped), draft-k tuning (shipped), adaptive
verification / MRV2 (in the v0.30 image, needs a boot test), and
prefix-cache-under-MTP (vllm#52244, staged).
