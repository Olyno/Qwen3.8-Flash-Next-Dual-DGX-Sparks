# Score proxy: decision middleware in front of vLLM

`proxy/score_proxy.py` is a thin OpenAI-compatible reverse proxy that sits
between any client/harness and the vLLM server. Everything it doesn't
recognize is forwarded byte-for-byte (SSE chat streams relay chunk-by-chunk,
never buffered); on top of that it adds two decision features. Stdlib Python
only, no dependencies.

## Run it

```
PROXY_UPSTREAM=http://localhost:8888 python3 proxy/score_proxy.py
```

Listens on `PROXY_PORT` (default 8889). Point clients at the proxy instead of
vLLM — same API.

## `POST /v1/score` — score, don't generate

Scores N options in one batched prefill instead of generating and parsing;
the same echo-logprobs pattern as `bench/option_scoring.py` (measured
context in [decision-scoring.md](decision-scoring.md)). Options may carry a
`use_when` description that is inlined into the scoring prompt, and by
default the option list is scored forward and reversed and the two
distributions averaged (position-bias mitigation).

```
curl localhost:8889/v1/score -d '{
  "model": "Qwen3.8-Flash-Next-NVFP4",
  "question": "Route this ticket: my invoice shows a charge after I cancelled.",
  "options": [
    {"label": "refund_request", "use_when": "customer wants money back"},
    {"label": "cancel_subscription", "use_when": "customer wants to end service"},
    "bug_report"
  ]
}'
```

→ `{"ranked": [{"option": "refund_request", "prob": 0.81}, ...],
"orders_averaged": true}`. Set `"average_orders": false` to skip the second
pass.

## Confidence-gated thinking escalation

Inspired by [autotrust/GEV-26B-Decide](https://huggingface.co/autotrust/GEV-26B-Decide):
answer without thinking first, and only pay for thinking when the model is
unsure (their reference result: 83.4% vs 83.8% accuracy at 48% of the
thinking tokens).

On non-streaming `POST /v1/chat/completions` with `"escalate": true` in the
request (or `PROXY_ESCALATE=1` for all traffic), the proxy:

1. Re-issues the request with `chat_template_kwargs.enable_thinking: false`
   and `logprobs: true`.
2. Computes mean token logprob over the assistant content tokens.
3. If it's at or above `PROXY_ESCALATE_THRESHOLD` (default **-0.35**),
   returns that answer with `"x_proxy": {"escalated": false, "mean_logprob": m}`.
4. Otherwise re-issues the original request unchanged (thinking on) and
   returns it with `"x_proxy": {"escalated": true, ...}`.

**Calibrate the threshold per workload before trusting the gate** — -0.35 is
a heuristic default, not a measured one. Log `x_proxy.mean_logprob` on real
traffic and pick the cut that trades accuracy vs thinking tokens for your
case.

**Streaming caveat:** `stream: true` chat requests always pass through
untouched — escalation only applies to non-streaming requests (confidence
needs the full response).

## Credits

- [autotrust/GEV-26B-Decide](https://huggingface.co/autotrust/GEV-26B-Decide)
  — confidence-gated thinking escalation.
- [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) —
  prompt rules: single multi-option question, use-when descriptions,
  order averaging.
- [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) and
  [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
  — the score-don't-generate pattern; see
  [decision-scoring.md](decision-scoring.md).
