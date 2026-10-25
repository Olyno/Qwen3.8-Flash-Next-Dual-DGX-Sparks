# Score proxy: decision middleware in front of vLLM

`proxy/score_proxy.py` is a thin OpenAI-compatible reverse proxy that sits
between any client/harness and the vLLM server. Everything it doesn't
recognize is forwarded byte-for-byte (SSE chat streams relay chunk-by-chunk,
never buffered); on top of that it adds decision features and a
non-streaming tool-call/answer repair pass. Stdlib Python only, no
dependencies.

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

**Streaming caveat:** escalation only applies to non-streaming requests
(confidence needs the full response). `stream: true` chat requests still relay
chunk-by-chunk with no buffering, but a loop guard watches the delta text for
degenerate repetition — the same normalized 20+ char phrase repeated
`PROXY_LOOP_GUARD_REPEAT` times (default 3) at the generation frontier, or a
2-char unit run ('abab…'). On a hit the upstream connection is closed (frees
the vLLM seat immediately) and the client gets a well-formed final SSE chunk
with `finish_reason: "stop"` plus `"x_proxy": {"loop_guard": "cut"}`, then
`data: [DONE]`. Set `PROXY_LOOP_GUARD=0` to disable.

## Tool-use steering (`PROXY_TOOL_ROUTER=1`)

Agentic nudge: on chat requests that carry a `tools` array, the proxy
scores whether the last user turn needs a tool (`use_tool` vs `answer`, both
option orders averaged — the same two-option prefill as `/v1/score`, ~0.5–1 s
at 1K tokens). Above `PROXY_TOOL_THRESHOLD` (default **0.5**) it either
appends a fixed nudge ("call the tool before answering when one
applies") as a trailing user message, or, with `PROXY_TOOL_FORCE=1`, sets
`tool_choice: "required"`. The nudge never touches the conversation head, so
it cannot invalidate a cached prompt prefix.
A `tool_choice` the client already sent is never overridden. Without tools,
below threshold, or on any scoring failure the request is forwarded
unchanged. Every firing logs `tool steer prob=... force=...` — calibrate the
threshold from those lines against a labeled sample before trusting it.

With an embedding endpoint configured (`PROXY_ROUTER_URL`), the same router
also prunes oversized tool lists: when a request carries more than
`PROXY_TOOL_TOP_K` tools (default **8**), the tools are ranked by cosine
against the last user message and only the top-k survive (the client's tool
order is preserved). `PROXY_TOOL_ALWAYS` names — comma-separated — are never
pruned. Weak tool selection is often a choice-among-many problem, so a
shorter, sharper list raises call rate and choice quality even when the
model never sees the pruned tools. Any embedding failure keeps the full list.

## Tool-call and truncated-answer repair

Non-streaming chat replies are inspected before they are returned and
repaired when the model emitted a call the harness cannot parse, or ran out
of budget in the middle of its thinking. A reply that already parses and has
an answer is relayed as the raw upstream bytes — the repair paths never
rewrite well-formed traffic. SSE streams still pass through untouched (a
buffered repair would defeat streaming), so call/answer repair applies to
`stream: false` only; the pre-request max_tokens clamp covers both.

The call/answer repair is always on and stdlib-only; the context clamp is
opt-in because the proxy cannot infer the model's context size. The
mechanisms:

- **Tolerant call parse.** Accepts a JSON object in a Markdown fence,
  trailing commas, single-quoted keys, a `tool_call` / `function_call` /
  `invoke` wrapper, `arguments` as a JSON string or an object, the Qwen
  `<tool_call><function=NAME><parameter=K>V</parameter>…` form, an
  Anthropic-style `<invoke>`, and a call left inside `reasoning_content`
  when no answer reached `content`. A truncated tail is closed (open
  quote/bracket/brace) and a dangling key is dropped before parsing. Only
  declared tool names become `tool_calls`; the entry is a normal OpenAI
  function call with a deterministic id.
- **Schema-typed coercion.** When the request declares `tools`, argument
  values are coerced to the declared type: `"5"` → `5`, `"true"` → `true`,
  numeric strings with whitespace or a unit suffix, a bare value into an
  `array` (whose items are then coerced), and a JSON string into a declared
  `object`. Undeclared keys and absent arguments are never touched or
  invented.
- **Unclosed wrapper/tail.** A reply that ends inside a JSON object or a
  tool-call wrapper is closed and emitted as a call instead of a truncated
  text turn; `finish_reason` becomes `tool_calls`.
- **Pre-send fit check (`max_tokens`).** With `PROXY_CONTEXT_TOKENS` set to
  the model's context size, the prompt is estimated (4 chars/token) and an
  explicit `max_tokens` larger than the room left after it is lowered to fit
  — a clamped request still produces an answer, while a rejected one wastes
  the whole turn and gets retried. When the prompt alone already exceeds the
  context, the request is forwarded untouched: conversation content is never
  truncated by the proxy, and the server's own overflow answer is the honest
  one. The close-thinking continuation is clamped against the same budget
  minus what was already generated.
- **Close-thinking retry.** A non-streaming reply with `finish_reason`
  `length` or `stop` that stops inside `<think>`/`<reasoning>` (or with
  `reasoning_content` and no answer) is continued **once** with the thinking
  closed via `continue_final_message`. The continuation is bounded by
  `THINK_RETRY_MAX` (1024) tokens. A second retry is never issued.
- **Answer-room reservation.** The continuation budget is the reply reserve
  — a quarter of the request cap, at least 512 tokens, at most half of it,
  and bounded to 1024 for the single continuation. Because the thinking is
  already closed, all of that budget is answer room; a cap below 1024 cannot
  fit both parts, so the whole cap goes to the answer rather than leaving
  it empty. A non-streaming proxy cannot intervene inside one generation,
  so the reserve is applied to the continuation, where the thinking is cut
  and the answer gets the reserved budget.

Repairs add `"x_proxy": {"thinking_retry": "closed"}` to the response; the
tool-call parse does not add a marker. The continuation is fail-open: if it
errors or returns no answer, the original response is relayed unchanged.
Streaming passthrough and escalation gating are unchanged.

## Token accounting

Every completed upstream chat completion logs one line to stderr:

```
score proxy: usage prompt_tokens=18422 cached_tokens=17920 completion_tokens=412 reasoning_tokens=128
```

`cached_tokens` comes from `usage.prompt_tokens_details.cached_tokens` and
`reasoning_tokens` from `usage.completion_tokens_details.reasoning_tokens`
(flat `usage.cached_tokens` / `usage.reasoning_tokens` are accepted too).
Only counters the server reported are printed; a response without a usage
object logs `score proxy: usage unavailable`. Streams are read out of band —
the SSE bytes relayed to the client are untouched — and the usage chunk is
picked up when the upstream sends one, so keep `stream_options:
{"include_usage": true}` on the client side if you want stream accounting.
This line is what attributes a prompt change (dedupe, layout, skills) to a
change in prompt or cache-hit tokens on the next turn. Nothing in the
request or response changes.

## Duplicate tool-output collapse (`PROXY_DEDUPE_TOOL_OUTPUT=1`)

Agent sessions re-send old tool output on every turn, and a harness that
retries a step can leave two byte-identical results in one transcript. With
`PROXY_DEDUPE_TOOL_OUTPUT=1`, a repeated `role: "tool"` message is replaced
before generation by:

```
[identical to the tool result above at message 7; 412 bytes]
```

The marker names the first occurrence's position in the message list
(1-based, as the transcript reads) and the payload's size in bytes, so the
model can still reason about how much output a step produced and know why it
is not seeing it again. Only exact byte-identical string content
participates; the first occurrence, every non-tool message, and anything
shorter than 512 bytes are left byte-identical. The transform is idempotent
(a marker cannot match a real payload), applies to streaming and
non-streaming alike, and runs before the fit check, so the collapsed prompt
also gets the freed room back. It changes what the model reads, which is why
it is off unless an operator turns it on. Every firing logs
`collapsed N duplicate tool result(s)`.

## Middleware triage (2026-10-10)

Research verdicts on the originally planned stage-2 items:

- **PASTE / Speculative Actions — skipped.** arXiv 2603.18897 ("Act While
  Thinking") and arXiv 2510.04371 ("Speculative Actions") both accelerate
  *agent tool execution* by speculatively running predicted tool calls in the
  client harness while the model generates. Tool execution does not exist in
  the serving path — this proxy sees only chat messages, and giving it
  filesystem/shell side effects is a security hazard the papers wall off
  behind isolation machinery. The proxy-feasible kernel (pre-warming the APC
  prefix for the likely follow-up turn) is already what `CacheKeepalive`
  does. Nothing left to build here.
- **Embedding router — adopted** (see below). A cheap embedding model
  classifies the last user message; `trivial` requests get thinking disabled
  and a capped `max_tokens` *before* generation (pre-hoc), complementing the
  post-hoc escalation gate. Target model: `google/embeddinggemma-2` (CPU,
  ~191 MB quantized), with `google/embeddinggemma-300m` as fallback — the
  proxy only needs an OpenAI-compatible `/v1/embeddings` endpoint.

## Embedding router service

`proxy/embedding_server.py` serves the OpenAI-compatible `/v1/embeddings`
endpoint the router POSTs to, on CPU, wrapping a sentence-transformers
model (stdlib HTTP otherwise). On the deployment host:

```
pip install --user sentence-transformers
python3 proxy/embedding_server.py            # --model/--host/--port to override
```

Then point the proxy at it with `PROXY_ROUTER_URL=http://127.0.0.1:8890`.
If the service is down the router's fail-open path already returns
passthrough, so chat traffic is unaffected.

## Skill index injection (`PROXY_SKILLS_DIR`)

When `PROXY_SKILLS_DIR` points at a directory of skills, the proxy stops
waiting for the model to discover them: it embeds each skill once, ranks them
by cosine against the last user message on every chat request, and appends the
best matches to the end of the message list as a trailing user message. This
needs the embedding service (`PROXY_ROUTER_URL`); without it the feature stays
off. The tail is the only place the chat template accepts an injection: it
refuses a system message that is not first, and the leading system message
must stay byte-identical or the cached prefix is lost. A pinned skill is the
exception — its body never changes, so it joins the leading system message and
costs nothing in the cache.

Layouts: `<dir>/<name>/SKILL.md` (primary) or `<dir>/*.md` (flat). The skill
name is the directory name or file stem. The description comes from a
`description:` line in YAML frontmatter between leading `---` markers; when
absent, the first non-heading paragraph is used, truncated to 200 characters.
The directory is scanned once (lazily on the first chat request) and the
vectors are cached until restart. The scan is sorted by name (path as
tie-break), so adding a skill cannot reshuffle the existing lines.

The top `PROXY_SKILLS_TOP_K` (default **3**) skills whose cosine similarity is
at least `PROXY_SKILLS_MIN_SIM` (default **0.25**) are appended as one system
block after the conversation:

```
Relevant skills (read the file before acting when it applies):
- <name>: <description> [<path>]
```

The block is only added when skills qualify, never twice in one request, and
ranks deterministically, so the same skill set produces byte-identical text.
Because it trails the conversation, a changed block leaves the prompt head
untouched for the prefix cache. Each injection logs `skills: injected N
(top sim X.XXX)` to stderr.

**Threshold note:** 0.25 is a heuristic default for embeddinggemma, chosen so
loosely related skills still surface. Calibrate it from the logged
similarities and lower it if relevant skills are being filtered out.

### Pinned skills (`PROXY_SKILLS_PIN`)

A skill named in `PROXY_SKILLS_PIN` (comma-separated) is always injected,
regardless of similarity. Its body is inlined after a `## <name>` heading
instead of the one-line pointer, so a harness that cannot read the path still
gets the rules. The body is stripped of YAML frontmatter and capped at
`PROXY_SKILLS_PIN_CHARS` (default **4000**), cut at a line boundary.

A pin takes the whole request: the directory is not scanned, no skills are
embedded or listed, and only the pinned bodies (in declaration order) are
injected. That keeps the per-request block fixed instead of a volatile listing
on every turn.

```
PROXY_SKILLS_DIR=skills PROXY_SKILLS_PIN=ponytail \
  python3 proxy/score_proxy.py
```

Pinned skills are policy, not retrieval: use them sparingly, because their
text is charged to every matching request. `skills/ponytail` ships in this
repository (MIT, see NOTICE) as the reference pin.

**Fail-open:** any scan or embedding error leaves the index empty and requests
pass through unchanged. If the embedding service is down on the first request,
the index stays empty until the proxy restarts. A pin that names a missing
file injects nothing.

**Prefix-cache note:** the injected block varies by request, so it is kept
after the conversation; the transcript ahead of it stays byte-identical and
the server can reuse its cached prefix. The block itself is still sent (and
charged) on every turn. Enable this only when skill discovery is actually the
failure mode; for a large static skill set, pre-filtering in the client is
cheaper.

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
- [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) — multi-syntax
  tool-call parsing, relaxed-JSON and schema-typed argument recovery, the
  reply-reserve / close-thinking retry design, the prefix-cache-safe prompt
  layout (volatile injected context after the stable transcript), and
  byte-accounted tool-output memory (identical tool results share one
  block, size tracked in bytes).
- [ashhart/TensorFold](https://github.com/ashhart/TensorFold) — the
  compaction pre-pass that replaces byte-identical `tool` results after the
  first with a placeholder before a costly context operation; the gated
  duplicate tool-output collapse is an independent reimplementation with a
  position- and byte-count-preserving marker.
- [zolotukhin/zinc](https://github.com/zolotukhin/zinc) — ChatML
  `<tool_call>` envelope parsing and schema-aware argument decoding.
- [uv-genai/colibri](https://github.com/uv-genai/colibri) — strict
  tool-call parsing with declared-type argument coercion.
- [MistVVK/XeStrata](https://github.com/MistVVK/XeStrata) — fenced-JSON
  call recovery, reasoning-channel calls, close-thinking retry and
  answer-room reserve.

All five are idea-level credits: the proxy code is an independent
reimplementation; see `NOTICE`.
