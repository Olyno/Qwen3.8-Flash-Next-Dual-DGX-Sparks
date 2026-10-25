#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""proxy/score_proxy.py — thin OpenAI-compatible decision proxy for vLLM.

Passthrough reverse proxy (SSE streams through chunk-by-chunk) plus
decision features and a non-streaming repair pass — see
docs/score-proxy.md:

  POST /v1/score              score N options in one batched prefill
  POST /v1/chat/completions   confidence-gated thinking escalation
                              ("escalate": true or PROXY_ESCALATE=1);
                              streaming requests get a degenerate-repetition
                              loop guard (PROXY_LOOP_GUARD=0 disables).
                              Opt-in cache care: PROXY_CANONICALIZE=1
                              hoists system blocks ahead of the first user
                              message (reorder only; PROXY_CANONICALIZE_STRIP_JITTER=1
                              also strips trailing 'Current time:'/'Date:'
                              lines from system messages);
                              PROXY_CACHE_KEEPALIVE_S=N re-touches APC
                              recency for clients idle past N seconds;
                              PROXY_ROUTER_URL=http://host:port enables the
                              embedding router: the last user message is
                              classified trivial/hard against prototype
                              centroids and 'trivial' requests are rewritten
                              pre-hoc (thinking off, max_tokens capped);
                              PROXY_SKILLS_DIR=path surfaces the top-k skills
                              relevant to the last user message in a trailing
                              system block after the conversation; a pinned
                              skill is read directly and nothing else is
                              listed.
                              Non-streaming replies are repaired in place:
                              malformed call syntax is normalized to
                              tool_calls and schema-typed, a reply that stops
                              inside thinking is continued once with the
                              thinking closed, and an explicit max_tokens
                              larger than the context is clamped.
                              Every completed upstream chat completion logs
                              one token accounting line (prompt/cached/
                              completion/reasoning) from the usage object;
                              streams read the usage chunk when one arrives.
                              Opt-in duplicate tool-output collapse
                              (PROXY_DEDUPE_TOOL_OUTPUT=1) replaces a repeated
                              role=tool result with a marker naming the first
                              occurrence and its byte size, leaving the first
                              message and every non-tool message untouched.

Env: PROXY_UPSTREAM (default http://localhost:8888), PROXY_PORT (8889),
PROXY_ESCALATE, PROXY_ESCALATE_THRESHOLD (-0.35),
PROXY_LOOP_GUARD (1), PROXY_LOOP_GUARD_REPEAT (3),
PROXY_ROUTER_URL (unset = off), PROXY_ROUTER_MODEL
(google/embeddinggemma-2), PROXY_ROUTER_THRESHOLD (0.10),
PROXY_ROUTER_TRIVIAL_MAX_TOKENS (1024), PROXY_ROUTER_TIMEOUT_S (2),
PROXY_TOOL_ROUTER (unset = off; scores whether a request with tools needs
one, then nudges), PROXY_TOOL_THRESHOLD (0.5), PROXY_TOOL_FORCE (1 = set
tool_choice=required instead of the system nudge), PROXY_TOOL_TOP_K (8;
prune larger tool lists by embedding similarity), PROXY_TOOL_ALWAYS
(comma-separated tool names never pruned), PROXY_SKILLS_DIR (unset = off;
directory of <name>/SKILL.md or *.md skills), PROXY_SKILLS_TOP_K (3),
PROXY_SKILLS_MIN_SIM (0.25, heuristic — calibrate per workload),
PROXY_SKILLS_PIN (comma-separated skill names whose bodies are inlined
instead of enumerated), PROXY_SKILLS_PIN_CHARS (4000),
PROXY_CONTEXT_TOKENS (0 = off; model context size; an explicit max_tokens
is lowered to the room the prompt leaves, while a prompt that alone
overflows is left untouched for the server to reject),
PROXY_DEDUPE_TOOL_OUTPUT (unset = off; collapse byte-identical repeated
role=tool results into a marker that keeps the earlier position and size).
PROXY_DIFF_TOOL_OUTPUT (unset = off; when the same call returns *changed*
content, send a unified diff against its first result instead of the whole
payload, and only when that diff is under half the payload's size).
Stdlib only.
"""
import ast
import difflib
import hashlib
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP_BY_HOP = frozenset(("connection", "keep-alive", "proxy-authenticate",
                        "proxy-authorization", "te", "trailer",
                        "transfer-encoding", "upgrade"))


def softmax(scores):
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def build_prefix(question, options):
    # JEV prompt rules: one multi-option question, use-when descriptions inline.
    lines = [question.strip(), "", "Options:"]
    for i, opt in enumerate(options, 1):
        line = f"{i}. {opt['label']}"
        if opt.get("use_when"):
            line += f" — use when: {opt['use_when']}"
        lines.append(line)
    lines += ["", "Answer: "]
    return "\n".join(lines)


MIN_PHRASE = 20   # shortest repeated phrase (normalized chars) the guard sees
WINDOW = 2048     # rolling normalized text kept for detection
PAIR_RUN = 16     # 'ab' x PAIR_RUN (2-char unit) also counts as a loop


def _degenerate_tail(text, repeat):
    """True if the normalized text ends in the same >=MIN_PHRASE phrase
    repeated `repeat` times, or in a 2-char unit repeated PAIR_RUN times.
    Tail-anchored on purpose: legit repetition that already ended (code,
    tables) must not trigger — only a loop still running at the frontier."""
    n = len(text)
    if n >= 2 * PAIR_RUN and text.endswith(text[n - 2:] * PAIR_RUN):
        return True
    for p in range(MIN_PHRASE, n // repeat + 1):
        # Necessary condition, and far cheaper than building the slice:
        # a true repeat has the same character at p and 2p from the end.
        if text[n - p] != text[n - 2 * p]:
            continue
        if text.endswith(text[n - p:] * repeat):
            return True
    return False


def _sse_objects(pending, chunk):
    """Split an SSE byte stream into complete events and collect the JSON
    payloads; returns (objects, remainder). Unparseable events are relayed by
    the caller untouched, they just contribute nothing here."""
    *events, pending = (pending + chunk).split(b"\n\n")
    objects = []
    for event in events:
        for line in event.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                objects.append(json.loads(payload))
            except ValueError:
                continue
    return objects, pending


class LoopGuard:
    """Streaming repetition detector for SSE chat responses. Raw chunks are
    relayed untouched; only delta text is kept (rolling, normalized) for
    detection. ponytail: tail-anchored check + 20-char minimum can still
    false-positive on pathological legit output (e.g. 3+ identical long JSON
    rows back-to-back, or 32+ dashes of table separator at the frontier) —
    PROXY_LOOP_GUARD=0 is the escape hatch."""

    def __init__(self, repeat):
        self.repeat = repeat
        self._pending = b""  # bytes of an SSE event split across chunks
        self._text = ""      # rolling normalized window

    def feed(self, chunk):
        """Inspect one raw upstream chunk; True => degenerate loop detected."""
        objects, self._pending = _sse_objects(self._pending, chunk)
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                self._text = (self._text
                              + " ".join(piece.split()).lower())[-WINDOW:]
                if _degenerate_tail(self._text, self.repeat):
                    return True
        return False

    @staticmethod
    def final_chunk():
        # finish_reason 'stop' (not 'content_filter'/'length': neither is what
        # happened — the proxy cut a loop). The cut stays machine-detectable
        # via a top-level extension field; delta text is left unpolluted and
        # SSE framing stays valid.
        note = json.dumps({"choices": [{"index": 0, "delta": {},
                                        "finish_reason": "stop"}],
                           "x_proxy": {"loop_guard": "cut"}})
        return f"data: {note}\n\ndata: [DONE]\n\n".encode()


class StreamUsage:
    """Collects the `usage` object an SSE stream carries on its last chunk.
    Relayed bytes are untouched: usage is read out of band and logged once
    the stream ends, so a server that reports nothing simply logs
    'unavailable'."""

    def __init__(self):
        self._pending = b""
        self.usage = None

    def feed(self, chunk):
        objects, self._pending = _sse_objects(self._pending, chunk)
        for obj in objects:
            if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
                self.usage = obj["usage"]

    def log(self):
        log_usage(self.usage)


def log_usage(usage):
    """One accounting line for a completed upstream chat response. vLLM with
    prompt-token details reports cache hits under prompt_tokens_details and
    thinking under completion_tokens_details; flat fields are accepted too.
    Missing counters are omitted, so the line says exactly what the server
    reported."""
    usage = usage if isinstance(usage, dict) else {}
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    fields = (
        ("prompt_tokens", usage.get("prompt_tokens")),
        ("cached_tokens", prompt.get("cached_tokens",
                                     usage.get("cached_tokens"))),
        ("completion_tokens", usage.get("completion_tokens")),
        ("reasoning_tokens", completion.get("reasoning_tokens",
                                            usage.get("reasoning_tokens"))),
    )
    reported = " ".join(
        f"{name}={value}" for name, value in fields
        if isinstance(value, int) and not isinstance(value, bool))
    print(f"score proxy: usage {reported or 'unavailable'}", file=sys.stderr)


def post_json(upstream, path, body, timeout=300):
    conn = http.client.HTTPConnection(upstream[0], upstream[1], timeout=timeout)
    try:
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def post_buffered(upstream, path, body, headers=None, timeout=600):
    """POST and buffer the whole response. Repair must see the complete body;
    callers relay the raw bytes when nothing fired."""
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    conn = http.client.HTTPConnection(upstream[0], upstream[1], timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers=request_headers)
        resp = conn.getresponse()
        return resp.status, resp.getheaders(), resp.read()
    finally:
        conn.close()


def score_order(upstream, model, prefix, labels):
    """One batched echo-logprobs call: prompt = prefix + label per option;
    sum each label's token logprobs past the shared prefix, softmax."""
    body = json.dumps({
        "model": model,
        "prompt": [prefix + label for label in labels],
        "echo": True, "logprobs": 0, "max_tokens": 1, "temperature": 0,
    }).encode()
    status, resp = post_json(upstream, "/v1/completions", body)
    if status != 200:
        raise ValueError(f"upstream /v1/completions returned {status}")
    scores = []
    for choice in sorted(resp["choices"], key=lambda c: c["index"]):
        lp = choice["logprobs"]
        scores.append(sum(t for t, off in zip(lp["token_logprobs"], lp["text_offset"])
                          if off >= len(prefix) and t is not None))
    return softmax(scores)


JITTER_RE = re.compile(
    r"(?:\s*\n\s*(?:Current (?:date|time|datetime)|Date|Time)\s*:[^\n]*)+\s*$")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# Router prototype phrases, embedded once and averaged into class centroids.
ROUTE_PROTOTYPES = {
    "trivial": [
        "What time is it?",
        "What is the capital of France?",
        "Translate 'good morning' to Spanish.",
        "What is 17 plus 25?",
        "Define the word 'verbose'.",
        "Convert 10 miles to kilometers.",
        "List three primary colors.",
    ],
    "hard": [
        "Prove that the square root of two is irrational.",
        "Design a rate limiter for a distributed API gateway.",
        "Debug this race condition in my async code.",
        "Analyze the time and space complexity of this algorithm.",
        "Write a parser for a small expression language.",
        "Explain the trade-offs between consensus protocols.",
        "Refactor this module to remove the circular dependency.",
    ],
}


def last_user_text(messages):
    """Text of the last user message ('' if none); multipart content is
    flattened to its text parts."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") == "text")
    return ""


TAIL_FRAME = "Context for the request above, not a new request:\n\n"


def _append_user_tail(messages, block):
    """Volatile proxy context goes after the conversation. Prefix caching
    matches from the start of the prompt, so text that can change between
    turns must never sit ahead of the stable transcript. A trailing user
    message is the only tail the chat template accepts (it rejects a system
    message that is not first), and the frame keeps it from reading as a new
    request."""
    messages.append({"role": "user", "content": TAIL_FRAME + block})


def _append_system_head(messages, block):
    """Constant proxy context goes into the leading system message: it is the
    same bytes on every request, so it costs nothing in the prefix cache and
    keeps the authority a policy block needs."""
    for message in messages:
        if (isinstance(message, dict) and message.get("role") == "system"
                and isinstance(message.get("content"), str)):
            message["content"] += "\n\n" + block
            return
    messages.insert(0, {"role": "system", "content": block})


TOOL_NEED_OPTIONS = [
    {"label": "use_tool",
     "use_when": "the request needs reading, searching, running, or changing "
                 "anything outside this conversation"},
    {"label": "answer",
     "use_when": "the request can be answered from general knowledge or the "
                 "conversation alone"},
]
TOOL_NUDGE = ("You have tools. When a request can be checked, read, or "
              "changed with a tool, call the tool before answering. Answer "
              "from memory only when no tool applies.")


class ToolRouter:
    """Tool-use steering: score whether the last user turn needs a
    tool, then nudge or force before generation. The score is a two-option
    prefill (same pattern as /v1/score, both orders averaged); `force` sets
    tool_choice='required' instead of appending a system nudge; with an
    embedding function it also prunes an oversized tool list to the top-k by
    cosine against the last user message (always-keep names survive).
    Fail-open: scoring/embedding trouble means passthrough. Calibrate the
    threshold on the logged `tool steer` lines before trusting it."""

    def __init__(self, score_fn, embed_fn=None, threshold=0.5, force=False,
                 top_k=8, always=()):
        self.score_fn = score_fn        # (model, prefix, labels) -> probs
        self.embed_fn = embed_fn        # (texts) -> vectors, or None
        self.threshold = threshold
        self.force = force
        self.top_k = top_k
        self.always = {n.lower() for n in always}

    def _prune(self, req):
        tools = req.get("tools") or []
        if self.embed_fn is None or len(tools) <= self.top_k:
            return False
        text = last_user_text(req.get("messages") or [])[:2000]
        if not text:
            return False
        names = [((t.get("function") or {}).get("name") or "") for t in tools]
        descs = [f"{n}. {((t.get('function') or {}).get('description') or '')}"
                 for n, t in zip(names, tools)]
        try:
            vecs = self.embed_fn([text] + descs)
        except Exception:  # fail-open: keep the full tool list
            return False
        q = vecs[0]
        order = sorted(range(len(tools)), key=lambda i: -cosine(q, vecs[i + 1]))
        keep = set(order[:self.top_k])
        keep |= {i for i, n in enumerate(names) if n.lower() in self.always}
        if len(keep) == len(tools):
            return False
        req["tools"] = [t for i, t in enumerate(tools) if i in keep]
        return True

    def needs_tool(self, messages, model):
        text = last_user_text(messages)[:2000]
        if not text:
            return 0.0
        labels = [o["label"] for o in TOOL_NEED_OPTIONS]
        try:
            probs = self.score_fn(model, build_prefix(text, TOOL_NEED_OPTIONS),
                                  labels)
            rev = self.score_fn(model, build_prefix(text, list(reversed(
                TOOL_NEED_OPTIONS))), labels)
        except Exception:  # fail-open: endpoint down/slow/garbage = passthrough
            return 0.0
        # position-bias mitigation (JEV): average the forward and reversed
        # option orders.
        probs = [(a + b) / 2 for a, b in zip(probs, rev)]
        return dict(zip(labels, probs))["use_tool"]

    def apply(self, req, model):
        """Returns (changed, prob). Mutates req in place."""
        if not req.get("tools"):
            return False, 0.0
        pruned = self._prune(req)
        prob = self.needs_tool(req.get("messages") or [], model)
        if prob < self.threshold:
            return pruned, prob
        if self.force:
            if req.get("tool_choice"):
                return pruned, prob   # client already decided; respect it
            req["tool_choice"] = "required"
            return True, prob
        messages = req.get("messages") or []
        if any(isinstance(m, dict) and m.get("role") == "system"
               and TOOL_NUDGE in (m.get("content") or "") for m in messages):
            return pruned, prob
        _append_user_tail(messages, TOOL_NUDGE)
        req["messages"] = messages
        return True, prob


# ── tool-call / truncated-answer repair ─────────────────────────────────
# Post-hoc repair for a model that emits call syntax the harness rejects or
# runs out of budget mid-thought. Every path is a no-op unless the reply is
# actually malformed; a well-formed response is relayed as the raw upstream
# bytes.

ANSWER_RESERVE_MIN = 512   # answer tokens kept when thinking is cut short
ANSWER_ROOM_MIN_CAP = 1024
THINK_RETRY_MAX = 1024     # single close-thinking continuation budget
CHARS_PER_TOKEN = 4        # clamp estimate only; no tokenizer in the proxy

FENCE_RE = re.compile(
    r"```[ \t]*(?:json|tool_call|function)?[ \t]*\r?\n?(.*?)(?:```|$)",
    re.S | re.I)
TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call\b[^>]*>(.*?)(?:</tool_call\s*>|$)", re.S | re.I)
FUNCTION_BLOCK_RE = re.compile(
    r"<function\s*(?:=\s*|name\s*=\s*)?['\"]?([^'\"\s>]+)['\"]?\s*>"
    r"(.*?)(?:</function\s*>|(?=<function\s)|$)", re.S | re.I)
INVOKE_BLOCK_RE = re.compile(
    r"<invoke\s+name\s*=\s*['\"]?([^'\"\s>]+)['\"]?\s*>"
    r"(.*?)(?:</invoke\s*>|$)", re.S | re.I)
PARAM_BLOCK_RE = re.compile(
    r"<parameter\s*(?:=\s*|name\s*=\s*)?['\"]?([^'\"\s>]+)['\"]?\s*>"
    r"(.*?)(?:</parameter\s*>|(?=<parameter\s)|$)", re.S | re.I)
THINK_OPEN_RE = re.compile(r"<(think|reasoning)\b[^>]*>", re.I)
THINK_CLOSE_RE = re.compile(r"</(?:think|reasoning)\s*>", re.I)
NUMBER_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[^\d\s]*\s*$")
CALL_ARG_KEYS = ("arguments", "parameters", "args", "params", "input")


def estimate_tokens(value):
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def tool_schemas(tools):
    """Declared tool name -> its parameters schema (both OpenAI
    `function.parameters` and Anthropic `input_schema` shapes)."""
    schemas = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        fn = fn if isinstance(fn, dict) else tool
        name = fn.get("name")
        if isinstance(name, str) and name.strip():
            schema = fn.get("parameters")
            if not isinstance(schema, dict):
                schema = fn.get("input_schema")
            schemas[name.strip()] = schema if isinstance(schema, dict) else {}
    return schemas


def _declared_type(prop):
    typ = prop.get("type") if isinstance(prop, dict) else None
    if isinstance(typ, list):
        typ = next((x for x in typ if isinstance(x, str) and x != "null"), None)
    return typ if isinstance(typ, str) else None


def _coerce_value(value, prop):
    typ = _declared_type(prop)
    if typ is None:
        return value                       # nothing declared: never guess
    if typ == "integer":
        if isinstance(value, bool) or isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if isinstance(value, str):
            match = NUMBER_RE.match(value)
            if match:
                number = float(match.group(1))
                if number.is_integer():
                    return int(number)
        return value
    if typ == "number":
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            match = NUMBER_RE.match(value)
            if match:
                number = float(match.group(1))
                return int(number) if number.is_integer() else number
        return value
    if typ == "boolean":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "false"):
                return lowered == "true"
        return value
    if typ == "array":
        items = prop.get("items")
        items = items if isinstance(items, dict) else None
        if isinstance(value, str) and value.lstrip().startswith("["):
            parsed = _loads_tolerant(value)
            if isinstance(parsed, list):
                value = parsed
        if not isinstance(value, list):
            value = [value]
        return [_coerce_value(item, items) for item in value] if items else value
    if typ == "object":
        if isinstance(value, str):
            parsed = _loads_tolerant(value)
            if isinstance(parsed, dict):
                return coerce_arguments(parsed, prop)
        if isinstance(value, dict):
            return coerce_arguments(value, prop)
        return value
    return value


def coerce_arguments(args, schema):
    """Coerce only keys the schema declares, and only their existing values:
    a missing argument is never invented, an undeclared one is never typed."""
    if not isinstance(args, dict) or not isinstance(schema, dict):
        return args
    props = schema.get("properties")
    if not isinstance(props, dict):
        return args
    out = dict(args)
    for key, value in out.items():
        prop = props.get(key)
        if isinstance(prop, dict):
            out[key] = _coerce_value(value, prop)
    return out


def _loads_tolerant(text):
    """Parse JSON-ish text: strict JSON, Python literals (single-quoted
    keys/strings), trailing commas, and a truncated tail with its open
    quote/bracket closed and a dangling key dropped."""
    if not isinstance(text, str):
        return None
    variants = []
    for candidate in (text.strip(), _strip_fence(text)):
        if candidate and candidate not in variants:
            variants.append(candidate)
    for candidate in variants:
        closed = _close_json(candidate)
        for attempt in (candidate, closed,
                        re.sub(r",\s*([}\]])", r"\1", closed)):
            if not attempt:
                continue
            try:
                return json.loads(attempt)
            except ValueError:
                pass
            try:
                return ast.literal_eval(attempt)
            except (ValueError, SyntaxError):
                pass
    return None


def _strip_fence(text):
    text = text.strip()
    if not text.startswith("```"):
        return text
    newline = text.find("\n")
    text = text[newline + 1:] if newline >= 0 else text[3:]
    text = text.rstrip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _close_json(text):
    stack, quote, escaped = [], "", False
    for char in text:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    out = text + quote if quote else text
    bare_key = r"(?:\"[^\"]*\"|'[^']*'|[A-Za-z_][\w.-]*)"
    out = re.sub(r",\s*" + bare_key + r"\s*:\s*$", "", out)
    out = re.sub(r"\{\s*" + bare_key + r"\s*:\s*$", "{", out)
    out = re.sub(r",\s*$", "", out)
    return out + "".join(reversed(stack))


def _balanced_span(text, start):
    """`text[start:]` through its matching brace (or to the end)."""
    stack, quote, escaped = [], "", False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
            if not stack:
                return text[start:index + 1]
    return text[start:]


def _arguments(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _loads_tolerant(value)
        return parsed if isinstance(parsed, dict) else None
    return None


def _outer_arguments(obj):
    return _arguments(next((obj[k] for k in CALL_ARG_KEYS if k in obj), {}))


def _call_from_object(obj, wrapped=False):
    """A dict -> (name, arguments) when it has a call shape; else None."""
    if not isinstance(obj, dict):
        return None
    for key in ("tool_call", "function_call", "invoke"):
        inner = obj.get(key)
        if isinstance(inner, dict):
            call = _call_from_object(inner, True)
            if call is not None:
                name, args = call
                return name, args or _outer_arguments(obj) or {}
        elif isinstance(inner, str) and key == "invoke":
            args = _outer_arguments(obj)
            if args is not None and inner.strip():
                return inner.strip(), args
    fn = obj.get("function")
    if isinstance(fn, dict):
        call = _call_from_object(fn, True)
        if call is not None:
            name, args = call
            return name, args or _outer_arguments(obj) or {}
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) and isinstance(fn, str):
        name = fn                          # {"function": "run", "arguments": ...}
    if not isinstance(name, str) or not name.strip():
        return None
    has_args = any(key in obj for key in CALL_ARG_KEYS)
    if not has_args and not wrapped:
        return None
    args = _outer_arguments(obj)
    if args is None:
        return None
    return name.strip(), args


def _xml_parameters(body):
    args = {}
    for match in PARAM_BLOCK_RE.finditer(body):
        value = match.group(2)
        # A parameter left open at the end of the reply swallows the wrapper
        # close tags; they are structure, not part of the value.
        value = re.split(r"</(?:function|invoke|tool_call)\s*>", value,
                         maxsplit=1, flags=re.I)[0]
        args[match.group(1)] = value.strip()
    return args


def _calls_from_xml(text):
    blocks = TOOL_CALL_BLOCK_RE.findall(text)
    if blocks:
        calls = []
        for body in blocks:
            functions = list(FUNCTION_BLOCK_RE.finditer(body))
            if functions:
                calls.extend((match.group(1), _xml_parameters(match.group(2)))
                             for match in functions)
            else:
                calls.extend(_calls_from_json(body))
        return calls, True
    for regex in (INVOKE_BLOCK_RE, FUNCTION_BLOCK_RE):
        matches = list(regex.finditer(text))
        if matches:
            return [(match.group(1), _xml_parameters(match.group(2)))
                    for match in matches], True
    return [], False


def _tail_is_empty(text, end):
    """True when nothing but punctuation follows the span. A call object in the
    middle of an answer is prose quoting an example, not an intended call."""
    return not text[end:].strip(" \t\r\n.,;:)]}>")


def _calls_from_json(text):
    for match in FENCE_RE.finditer(text):
        call = _call_from_object(_loads_tolerant(match.group(1)))
        if call is not None:
            return [call]
    # Bounded scan: a long non-call answer full of braces must not turn
    # repair into an O(n²) parse.
    position = attempts = 0
    while attempts < 64:
        attempts += 1
        start = text.find("{", position)
        if start < 0:
            return []
        span = _balanced_span(text, start)
        call = _call_from_object(_loads_tolerant(span))
        if call is not None and _tail_is_empty(text, start + len(span)):
            return [call]
        position = start + max(1, len(span) if span else 1)
    return []


def extract_tool_calls(text):
    """Recover (name, arguments) pairs from a reply's text, in every syntax
    the model has been seen to emit around the declared one."""
    if not isinstance(text, str) or not text.strip():
        return []
    calls, wrapped = _calls_from_xml(text)
    if wrapped:
        return calls
    return _calls_from_json(text)


def _call_entry(name, arguments):
    serialized = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha1(f"{name}\0{serialized}".encode()).hexdigest()[:24]
    return {"id": f"call_{digest}", "type": "function",
            "function": {"name": name, "arguments": serialized}}


def recover_tool_calls(message, schemas):
    """Add a `tool_calls` entry recovered from content, or from the reasoning
    channel when that is where the call died. Returns True when it fired."""
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    calls = extract_tool_calls(content) if isinstance(content, str) else []
    reasoning = message.get("reasoning_content")
    if (not calls and isinstance(reasoning, str)
            and (not isinstance(content, str) or not content.strip())):
        calls = extract_tool_calls(reasoning)
    entries = []
    for name, arguments in calls:
        schema = schemas.get(name)
        if schema is None:
            continue                       # undeclared name: not a valid call
        entries.append(_call_entry(name, coerce_arguments(arguments, schema)))
    if not entries:
        return False
    message["tool_calls"] = entries
    return True


def open_thinking(message):
    """(field, tag, thinking_text) when the reply stops inside a thinking
    block with no answer before it; else None."""
    content = message.get("content")
    if isinstance(content, str):
        opens = list(THINK_OPEN_RE.finditer(content))
        if opens:
            match = opens[-1]
            tag = match.group(1).lower()
            closed_at = content.lower().rfind(f"</{tag}>")
            if closed_at < match.start():
                if content[:match.start()].strip():
                    return None            # text before the block is an answer
                return "content", tag, content[match.end():]
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        if isinstance(content, str) and content.strip():
            return None                    # the answer arrived
        if THINK_CLOSE_RE.search(reasoning):
            return None                    # thinking ended on purpose
        return "reasoning_content", "think", reasoning
    return None


def answer_room(max_tokens):
    """Answer budget after the thinking is cut: a quarter of the request cap
    (at least ANSWER_RESERVE_MIN, at most half of it), bounded small because
    this is one continuation. A cap under ANSWER_ROOM_MIN_CAP cannot fit both
    parts, so the whole cap becomes answer room."""
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        return THINK_RETRY_MAX
    if max_tokens < ANSWER_ROOM_MIN_CAP:
        return max_tokens
    reserve = min(max(ANSWER_RESERVE_MIN, max_tokens // 4), max_tokens // 2)
    return min(THINK_RETRY_MAX, reserve)


def clamp_max_tokens(req, context_tokens):
    """Pre-send fit check: lower an explicit max_tokens to the room the
    prompt leaves. When the prompt alone does not fit, the request is left
    byte-identical — conversation content is never truncated here, and the
    server's own overflow answer is more informative than a 1-token reply
    from a silently shrunk request.
    ponytail: 4 chars/token is a heuristic; without a tokenizer the proxy can
    only under-reserve, which leaves the server's own truncation in place."""
    max_tokens = req.get("max_tokens")
    if not context_tokens or not isinstance(max_tokens, int) or max_tokens <= 0:
        return False
    prompt = estimate_tokens(req.get("messages") or [])
    if req.get("tools"):
        prompt += estimate_tokens(req["tools"])
    room = context_tokens - prompt
    if max_tokens <= room:
        return False
    if room < 1:
        return False          # prompt alone overflows: let the server answer
    req["max_tokens"] = room
    return True


def _thinking_retry(req, field, tag, thinking, used, context_tokens):
    """One continuation with the thinking already closed. The template
    renders `reasoning_content` inside the close tag, which is the same cut
    the reply needed; every token of the retry budget is answer room."""
    request = dict(req)
    request["messages"] = list(req.get("messages") or []) + [{
        "role": "assistant", "content": "",
        "reasoning_content": thinking}]
    request["continue_final_message"] = True
    request["add_generation_prompt"] = False
    request["stream"] = False
    budget = answer_room(req.get("max_tokens"))
    if context_tokens:
        prompt = estimate_tokens(req.get("messages") or [])
        if req.get("tools"):
            prompt += estimate_tokens(req["tools"])
        room = context_tokens - prompt - max(0, used or 0)
        budget = max(1, min(budget, room))
    request["max_tokens"] = budget
    return request


def apply_thinking_retry(resp, retry_resp, plan):
    """Merge a close-thinking continuation into the original reply. Returns
    False (and leaves it alone) when the retry produced no answer."""
    choices = resp.get("choices") or []
    retry_choices = retry_resp.get("choices") or []
    index = plan.get("index", 0)
    if index >= len(choices) or not retry_choices:
        return False
    choice, retry_choice = choices[index], retry_choices[0]
    if not isinstance(choice, dict) or not isinstance(retry_choice, dict):
        return False
    message = choice.get("message")
    retry_message = retry_choice.get("message")
    if not isinstance(message, dict) or not isinstance(retry_message, dict):
        return False
    calls = retry_message.get("tool_calls")
    answer = retry_message.get("content")
    if not isinstance(answer, str) or not answer.strip():
        answer = None
    if not calls and answer is None:
        return False
    if plan["field"] == "content":
        cut = plan["content"].rstrip()
        closed = f"{cut}\n\n</{plan['tag']}>"
        message["content"] = f"{closed}\n\n{answer}" if answer else closed
    elif answer is not None:
        message["content"] = answer
    if calls:
        message["tool_calls"] = calls
        choice["finish_reason"] = "tool_calls"
    else:
        choice["finish_reason"] = retry_choice.get("finish_reason") or "stop"
    usage, retry_usage = resp.get("usage"), retry_resp.get("usage")
    if isinstance(usage, dict) and isinstance(retry_usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            first, second = usage.get(key), retry_usage.get(key)
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                usage[key] = first + second
    if not isinstance(resp.get("x_proxy"), dict):
        resp["x_proxy"] = {}
    resp["x_proxy"]["thinking_retry"] = "closed"
    return True


def repair_reply(resp, req, context_tokens=0, allow_retry=True):
    """Repair one non-streaming chat response in place. Returns
    (changed, retry_plan); `retry_plan` is at most one close-thinking
    continuation for the caller to issue."""
    if not isinstance(resp, dict):
        return False, None
    schemas = tool_schemas(req.get("tools"))
    changed = False
    retry = None
    for index, choice in enumerate(resp.get("choices") or []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if schemas and not message.get("tool_calls"):
            if recover_tool_calls(message, schemas):
                choice["finish_reason"] = "tool_calls"
                changed = True
        if (retry is None and allow_retry and not message.get("tool_calls")
                and choice.get("finish_reason") in ("length", "stop")):
            opened = open_thinking(message)
            if opened is not None:
                field, tag, thinking = opened
                usage = resp.get("usage")
                used = (usage.get("completion_tokens") or 0
                        if isinstance(usage, dict) else 0)
                retry = {
                    "request": _thinking_retry(
                        req, field, tag, thinking, used, context_tokens),
                    "field": field, "tag": tag, "index": index,
                    "content": message.get("content") or "",
                }
    return changed, retry


SKILLS_MARKER = "Relevant skills (read the file before acting when it applies):"
SKILL_DESC_MAX = 200


def skill_description(text):
    """Skill description from YAML frontmatter (`description:` between
    leading `---` markers), else the first non-heading paragraph, truncated.
    A missing frontmatter description falls through to the body so a bare
    SKILL.md still gets a usable line."""
    lines = text.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for end in range(1, len(lines)):
            if lines[end].strip() == "---":
                for line in lines[1:end]:
                    key, sep, value = line.partition(":")
                    if sep and key.strip() == "description":
                        desc = value.strip().strip("'\"")
                        if desc:
                            return desc
                start = end + 1
                break
    para = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if para:
                break
            continue
        if stripped.startswith("#"):
            if para:
                break
            continue
        para.append(stripped)
    return " ".join(para)[:SKILL_DESC_MAX]


class SkillIndex:
    """Preemptive skill surfacing: scan a skills directory once, embed each
    skill's name + description through the embedding endpoint, and append the
    top-k matches to the end of the message list as a system message. Because
    the block trails the conversation, a changed block leaves the stable head
    byte-identical. With a pin the directory is not scanned at all: the pinned
    bodies are read directly and nothing else is listed. Fail-open: any
    scan/embedding error leaves the index empty and requests pass through
    unchanged. ponytail: one lazy scan cached forever — restart the proxy
    after editing the skills directory."""

    def __init__(self, directory, embed_fn, top_k=3, min_sim=0.25,
                 pin=(), pin_chars=4000):
        self.directory = directory
        self.embed_fn = embed_fn
        self.top_k = top_k
        self.min_sim = min_sim
        self.pin = tuple(n.strip() for n in pin if n.strip())
        self.pin_chars = pin_chars
        self._skills = None
        self._lock = threading.Lock()

    def _scan(self):
        found = []
        for entry in os.listdir(self.directory):
            path = os.path.join(self.directory, entry)
            if os.path.isdir(path):
                skill = os.path.join(path, "SKILL.md")
                if os.path.isfile(skill):
                    found.append((entry, skill))
            elif entry.endswith(".md") and os.path.isfile(path):
                found.append((os.path.splitext(entry)[0], path))
        # Name order, path as tie-break: the directory's own iteration order
        # must never reshuffle the listing.
        found.sort()
        skills = []
        for name, path in found:
            with open(path, encoding="utf-8", errors="replace") as f:
                skills.append((name, skill_description(f.read()), path))
        return skills

    def _load(self):
        if self._skills is not None:
            return self._skills
        with self._lock:
            if self._skills is None:
                try:
                    skills = self._scan()
                    if skills:
                        vectors = self.embed_fn(
                            [f"{name}: {desc}" for name, desc, _ in skills])
                        if len(vectors) != len(skills):
                            raise ValueError("embedding count mismatch")
                        skills = [(*skill, vec)
                                  for skill, vec in zip(skills, vectors)]
                    self._skills = skills
                except Exception:  # fail-open: no skills beats a broken request
                    self._skills = []
        return self._skills

    def rank(self, text):
        """[(similarity, name, description, path)] best first, filtered by
        min_sim and capped at top_k; [] on any embedding trouble."""
        skills = self._load()
        if not skills or not text or self.embed_fn is None:
            return []
        try:
            query = self.embed_fn([text[:2000]])[0]
        except Exception:  # fail-open: endpoint down/slow/garbage = no injection
            return []
        ranked = [(cosine(query, vec), name, desc, path)
                  for name, desc, path, vec in skills]
        ranked = [s for s in ranked if s[0] >= self.min_sim]
        ranked.sort(key=lambda s: (-s[0], s[1]))
        return ranked[:self.top_k]

    def _body(self, path):
        """Pinned skills ship their rules, not a path the caller may not be
        able to read. Frontmatter is stripped and the body hard-capped."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return ""
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                text = text[end + 4:]
        trimmed = text.strip()[:self.pin_chars]
        if len(trimmed) == self.pin_chars:
            trimmed = trimmed.rsplit("\n", 1)[0]
        return trimmed

    def _pinned(self):
        """[(name, body)] for the pinned skills, in declaration order. The pin
        is policy, not retrieval: probe its two layouts directly instead of
        scanning the directory, and never enumerate anything else."""
        pinned = []
        for name in self.pin:
            if name in (".", "..") or name != os.path.basename(name):
                continue
            path = os.path.join(self.directory, name, "SKILL.md")
            if not os.path.isfile(path):
                path = os.path.join(self.directory, name + ".md")
            body = self._body(path)
            if body:
                pinned.append((name, body))
        return pinned

    def apply(self, req):
        """Returns (changed, top_sim, count); mutates req in place. The
        marker check makes injection idempotent per request. A pinned body is
        constant, so it joins the leading system message; the ranked listing
        changes as the conversation moves, so it trails the conversation and
        leaves everything ahead of it byte-identical."""
        messages = req.get("messages") or []
        if any(isinstance(m, dict) and isinstance(m.get("content"), str)
               and SKILLS_MARKER in m["content"] for m in messages):
            return False, 0.0, 0
        if self.pin:
            pinned = self._pinned()
            if not pinned:
                return False, 0.0, 0
            block = SKILLS_MARKER + "\n" + "\n".join(
                f"## {name}\n{body}" for name, body in pinned)
            _append_system_head(messages, block)
            req["messages"] = messages
            return True, 0.0, len(pinned)
        ranked = self.rank(last_user_text(messages)[:2000])
        if not ranked:
            return False, 0.0, 0
        block = SKILLS_MARKER + "\n" + "\n".join(
            f"- {name}: {desc} [{path}]" for _, name, desc, path in ranked)
        _append_user_tail(messages, block)
        req["messages"] = messages
        return True, ranked[0][0], len(ranked)


class Router:
    """Pre-hoc request router (embeddinggemma-style): embed the last user
    message, cosine-compare against trivial/hard prototype centroids; a
    'trivial' verdict by enough margin gets thinking disabled and max_tokens
    capped before generation — unlike escalation there is no double
    generation. ponytail: centroids are embedded lazily on the first routable
    request and cached forever — restart the proxy after switching embedding
    models. Threshold note: embeddinggemma-2 clusters short questions tightly,
    an exact trivial-prototype match scores margin ~0.12, so the default is
    0.10 — 0.15 would never fire."""

    def __init__(self, url, model, threshold, max_tokens, timeout):
        u = urllib.parse.urlparse(url)
        self.endpoint = (u.hostname, u.port or 80)
        self.model = model
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._centroids = None

    def embed(self, texts):
        """Public embedding call (also used by tool-list pruning)."""
        return self._embed(texts)

    def _embed(self, texts):
        body = json.dumps({"model": self.model, "input": texts}).encode()
        status, resp = post_json(self.endpoint, "/v1/embeddings", body,
                                 timeout=self.timeout)
        if status != 200:
            raise ValueError(f"embeddings endpoint returned {status}")
        data = sorted(resp["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def _centroid(self):
        if self._centroids is None:
            self._centroids = {}
            for cls, phrases in ROUTE_PROTOTYPES.items():
                vecs = self._embed(phrases)
                self._centroids[cls] = [
                    sum(v[i] for v in vecs) / len(vecs)
                    for i in range(len(vecs[0]))]
        return self._centroids

    def classify(self, text):
        """-> (route, margin); route None means passthrough. Fail-open: any
        endpoint problem returns passthrough, never an error."""
        try:
            vec = self._embed([text])[0]
            sims = {c: cosine(vec, v) for c, v in self._centroid().items()}
            margin = sims["trivial"] - sims["hard"]
            return ("trivial", margin) if margin >= self.threshold \
                else (None, margin)
        except Exception:  # fail-open: endpoint down/slow/garbage = passthrough
            return None, 0.0


def canonicalize_messages(messages, strip_jitter=False):
    """Normalize the messages array to protect the prefix cache ('Don't
    Break the Cache', arXiv 2601.06007): system messages sitting before the
    first user message are hoisted to the front in stable order, so static
    blocks stay ahead of the cache boundary. Content is never edited unless
    strip_jitter=True, and then only trailing 'Current time:'/'Date:'-style
    lines inside system messages are dropped. Returns (messages, changed)."""
    head_end = next((i for i, m in enumerate(messages)
                     if isinstance(m, dict) and m.get("role") == "user"),
                    len(messages))
    head, tail = messages[:head_end], messages[head_end:]
    is_system = lambda m: isinstance(m, dict) and m.get("role") == "system"
    out = [m for m in head if is_system(m)] + \
          [m for m in head if not is_system(m)] + tail
    if strip_jitter:
        stripped = []
        for m in out:
            if is_system(m) and isinstance(m.get("content"), str):
                content = JITTER_RE.sub("", m["content"])
                if content != m["content"]:
                    m = {**m, "content": content}
            stripped.append(m)
        out = stripped
    return out, out != messages


DEDUPE_MIN_BYTES = 512
DEDUPE_MARKER = ("[identical to the tool result above at message {position}; "
                 "{size} bytes]")


DEDUPE_MARKER = ("[identical to the {call} result at message {position}; "
                 "{size} bytes]")
DIFF_HEADER = ("[diff against the {call} result at message {position}; "
               "{old} -> {new} bytes]\n")
DIFF_CONTEXT = 2
DIFF_MAX_CHARS = 200_000     # above this the cost is CPU, not tokens
DIFF_MAX_SHARE = 0.5         # a diff this large is not a saving


def _tool_call_index(messages):
    """tool_call_id -> (name, arguments) from the assistant messages, so a
    result can be tied to the call that produced it."""
    calls = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            call_id = call.get("id")
            if isinstance(call_id, str) and isinstance(function, dict):
                calls[call_id] = (str(function.get("name") or ""),
                                  str(function.get("arguments") or ""))
    return calls


def _unified_diff(old, new):
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="before", tofile="after", n=DIFF_CONTEXT))


def collapse_duplicate_tool_output(req, min_bytes=DEDUPE_MIN_BYTES, diff=False):
    """Replace a later result of the same call with a marker, or with a diff.

    A result is touched only when the call that produced it repeats — same
    tool, same arguments. What it becomes depends on the payload:

    - byte-identical to the first result of that call: a marker naming it;
    - changed, and `diff` is on: a unified diff against the first result, when
      that diff is smaller than half the new payload. This is what a coding
      harness gets from `git diff`, and it is the whole point: a one-line edit
      to a 30k-token file should not cost 30k tokens again;
    - otherwise: delivered in full.

    The reference is always the *first* result of that call, which always
    keeps its full content, so the marker and the diff both point at a message
    the model can still read. A result whose originating call cannot be
    resolved is left untouched, because sameness cannot be proved. Only
    role="tool" messages with string content of at least min_bytes take part.
    Returns how many messages were replaced."""
    messages = req.get("messages")
    if not isinstance(messages, list):
        return 0
    calls = _tool_call_index(messages)
    roots = {}
    replaced = 0
    for position, message in enumerate(messages, 1):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        raw = content.encode()
        if len(raw) < min_bytes:
            continue
        signature = calls.get(message.get("tool_call_id"))
        if signature is None:
            continue
        name, arguments = signature
        call = f"{name}({arguments[:60]})"
        root = roots.get(signature)
        if root is None:
            roots[signature] = (position, raw)
            continue
        root_position, root_raw = root
        if raw == root_raw:
            message["content"] = DEDUPE_MARKER.format(
                call=call, position=root_position, size=len(raw))
            replaced += 1
            continue
        if not diff:
            continue
        if len(raw) > DIFF_MAX_CHARS or len(root_raw) > DIFF_MAX_CHARS:
            continue
        patch = _unified_diff(root_raw.decode("utf-8", "replace"), content)
        if not patch:
            continue
        blob = DIFF_HEADER.format(call=call, position=root_position,
                                  old=len(root_raw), new=len(raw)) + patch
        if len(blob.encode()) > DIFF_MAX_SHARE * len(raw):
            continue
        message["content"] = blob
        replaced += 1
    return replaced


class CacheKeepalive(threading.Thread):
    """Re-touch APC recency during idle gaps (arXiv 2607.19214): when a
    client has been silent past the interval AND the proxy saw zero traffic
    in that window, replay its first system+user prefix with max_tokens=1.
    ponytail: one dict guarded only by the GIL; a lost update just means a
    ping fires one tick late."""

    def __init__(self, server, interval):
        super().__init__(daemon=True)
        self.server = server
        self.interval = interval
        self.clients = {}        # client ip -> last prefix + touch time
        self.last_traffic = 0.0

    def note_traffic(self):
        self.last_traffic = time.monotonic()

    def note_prefix(self, client, model, messages):
        prefix = [m for m in messages
                  if isinstance(m, dict) and m.get("role") == "system"][:1]
        prefix += [m for m in messages
                   if isinstance(m, dict) and m.get("role") == "user"][:1]
        if prefix:
            self.clients[client] = {"t": time.monotonic(), "model": model,
                                    "messages": prefix}

    def run(self):
        while True:
            time.sleep(max(0.05, min(self.interval / 4, 60)))
            now = time.monotonic()
            if now - self.last_traffic < self.interval:
                continue  # mid-request flood: don't add to it
            for c in list(self.clients.values()):
                if now - c["t"] < self.interval or not c["model"]:
                    continue
                body = json.dumps({"model": c["model"],
                                   "messages": c["messages"],
                                   "max_tokens": 1}).encode()
                try:
                    post_json(self.server.upstream, "/v1/chat/completions", body)
                except Exception:
                    pass  # upstream down: retry next interval
                c["t"] = now


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_PUT(self):
        self._handle()

    def do_PATCH(self):
        self._handle()

    def do_DELETE(self):
        self._handle()

    def _handle(self):
        if self.server.keepalive is not None:
            self.server.keepalive.note_traffic()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        try:
            if self.command == "POST" and self.path == "/v1/score":
                self._score(body)
            elif self.command == "POST" and self.path == "/v1/chat/completions":
                self._chat(body)
            else:
                self._passthrough(body)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self._send_json(400, {"error": str(e)})

    def _passthrough(self, body, guard=None, usage=None):
        conn = http.client.HTTPConnection(*self.server.upstream, timeout=600)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP | {"host", "content-length"}}
            conn.request(self.command, self.path, body=body, headers=headers)
            self._relay(conn.getresponse(), guard, usage)
        except (OSError, http.client.HTTPException):
            self._send_json(502, {"error": "upstream unreachable"})
        finally:
            conn.close()  # on a guard cut this aborts upstream generation

    def _relay(self, resp, guard=None, usage=None):
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.end_headers()
        if resp.getheader("Content-Length") is None:
            # streamed (e.g. SSE): nothing to frame against, delimit by close
            self.close_connection = True
        while chunk := resp.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()
            if usage is not None:
                usage.feed(chunk)
            if guard is not None and guard.feed(chunk):
                self.wfile.write(guard.final_chunk())
                self.wfile.flush()
                break
        if usage is not None:
            usage.log()

    def _score(self, body):
        req = json.loads(body)
        model, question = req["model"], req["question"]
        options = [{"label": o} if isinstance(o, str)
                   else {"label": o["label"], "use_when": o.get("use_when")}
                   for o in req["options"]]
        if not options:
            raise ValueError("options must be a non-empty list")
        labels = [o["label"] for o in options]
        probs = score_order(self.server.upstream, model,
                            build_prefix(question, options), labels)
        averaged = False
        if req.get("average_orders", True) and len(options) > 1:
            # JEV position-bias mitigation: rescore with the list reversed
            # and average the two distributions.
            rev = score_order(self.server.upstream, model,
                              build_prefix(question, list(reversed(options))), labels)
            probs = [(a + b) / 2 for a, b in zip(probs, rev)]
            averaged = True
        ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
        self._send_json(200, {
            "ranked": [{"option": label, "prob": p} for label, p in ranked],
            "orders_averaged": averaged,
        })

    def _chat(self, body):
        req = json.loads(body)
        if self.server.reasoning_effort:
            # vLLM has no server-side default for this and ignores a top-level
            # reasoning_effort (only the harmony parser reads one), so the chat
            # template sees it only through chat_template_kwargs. A caller that
            # already set it wins.
            kwargs = dict(req.get("chat_template_kwargs") or {})
            if "reasoning_effort" not in kwargs:
                kwargs["reasoning_effort"] = self.server.reasoning_effort
                req["chat_template_kwargs"] = kwargs
                body = json.dumps(req).encode()
                if not getattr(self.server, "_re_logged", False):
                    self.server._re_logged = True
                    print(f"score proxy: reasoning_effort="
                          f"{self.server.reasoning_effort} on requests",
                          file=sys.stderr)
        if (self.server.canonicalize
                and isinstance(req.get("messages"), list)):
            messages, changed = canonicalize_messages(
                req["messages"], self.server.strip_jitter)
            if changed:
                print(f"score proxy: canonicalized chat messages "
                      f"({len(messages)} messages)", file=sys.stderr)
                req["messages"] = messages
                body = json.dumps(req).encode()
        skills = self.server.skill_index
        if skills is not None:
            changed, sim, count = skills.apply(req)
            # Injected before tool steering so tool necessity/pruning see the
            # final messages, and before keepalive snapshots the prefix that
            # is actually sent upstream.
            if changed:
                body = json.dumps(req).encode()
                print(f"score proxy: skills: injected {count} "
                      f"(top sim {sim:.3f})", file=sys.stderr)
        if self.server.keepalive is not None:
            self.server.keepalive.note_prefix(
                self.client_address[0], req.get("model"),
                req.get("messages") or [])
        steer = self.server.tool_router
        if steer is not None and req.get("tools"):
            changed, prob = steer.apply(req, req.get("model"))
            if changed:
                body = json.dumps(req).encode()
            if prob >= steer.threshold:
                print(f"score proxy: tool steer prob={prob:.3f} "
                      f"force={steer.force}", file=sys.stderr)
        if self.server.dedupe_tool_output or self.server.diff_tool_output:
            # Off by default: this changes what the model reads, so it runs
            # only when an operator opted in. Positions in the marker are
            # final because every message insertion has already happened.
            collapsed = collapse_duplicate_tool_output(
                req, diff=self.server.diff_tool_output)
            if collapsed:
                body = json.dumps(req).encode()
                print(f"score proxy: collapsed {collapsed} duplicate tool "
                      f"result(s)", file=sys.stderr)
        router = self.server.router
        if router is not None and req.get("route") is not False:
            text = last_user_text(req.get("messages") or [])
            route, margin = router.classify(text) if text else (None, 0.0)
            if route == "trivial":
                # pre-hoc no-thinking: also skips the escalation pass below
                kwargs = dict(req.get("chat_template_kwargs") or {})
                kwargs["enable_thinking"] = False
                req["chat_template_kwargs"] = kwargs
                req["max_tokens"] = min(req.get("max_tokens")
                                        or router.max_tokens,
                                        router.max_tokens)
                clamp_max_tokens(req, self.server.context_tokens)
                body = json.dumps(req).encode()
                print(f"score proxy: routed trivial "
                      f"(margin {margin:.3f})", file=sys.stderr)
                if req.get("stream"):
                    guard = (LoopGuard(self.server.loop_repeat)
                             if self.server.loop_guard else None)
                    return self._passthrough(body, guard, StreamUsage())
                status, resp = post_json(self.server.upstream,
                                         "/v1/chat/completions", body)
                if isinstance(resp, dict):
                    if status == 200:
                        log_usage(resp.get("usage"))
                    repair_reply(resp, req, self.server.context_tokens,
                                 allow_retry=False)
                    resp["x_proxy"] = {"route": "trivial",
                                       "margin": round(margin, 3)}
                return self._send_json(status, resp)
        if clamp_max_tokens(req, self.server.context_tokens):
            body = json.dumps(req).encode()
            print(f"score proxy: clamped max_tokens to {req['max_tokens']} "
                  f"for a {self.server.context_tokens}-token context",
                  file=sys.stderr)
        escalate = self.server.escalate_env or req.get("escalate") is True
        if req.get("stream"):
            # escalation needs the full response; streams pass through with
            # only the loop guard watching the deltas
            guard = (LoopGuard(self.server.loop_repeat)
                     if self.server.loop_guard else None)
            return self._passthrough(body, guard, StreamUsage())
        if not escalate:
            return self._chat_result(req, body)
        # pass 1: thinking off, logprobs on
        p1 = dict(req)
        kwargs = dict(req.get("chat_template_kwargs") or {})
        kwargs["enable_thinking"] = False
        p1["chat_template_kwargs"] = kwargs
        p1["logprobs"] = True
        p1["stream"] = False
        status, resp = post_json(self.server.upstream, "/v1/chat/completions",
                                 json.dumps(p1).encode())
        if status != 200:
            return self._send_json(status, resp)
        log_usage(resp.get("usage") if isinstance(resp, dict) else None)
        tokens = ((resp.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        vals = [t["logprob"] for t in tokens if t.get("logprob") is not None]
        mean = sum(vals) / len(vals) if vals else 0.0
        if mean >= self.server.threshold:
            repair_reply(resp, req, self.server.context_tokens,
                         allow_retry=False)
            resp["x_proxy"] = {"escalated": False, "mean_logprob": mean}
            return self._send_json(status, resp)
        # low confidence: re-issue the original request unchanged (thinking on)
        status, headers, raw, resp, _ = self._complete(req, body)
        if resp is None:
            return self._send_raw(status, headers, raw)
        resp.setdefault("x_proxy", {})["escalated"] = True
        resp["x_proxy"]["mean_logprob"] = mean
        self._send_json(status, resp)

    def _complete(self, req, body):
        """Buffered call with repair. The retry is issued at most once here;
        a failed continuation leaves the original response untouched."""
        client_headers = {
            key: value for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP | {"host", "content-length"}}
        status, headers, raw = post_buffered(
            self.server.upstream, "/v1/chat/completions", body,
            headers=client_headers)
        if status != 200:
            return status, headers, raw, None, False
        try:
            resp = json.loads(raw)
        except ValueError:
            return status, headers, raw, None, False
        log_usage(resp.get("usage") if isinstance(resp, dict) else None)
        changed, retry = repair_reply(resp, req, self.server.context_tokens)
        if retry is not None:
            rstatus, _, rraw = post_buffered(
                self.server.upstream, "/v1/chat/completions",
                json.dumps(retry["request"]).encode(), headers=client_headers)
            retry_resp = None
            if rstatus == 200:
                try:
                    retry_resp = json.loads(rraw)
                except ValueError:
                    pass
            if isinstance(retry_resp, dict):
                log_usage(retry_resp.get("usage"))
            merged = (retry_resp is not None
                      and apply_thinking_retry(resp, retry_resp, retry))
            if merged:
                changed = True
                changed = repair_reply(
                    resp, req, self.server.context_tokens,
                    allow_retry=False)[0] or changed
            print(f"score proxy: thinking retry "
                  f"{'closed' if merged else 'did not recover'}", file=sys.stderr)
        return status, headers, raw, resp, changed

    def _chat_result(self, req, body):
        status, headers, raw, resp, changed = self._complete(req, body)
        if resp is None or not changed:
            return self._send_raw(status, headers, raw)
        return self._send_json(status, resp)

    def _send_raw(self, status, headers, raw):
        self.send_response(status)
        for key, value in headers:
            if key.lower() not in HOP_BY_HOP | {"content-length"}:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(port, upstream_url, escalate_env=False, threshold=-0.35,
                loop_guard=True, loop_repeat=3, canonicalize=False,
                strip_jitter=False, keepalive_s=0, router_url=None,
                router_model="google/embeddinggemma-2",
                router_threshold=0.10, router_max_tokens=1024,
                router_timeout=2.0, tool_steer=False, tool_threshold=0.5,
                tool_force=False, tool_top_k=8, tool_always=(),
                skills_dir=None, skills_top_k=3, skills_min_sim=0.25,
                skills_pin=(), skills_pin_chars=4000, context_tokens=0,
                dedupe_tool_output=False, diff_tool_output=False,
                reasoning_effort=None):
    if reasoning_effort not in (None, "", "xhigh", "medium", "low"):
        raise SystemExit(
            f"PROXY_REASONING_EFFORT={reasoning_effort!r}: use low, medium or xhigh")
    u = urllib.parse.urlparse(upstream_url)
    srv = ThreadingHTTPServer(("", port), Handler)
    srv.upstream = (u.hostname, u.port or 80)
    srv.context_tokens = context_tokens
    srv.dedupe_tool_output = dedupe_tool_output
    srv.diff_tool_output = diff_tool_output
    srv.escalate_env = escalate_env
    srv.threshold = threshold
    srv.loop_guard = loop_guard
    srv.loop_repeat = loop_repeat
    srv.canonicalize = canonicalize
    srv.strip_jitter = strip_jitter
    srv.router = (Router(router_url, router_model, router_threshold,
                         router_max_tokens, router_timeout)
                  if router_url else None)
    srv.tool_router = None
    if tool_steer:
        srv.tool_router = ToolRouter(
            lambda model, prefix, labels: score_order(
                srv.upstream, model, prefix, labels),
            embed_fn=srv.router.embed if srv.router else None,
            threshold=tool_threshold, force=tool_force,
            top_k=tool_top_k, always=tool_always)
    srv.skill_index = None
    # A pin needs no embeddings — it reads named files directly — so it must
    # not be gated on the embedding service being reachable.
    if skills_dir and (srv.router is not None or skills_pin):
        srv.skill_index = SkillIndex(
            skills_dir, srv.router.embed if srv.router else None,
            top_k=skills_top_k,
            min_sim=skills_min_sim, pin=skills_pin,
            pin_chars=skills_pin_chars)
    srv.reasoning_effort = reasoning_effort or None
    srv.keepalive = None
    if keepalive_s > 0:
        srv.keepalive = CacheKeepalive(srv, keepalive_s)
        srv.keepalive.start()
    return srv


def main():
    srv = make_server(
        int(os.environ.get("PROXY_PORT", 8889)),
        os.environ.get("PROXY_UPSTREAM", "http://localhost:8888"),
        os.environ.get("PROXY_ESCALATE") == "1",
        # ponytail: -0.35 is a heuristic default — calibrate per workload
        # on real traffic before trusting the escalation gate.
        float(os.environ.get("PROXY_ESCALATE_THRESHOLD", -0.35)),
        os.environ.get("PROXY_LOOP_GUARD", "1") != "0",
        int(os.environ.get("PROXY_LOOP_GUARD_REPEAT", 3)),
        os.environ.get("PROXY_CANONICALIZE") == "1",
        os.environ.get("PROXY_CANONICALIZE_STRIP_JITTER") == "1",
        float(os.environ.get("PROXY_CACHE_KEEPALIVE_S", 0) or 0),
        os.environ.get("PROXY_ROUTER_URL") or None,
        os.environ.get("PROXY_ROUTER_MODEL", "google/embeddinggemma-2"),
        float(os.environ.get("PROXY_ROUTER_THRESHOLD", 0.10)),
        int(os.environ.get("PROXY_ROUTER_TRIVIAL_MAX_TOKENS", 1024)),
        float(os.environ.get("PROXY_ROUTER_TIMEOUT_S", 2)),
        os.environ.get("PROXY_TOOL_ROUTER") == "1",
        float(os.environ.get("PROXY_TOOL_THRESHOLD", 0.5)),
        os.environ.get("PROXY_TOOL_FORCE") == "1",
        int(os.environ.get("PROXY_TOOL_TOP_K", 8)),
        tuple(n for n in os.environ.get("PROXY_TOOL_ALWAYS", "").split(",")
              if n.strip()),
        os.environ.get("PROXY_SKILLS_DIR") or None,
        int(os.environ.get("PROXY_SKILLS_TOP_K", 3)),
        float(os.environ.get("PROXY_SKILLS_MIN_SIM", 0.25)),
        tuple(n for n in os.environ.get("PROXY_SKILLS_PIN", "").split(",")
              if n.strip()),
        int(os.environ.get("PROXY_SKILLS_PIN_CHARS", 4000)),
        int(os.environ.get("PROXY_CONTEXT_TOKENS", 0) or 0),
        os.environ.get("PROXY_DEDUPE_TOOL_OUTPUT") == "1",
        os.environ.get("PROXY_DIFF_TOOL_OUTPUT") == "1",
        reasoning_effort=os.environ.get("PROXY_REASONING_EFFORT") or None)
    print(f"score proxy on :{srv.server_port} -> "
          f"http://{srv.upstream[0]}:{srv.upstream[1]}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
