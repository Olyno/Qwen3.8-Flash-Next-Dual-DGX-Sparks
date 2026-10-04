#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""bench/option_scoring.py — score N options in one batched prefill instead
of generating. See docs/decision-scoring.md.

One /v1/completions call with prompt = [prefix + opt for each option],
echo + logprobs: sum each option's token logprobs (prefix caching makes the
shared prefix cost one prefill), softmax, done. Demo routes a support
message into intents; --prompt/--options for your own.
"""
import argparse
import json
import math
import sys
import urllib.request

DEMO_PROMPT = (
    "Classify the customer message into exactly one intent.\n"
    "Message: My subscription renewed after the service was already down. "
    "Can I get that charge refunded?\n"
    "Intent: "
)
DEMO_OPTIONS = [
    "order_status", "refund_request", "cancel_subscription", "update_payment",
    "login_problem", "shipping_delay", "bug_report", "speak_to_human",
]
DEMO_EXPECT = "refund_request"


def score(host, model, prefix, options):
    body = json.dumps({
        "model": model,
        "prompt": [prefix + o for o in options],
        "echo": True,
        "logprobs": 0,
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req))
    scores = []
    for choice in sorted(resp["choices"], key=lambda c: c["index"]):
        lp = choice["logprobs"]
        # text_offset[i] is the char offset of token i in the echoed text;
        # the option is everything past the shared prefix.
        scores.append(sum(t for t, off in zip(lp["token_logprobs"], lp["text_offset"])
                          if off >= len(prefix) and t is not None))
    top = max(scores)
    probs = [math.exp(s - top) for s in scores]
    total = sum(probs)
    return sorted(zip(options, (p / total for p in probs)), key=lambda x: -x[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="localhost:8888")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--prompt", default=DEMO_PROMPT)
    ap.add_argument("--options", default=",".join(DEMO_OPTIONS))
    args = ap.parse_args()
    options = [o.strip() for o in args.options.split(",") if o.strip()]
    ranked = score(args.host, args.model, args.prompt, options)
    for opt, p in ranked:
        print(f"{p:6.1%}  {opt}")
    if args.prompt == DEMO_PROMPT and options == DEMO_OPTIONS:
        assert ranked[0][0] == DEMO_EXPECT, f"expected {DEMO_EXPECT}, got {ranked[0][0]}"
        print(f"self-check OK: routed to {DEMO_EXPECT}")


if __name__ == "__main__":
    sys.exit(main())
