#!/usr/bin/env python3
"""Render check for chat_template_cod.jinja (run: python3 tests/test_chat_template_cod.py).

The CoD prior replaces the xhigh reasoning blurb, so it must land in the
system block on the default path — with and without a caller system message,
and in front of the tools preamble — and stay out of the low-effort path.
"""
import pathlib
import sys

import jinja2
import jinja2.sandbox

TPL = pathlib.Path(__file__).resolve().parent.parent / "chat_template_cod.jinja"
COD = "minimum draft for each thinking step"
XHIGH = "Reasoning effort is set to xhigh"


def render(messages, **kwargs):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise ValueError(msg)

    return env.from_string(TPL.read_text()).render(
        messages=messages, raise_exception=raise_exception, **kwargs
    )


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return cond


ok = True
user = [{"role": "user", "content": "2+2?"}]

out = render(user)
ok &= check("default path injects CoD system block",
            "<|im_start|>system\n" in out and COD in out)
ok &= check("default path drops the xhigh blurb", XHIGH not in out)

out = render([{"role": "system", "content": "Be terse."}] + user)
ok &= check("caller system message kept, CoD prepended",
            COD in out and "Be terse." in out and out.index(COD) < out.index("Be terse."))

out = render(user, reasoning_effort="low")
ok &= check("low effort untouched (no CoD)",
            "Reasoning effort is set to low" in out and COD not in out)

tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
out = render(user, tools=tools)
ok &= check("tools path: CoD precedes the tools preamble",
            COD in out and out.index(COD) < out.index("# Tools"))

sys.exit(0 if ok else 1)
