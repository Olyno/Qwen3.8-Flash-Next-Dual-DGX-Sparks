#!/usr/bin/env python3
"""Engine-side request trace: what the server spent, per request.

The harness times a task from outside (model call vs tool call). This reads the
engine's own Prometheus counters and decomposes a window into queue, prefill,
decode, time-to-first-token and inter-token latency. Together they answer the
only question that matters for optimisation: is the wall-clock in the engine,
in the tools, or waiting?

    python3 bench/engine_trace.py snapshot /tmp/m0.json
    # ... run a benchmark ...
    python3 bench/engine_trace.py snapshot /tmp/m1.json
    python3 bench/engine_trace.py report /tmp/m0.json /tmp/m1.json

Counters are process-lifetime, so the window is the difference between the two
snapshots. A snapshot taken right after boot is a valid zero point.

    python3 bench/engine_trace.py --selftest
"""
import json
import math
import sys
import urllib.request

FAMILIES = (
    "time_to_first_token_seconds",
    "inter_token_latency_seconds",
    "e2e_request_latency_seconds",
    "request_queue_time_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
)
GAUGES = ("num_requests_running", "num_requests_waiting")


def fetch(url, timeout=10):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics",
                                timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def parse(text):
    """Keep sum/count for each histogram family and the two gauges."""
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, raw = line.rpartition(" ")
        try:
            value = float(raw)
        except ValueError:
            continue
        base = name.split("{", 1)[0]
        short = base[len("vllm:"):] if base.startswith("vllm:") else base
        for family in FAMILIES:
            if short in (family + "_sum", family + "_count"):
                values[short] = values.get(short, 0.0) + value
                break
        else:
            if short in GAUGES:
                values[short] = value
    return values


def snapshot(url, path):
    payload = {"when": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
               "url": url, "values": parse(fetch(url))}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"wrote {path}: {len(payload['values'])} counters")
    return 0


def window(before, after):
    """Per-family (n, mean) for the window between two snapshots."""
    rows = {}
    for family in FAMILIES:
        count = after.get(family + "_count", 0.0) - before.get(family + "_count", 0.0)
        total = after.get(family + "_sum", 0.0) - before.get(family + "_sum", 0.0)
        rows[family] = (int(round(count)), total / count if count > 0 else None)
    return rows


def report(before_path, after_path):
    with open(before_path, encoding="utf-8") as handle:
        before = json.load(handle)["values"]
    with open(after_path, encoding="utf-8") as handle:
        after = json.load(handle)["values"]
    rows = window(before, after)
    requests = rows["e2e_request_latency_seconds"][0]
    print(f"requests in window: {requests}")
    if not requests:
        print("nothing to report")
        return 0
    labels = {
        "request_queue_time_seconds": "queue",
        "request_prefill_time_seconds": "prefill",
        "request_decode_time_seconds": "decode",
        "time_to_first_token_seconds": "ttft",
        "inter_token_latency_seconds": "inter-token",
        "e2e_request_latency_seconds": "end to end",
    }
    for family in ("request_queue_time_seconds",
                   "request_prefill_time_seconds",
                   "request_decode_time_seconds",
                   "time_to_first_token_seconds",
                   "inter_token_latency_seconds",
                   "e2e_request_latency_seconds"):
        count, mean = rows[family]
        share = ""
        e2e = rows["e2e_request_latency_seconds"][1]
        if mean is not None and e2e:
            share = f"  {100.0 * mean / e2e:5.1f}% of e2e"
        shown = "n/a" if mean is None else f"{mean * 1000:8.1f} ms"
        print(f"  {labels[family]:<12} n={count:<5} {shown}{share}")
    prefill = rows["request_prefill_time_seconds"][1] or 0.0
    decode = rows["request_decode_time_seconds"][1] or 0.0
    if prefill + decode:
        print(f"  prefill share of engine time: "
              f"{100.0 * prefill / (prefill + decode):.0f}%")
    return 0


def selftest():
    before = {"e2e_request_latency_seconds_count": 0.0,
              "e2e_request_latency_seconds_sum": 0.0,
              "request_prefill_time_seconds_count": 0.0,
              "request_prefill_time_seconds_sum": 0.0,
              "request_decode_time_seconds_count": 0.0,
              "request_decode_time_seconds_sum": 0.0,
              "request_queue_time_seconds_count": 0.0,
              "request_queue_time_seconds_sum": 0.0,
              "time_to_first_token_seconds_count": 0.0,
              "time_to_first_token_seconds_sum": 0.0,
              "inter_token_latency_seconds_count": 0.0,
              "inter_token_latency_seconds_sum": 0.0}
    after = dict(before)
    after.update({
        "e2e_request_latency_seconds_count": 2.0,
        "e2e_request_latency_seconds_sum": 10.0,
        "request_prefill_time_seconds_count": 2.0,
        "request_prefill_time_seconds_sum": 2.0,
        "request_decode_time_seconds_count": 2.0,
        "request_decode_time_seconds_sum": 6.0,
        "request_queue_time_seconds_count": 2.0,
        "request_queue_time_seconds_sum": 0.2,
        "time_to_first_token_seconds_count": 2.0,
        "time_to_first_token_seconds_sum": 1.2,
        "inter_token_latency_seconds_count": 40.0,
        "inter_token_latency_seconds_sum": 2.0,
    })
    rows = window(before, after)
    assert rows["e2e_request_latency_seconds"] == (2, 5.0), rows
    assert rows["request_prefill_time_seconds"] == (2, 1.0), rows
    assert rows["request_decode_time_seconds"] == (2, 3.0), rows
    assert rows["inter_token_latency_seconds"] == (40, 0.05), rows
    assert math.isclose(rows["request_queue_time_seconds"][1], 0.1)
    text = ("# HELP vllm:time_to_first_token_seconds\n"
            'vllm:time_to_first_token_seconds_count{engine="0"} 3.0\n'
            'vllm:time_to_first_token_seconds_sum{engine="0"} 1.5\n'
            'vllm:num_requests_running{engine="0"} 1.0\n')
    parsed = parse(text)
    assert parsed["time_to_first_token_seconds_count"] == 3.0, parsed
    assert parsed["time_to_first_token_seconds_sum"] == 1.5, parsed
    assert parsed["num_requests_running"] == 1.0, parsed
    print("SELFTEST PASS")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) == 4 and argv[1] == "snapshot":
        return snapshot(argv[2], argv[3])
    if len(argv) == 4 and argv[1] == "report":
        return report(argv[2], argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
