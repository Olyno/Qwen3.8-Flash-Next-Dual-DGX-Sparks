#!/usr/bin/env python3
"""concbench.py — concurrency ladder for one vLLM OpenAI endpoint.

Finds the cliff: per-stream decode speed and TTFT as C grows.
Protocol mirrors decodebench (same prompt families, temp 0.6), so C=1 rows
are comparable with the banked single-stream table; C>1 rows are new.

Usage: concbench.py --port 8892 [--levels 1,4,8,16,32] [--decode 600]
"""
import argparse, asyncio, json, statistics, sys, time

import aiohttp

LEVELS = (1, 4, 8, 16, 32)


def prompt(i):
    # vary prompts slightly so prefix cache doesn't hand out free wins
    seeds = ["hydrology", "glaciology", "sedimentology", "geomorphology",
             "tidal dynamics", "turbidity", "avulsion", "meander migration"]
    return (f"Explain in careful detail how {seeds[i % len(seeds)]} shapes "
            f"coastal sediment transport, reasoning step by step.")


async def stream_one(sess, port, model, max_tokens, temp, i, min_tokens=0):
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt(i)}],
        "max_tokens": max_tokens, "temperature": temp, "stream": True,
        "stop": [],
    }
    if min_tokens:
        # vLLM extra sampling key: keep generating past EOS so every stream
        # fills its budget -> measures scheduler behavior, not early stops
        body["min_tokens"] = min_tokens
    try:
        async with sess.post(f"http://localhost:{port}/v1/chat/completions",
                             json=body) as r:
            if r.status != 200:
                return {"err": r.status}
            async for chunk in r.content:
                for line in chunk.split(b"\n"):
                    if not line.startswith(b"data: "):
                        continue
                    p = line[6:].strip()
                    if p == b"[DONE]":
                        continue
                    try:
                        d = json.loads(p)
                    except Exception:
                        continue
                    dl = d.get("choices", [{}])[0].get("delta", {})
                    # qwen3 reasoning-parser: thinking tokens arrive under the
                    # delta key "reasoning" (VERIFIED against the live stream,
                    # 10-01: chunk keys are role/content then `reasoning`). The
                    # old test looked for content/reasoning_content — keys that
                    # never appear inside a short budget — so EVERY concurrent
                    # level counted zero tokens (the A1 all-fail and the R7/r1b
                    # zero-rows were this counter, not the engine). Count any
                    # delta that carries a text payload under any of the three
                    # names; role-only/empty chunks don't qualify.
                    if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        ntok += 1
        dt = time.perf_counter() - t0
        return {"ttft": ttft, "tok": ntok, "secs": dt,
                "tps": ntok / (dt - (ttft or 0)) if dt > (ttft or 0) + 0.5 else 0}
    except Exception as e:  # noqa: BLE001
        return {"err": repr(e)[:80]}


async def level(port, model, c, max_tokens, temp, warm, min_tokens=0):
    conn = aiohttp.TCPConnector(limit=c + 4)
    async with aiohttp.ClientSession(connector=conn) as sess:
        rs = await asyncio.gather(*[
            stream_one(sess, port, model, max_tokens, temp, warm + i, min_tokens)
            for i in range(c)])
    err = sum(1 for r in rs if "err" in r)
    zero = sum(1 for r in rs if "err" not in r and r["tok"] == 0)
    ok = [r for r in rs if "err" not in r and r["tps"] > 0]
    slow = [r for r in rs if "err" not in r and r["tps"] == 0 and r["tok"] > 0]
    if not ok and not zero and not slow:
        return None
    agg = sum(r["tok"] for r in ok) / max((r["secs"] for r in ok), default=1)
    tps_sorted = sorted(r["tps"] for r in ok)
    return {"C": c, "n_ok": len(ok), "n_err": err, "n_zero": zero, "n_slow": slow,
            "per_stream_med": round(statistics.median(tps_sorted), 1) if tps_sorted else 0,
            "per_stream_min": round(tps_sorted[0], 1) if tps_sorted else 0,
            "per_stream_p90": round(tps_sorted[int(len(tps_sorted) * 0.9) - 1], 1) if tps_sorted else 0,
            "aggregate": round(agg, 1),
            "ttft_med": round(statistics.median(r["ttft"] for r in ok), 2) if ok else None,
            "ttft_p95": round(max(r["ttft"] for r in ok), 2) if ok else None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--levels", default="1,4,8,16,32")
    ap.add_argument("--decode", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--min-tokens", type=int, default=0)
    a = ap.parse_args()
    async with aiohttp.ClientSession() as s:
        model = (await (await s.get(f"http://localhost:{a.port}/v1/models")
                        ).json())["data"][0]["id"]
    warm = 0
    for c in [int(x) for x in a.levels.split(",")]:
        for rep in range(a.reps):
            r = await level(a.port, model, c, a.decode, a.temp, warm)
            warm += c
            if r:
                print(json.dumps(r | {"rep": rep, "port": a.port}), flush=True)
            else:
                print(json.dumps({"C": c, "err": "all-failed"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
