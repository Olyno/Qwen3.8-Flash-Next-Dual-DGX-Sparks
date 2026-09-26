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


async def stream_one(sess, port, model, max_tokens, temp, i):
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt(i)}],
        "max_tokens": max_tokens, "temperature": temp, "stream": True,
    }
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
                    if dl.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        ntok += 1
        dt = time.perf_counter() - t0
        return {"ttft": ttft, "tok": ntok, "secs": dt,
                "tps": ntok / (dt - (ttft or 0)) if dt > (ttft or 0) + 0.5 else 0}
    except Exception as e:  # noqa: BLE001
        return {"err": repr(e)[:80]}


async def level(port, model, c, max_tokens, temp, warm):
    conn = aiohttp.TCPConnector(limit=c + 4)
    async with aiohttp.ClientSession(connector=conn) as sess:
        rs = await asyncio.gather(*[
            stream_one(sess, port, model, max_tokens, temp, warm + i)
            for i in range(c)])
    ok = [r for r in rs if "err" not in r and r["tps"] > 0]
    if not ok:
        return None
    agg = sum(r["tok"] for r in ok) / max(r["secs"] for r in ok)
    return {"C": c, "n_ok": len(ok),
            "per_stream_med": round(statistics.median(r["tps"] for r in ok), 1),
            "per_stream_min": round(min(r["tps"] for r in ok), 1),
            "aggregate": round(agg, 1),
            "ttft_med": round(statistics.median(r["ttft"] for r in ok), 2),
            "ttft_p95": round(max(r["ttft"] for r in ok), 2)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--levels", default="1,4,8,16,32")
    ap.add_argument("--decode", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--reps", type=int, default=1)
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
