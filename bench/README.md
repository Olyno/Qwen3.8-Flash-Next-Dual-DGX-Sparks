# bench/ — what to run, in what order

Everything talks to a live server on `:8888` (override with `--port` / `--base`
where the script has the flag). One workload at a time: these measure
bandwidth-bound decode, and a second client pollutes every number.

## CURRENT (consolidated repo, v0.30 protocol)

| script | what it measures | canonical invocation |
|---|---|---|
| `decodebench.py` | single-stream decode, CONTENT TYPE vs CONTEXT LENGTH (ignore_eos, exact token counts) | `python3 bench/decodebench.py --decode 600 --contexts 1000,600000 --temps 0.0` — the 600-tok ctx-sweep protocol every banked table row uses |
| `longctx.py` | prefill TTFT + needle retrieval at a target context | `python3 bench/longctx.py --target 600000 --max-tokens 1024` |
| `mtp_accept.py` | MTP draft acceptance as a *delta* around your workload (never cumulative) | `--save before.json` → run bench → `--since before.json` |
| `reasoning_check.py` | deterministic reasoning/needle quality, machine-graded; the fp8-KV quality gate | bf16 run `--out bf16.json`, then `--out fp8.json --compare bf16.json` |

**Standard arm protocol** (what `docs/verdicts/*` rows are comparable with):
`mtp_accept --save` → `decodebench --decode 600 --contexts 1k,... --temps 0.0`
→ `mtp_accept --since` → `reasoning_check` score → `longctx` for the TTFT
row. Run `concbench` (below) for the concurrency column.

## CURRENT but lives in `experiments/v30/`

| script | what it measures |
|---|---|
| `experiments/v30/concbench.py` | concurrency ladder C=1..32: per-stream decode + TTFT vs load; protocol mirrors decodebench (same prompt families, temp 0.6) so the C=1 row is comparable with the single-stream table. `python3 experiments/v30/concbench.py --port 8888 [--levels 1,4,8,16,32] [--decode 600]`. Needs `aiohttp`. |

It stays in the spike lane until that lane retires; it has no imports from
there and can be run straight from that path.

## LEGACY (kept for reference; do not add new numbers from these)

| script | status |
|---|---|
| `acceptance_sweep.py` | legacy: dflash era, kept for reference. Per-task tau (accepted/draft) sweep that answered the drafter-selection question in `docs/verdicts/DFLASH.md`. Its header references `bench/sweep.py`, which no longer exists. |

## Counter hygiene (all scripts)

Cumulative `/metrics` counters mix every request the server has ever seen.
Snapshot with `mtp_accept.py --save` before the workload, diff with `--since`
after. Never quote a mean acceptance that spans another client's traffic.
