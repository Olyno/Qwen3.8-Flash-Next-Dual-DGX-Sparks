#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# Regression-gate matrix runner — the executable form of docs/bench-gates.md.
#
#   bench/gate_report.sh --record            # measure, store as perf baseline
#   bench/gate_report.sh                     # measure, gate against stored baseline
#   bench/gate_report.sh --candidate URL     # also run the p99-KL quality gate
#
# Suite: decodebench (prefill TTFT + decode, per context bucket, with the
# mtp_accept counters around it) and kl_yardstick (only when --candidate is
# given). Verdicts: prefill regression > ${GATE_PREFILL_PCT:-1}% FAIL, decode
# (ms/step) regression > ${GATE_DECODE_PCT:-5}% FAIL, KL gate per
# kl_yardstick's own thresholds. Exit 0 all pass, 1 any FAIL, 2 host offline
# (nothing is measured against a dead or busy server — SKIP is not PASS).
set -u

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
URL="${BENCH_BASE:-http://localhost:8888}"
CANDIDATE=""
CONTEXTS="1000,32000,65000"
DECODE=600
MODEL="${BENCH_MODEL:-Qwen3.8-Flash-Next-NVFP4}"
BASELINE="$(dirname "$BENCH")/bench/data/gate_perf.json"
RECORD=0
PREFILL_PCT="${GATE_PREFILL_PCT:-1.0}"
DECODE_PCT="${GATE_DECODE_PCT:-5.0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --contexts) CONTEXTS="$2"; shift 2 ;;
    --decode) DECODE="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --record) RECORD=1; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 64 ;;
  esac
done

ROWS=""
add() { ROWS="${ROWS}${1}|${2}|${3}|${4}|${5}|${6}\n"; }
any=0

health() {
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "$URL/health" 2>/dev/null)
  [ "$code" = "200" ]
}

if ! health; then
  for b in ${CONTEXTS//,/ }; do
    add "prefill" "$b" "-" "-" "-" "SKIP"
    add "decode"  "$b" "-" "-" "-" "SKIP"
  done
  add "kl-quality" "-" "-" "-" "-" "SKIP"
  printf '\nGATE MATRIX  url=%s  (host not serving /health — nothing measured)\n\n' "$URL"
  printf '%-14s %-8s %12s %12s %9s  %s\n' GATE BUCKET VALUE BASELINE DELTA VERDICT
  printf '%b' "$ROWS" | awk -F'|' '{printf "%-14s %-8s %12s %12s %10s  %s\n",$1,$2,$3,$4,$5,$6}'
  echo; echo "RESULT: OFFLINE — start the server (or point --url at one), then re-run."
  echo "Conventions: docs/bench-gates.md. A SKIP never satisfies a gate."
  exit 2
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
$PY "$BENCH/mtp_accept.py" --url "$URL/metrics" --save "$TMP/metrics_before.json" 2>/dev/null \
  || echo '{"note":"no /metrics"}' > "$TMP/metrics_before.json"
echo "== decodebench (contexts=$CONTEXTS decode=$DECODE temp=0) =="
BENCH_BASE="$URL" BENCH_MODEL="$MODEL" $PY "$BENCH/decodebench.py" \
  --contexts "$CONTEXTS" --temps 0.0 --decode "$DECODE" \
  --jsonl "$TMP/dec.jsonl" || exit 1
echo "== mtp_accept (delta across the decodebench run) =="
$PY "$BENCH/mtp_accept.py" --url "$URL/metrics" --since "$TMP/metrics_before.json" || true

# one perf row per context bucket; aggregate across tasks
$PY - "$TMP/dec.jsonl" "$BASELINE" "$URL" "$RECORD" "$PREFILL_PCT" "$DECODE_PCT" <<'PY' > "$TMP/perf.out"
import json, math, os, sys, time
rows_path, base_path, url, record, pref_pct, dec_pct = sys.argv[1:7]
rows = [json.loads(l) for l in open(rows_path)]
buckets = {}
for r in rows:
    b = buckets.setdefault(r["context"], {"ttft": [], "tps": [], "ms_step": [], "ptok": []})
    b["ttft"].append(r["ttft_s"]); b["tps"].append(r["decode_tps"])
    b["ptok"].append(r["ptok"])
    sps = r.get("steps_ps")
    if sps and sps == sps and sps > 0:            # counters present: ms/step
        b["ms_step"].append(1000.0 / sps)
    elif r["decode_tps"] == r["decode_tps"] and r["decode_tps"] > 0:
        b["ms_step"].append(1000.0 / r["decode_tps"])   # 1 step = 1 token
mean = lambda xs: sum(xs) / len(xs)
cur = {}
for c, b in buckets.items():
    ptok = mean(b["ptok"])
    cur[f"{c}"] = {"prefill_ms_per_1k": mean(b["ttft"]) / max(ptok, 1) * 1000.0,
                   "decode_ms_per_step": mean(b["ms_step"]),
                   "decode_tps": mean(b["tps"])}
base = {}
if os.path.isfile(base_path):
    base = json.load(open(base_path))
if record == "1":
    base.update({f"{url}|{k}": v for k, v in cur.items()})
    base.update({"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": url})
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    json.dump(base, open(base_path, "w"), indent=1, sort_keys=True)
for c in sorted(cur, key=int):
    p, d = cur[c]["prefill_ms_per_1k"], cur[c]["decode_ms_per_step"]
    bp, bd = base.get(f"{url}|{c}", {}).get("prefill_ms_per_1k"), \
             base.get(f"{url}|{c}", {}).get("decode_ms_per_step")
    for name, val, bval, thr in (("prefill", p, bp, float(pref_pct)),
                                ("decode", d, bd, float(dec_pct))):
        if record == "1":
            print(f"{name}|{c}|{val:.3f}|-|-|RECORDED")
        elif bval is None:
            print(f"{name}|{c}|{val:.3f}|-|-|SKIP")
        else:
            reg = 100.0 * (val - bval) / bval
            v = "PASS" if reg <= thr else "FAIL"
            print(f"{name}|{c}|{val:.3f}|{bval:.3f}|{reg:+.2f}%|{v}")
PY
while IFS= read -r line; do add $(echo "$line" | tr '|' ' '); done < "$TMP/perf.out"

KL_RC=0
if [ -n "$CANDIDATE" ]; then
  echo "== kl_yardstick (baseline $URL vs candidate $CANDIDATE) =="
  $PY "$BENCH/kl_yardstick.py" compare --url "$URL" --candidate "$CANDIDATE" \
    --model "$MODEL" || KL_RC=$?
  if [ "$KL_RC" = "0" ]; then add "kl-quality" "-" "-" "-" "-" "PASS"
  else add "kl-quality" "-" "-" "-" "-" "FAIL"; fi
else
  add "kl-quality" "-" "-" "-" "-" "SKIP (no --candidate)"
fi

printf '\nGATE MATRIX  url=%s  candidate=%s  %s\n\n' "$URL" "${CANDIDATE:-none}" \
  "$(date '+%Y-%m-%d %H:%M')"
printf '%-14s %-8s %12s %12s %10s  %s\n' GATE BUCKET VALUE BASELINE DELTA VERDICT
printf '%b' "$ROWS" | awk -F'|' '{printf "%-14s %-8s %12s %12s %10s  %s\n",$1,$2,$3,$4,$5,$6}'
echo
if printf '%b' "$ROWS" | grep -q '|FAIL$'; then
  echo "RESULT: FAIL — a measured configuration got slower (docs/bench-gates.md)."
  exit 1
fi
if printf '%b' "$ROWS" | grep -q '|SKIP'; then
  echo "RESULT: PASS (some gates skipped — see matrix; a SKIP is not a pass)."
  exit 0
fi
echo "RESULT: PASS — no measured configuration regressed beyond its threshold."
exit 0
