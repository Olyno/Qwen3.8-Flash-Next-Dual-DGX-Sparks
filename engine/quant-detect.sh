#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# engine/quant-detect.sh — ported from the single-Spark monolith start.sh:760-851.
# Refuse to serve a checkpoint whose declared quant_algo the pinned image does
# not dispatch: the ModelOptMixedPrecisionConfig falls back to
# UnquantizedLinearMethod for unrecognized algos, i.e. silent garbage (monolith
# :843-844).
#
# Hooks for offline tests: QUANT_PREFLIGHT=0 skips entirely. Host-side
# introspection runs `$QUANT_HOST_PY_CMD - <snapshot>` (default `python3`).
# Image-side introspection runs `bash -c "$QUANT_IMG_PY_SH"` with
# MODEL_SNAPSHOT/IMAGE exported and the python probe on stdin; the default is
# the monolith's docker one-liner (:797-798) with the snapshot mounted
# read-only at /m. A failed image probe warns + skips (:824-826) — never
# hard-fail over an offline box or an older image.

# Shared probe: prints the sorted set of quant_algos present. The engine reads
# quantized_layers per class, not a dispatchable algo, so only when no
# quantized_layers exist is the top-level value checked (monolith :779-780).
# argv[1] = snapshot dir; with no argv (docker case) it reads the /m mount.
quant_probe_py='
import json, pathlib, sys
m = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/m")
cfg = json.loads((m / "config.json").read_text()) if (m / "config.json").is_file() else {}
qc = cfg.get("quantization_config")
if not qc and (m / "hf_quant_config.json").is_file():
    qc = json.loads((m / "hf_quant_config.json").read_text())
if not qc:
    print(""); sys.exit(0)
ql = (qc.get("quantization") or qc).get("quantized_layers", {})
algos = {str(v["quant_algo"]).upper() for v in ql.values()
         if isinstance(v, dict) and v.get("quant_algo")}
if not algos:
    top = qc.get("quant_algo") or qc.get("quant_method") or ""
    algos = {str(top).upper()} if isinstance(top, str) else {str(a).upper() for a in top}
print(" ".join(sorted(algos)))
'

quant_preflight() {  # consumes MODEL_SNAPSHOT, IMAGE
    [[ "${QUANT_PREFLIGHT:-1}" == 1 ]] || { warn "quant pre-flight disabled (QUANT_PREFLIGHT=0)"; return 0; }
    QUANT_HOST_PY_CMD="${QUANT_HOST_PY_CMD:-python3}"
    QUANT_IMG_PY_SH="${QUANT_IMG_PY_SH:-docker run --rm -i --entrypoint python3 -v \"\$MODEL_SNAPSHOT:/m:ro\" \"\$IMAGE\" -}"

    local declared supported
    declared=$(printf '%s\n' "$quant_probe_py" | $QUANT_HOST_PY_CMD - "$MODEL_SNAPSHOT" 2>/dev/null || true)
    [[ -n "$declared" ]] || { info "quant pre-flight: unquantized checkpoint (or config unreadable); nothing to check."; return 0; }

    # What the image dispatches: exactly the algos present in quantized_layers —
    # get_quant_method() returns UnquantizedLinearMethod for anything else
    # (monolith :816-817).
    supported=$(MODEL_SNAPSHOT="$MODEL_SNAPSHOT" IMAGE="$IMAGE" \
        bash -c "$QUANT_IMG_PY_SH" <<PY 2>/dev/null || true
$quant_probe_py
PY
)
    if [[ -z "$supported" ]]; then
        warn "quant_algo pre-flight: image introspection failed (offline / older image); skipping."
        warn "     Checkpoint declares quant_algo: $declared"
        return 0
    fi

    local missing="" a
    for a in $declared; do
        [[ " $supported " == *" $a "* ]] || missing="$missing $a"
    done
    if [[ -n "$missing" ]]; then
        err "quant_algo$missing declared by the checkpoint is not dispatched by $IMAGE.
       The image's ModelOptMixedPrecisionConfig falls back to UnquantizedLinearMethod
       for unrecognized algos — silent garbage. Checkpoint/image mismatch."
    fi
    ok "quant_algo pre-flight: checkpoint algos ($declared) dispatched by this image."
}
