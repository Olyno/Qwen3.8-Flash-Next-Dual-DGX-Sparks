#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# engine/budget.sh — Step-2 memory budget, ported from the single-Spark monolith
# start.sh:496-560 (constants at :209-248). Requires engine/detect.sh
# (MEM_TOTAL_GIB, MEM_AVAIL_GIB, TOPO_MODE) and MODEL_SNAPSHOT (resolved path).
#
# WHY the GPU budget is capped from the HOST side (monolith :42-56):
#   GB10 has one unified LPDDR5X pool. vLLM treats "free memory" as
#   MemAvailable (page cache included) and fills the GPU side to exactly
#   GMU x MemTotal, so an uncapped KV wish comes straight out of the PLE page
#   cache and the free pages the NVIDIA driver needs. That is what killed three
#   servers on 2026-09-04 (docs/memory-incident-2026-09-04.md s.7). The reserve
#   covers: co-tenants (~7 GiB measured), vLLM's host-side procs (~6), PLE page
#   cache (>=6), driver free-page reserve (>=3), 2-3 GiB of per-request growth
#   that is never returned.
#
# Budget shape (monolith :521-522):
#   min(weights + overhead + MTP + max(kv_need, KV_TARGET_GIB),
#       MemTotal - HOST_RESERVE_GIB)
# and the KV figure is whatever the capped budget leaves. GMU is floored to the
# 3 decimals vLLM is given, so the printed figures are what vLLM will do.

# --- constants (monolith line references) -------------------------------------
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-29482}"  # :497 measured 28.8 KiB/tok, bf16 KV
# :209 runtime overhead on top of weights, GiB (measured at TP1: 3.37+1.92+0.12).
OVERHEAD_GIB="${OVERHEAD_GIB:-5.6}"
# :212 KV the derived budget targets when GMU is not pinned. More KV = more UVM.
# The host-side cap below wins over this wish.
KV_TARGET_GIB="${KV_TARGET_GIB:-12.0}"
# :217 memory the GPU budget may never take: GPU side capped at MemTotal minus
# this whatever KV_TARGET_GIB asks. Raise in 2 GiB steps if the watchdog log
# shows MemAvailable idling under ~9 GiB; do NOT lower it to buy KV.
HOST_RESERVE_GIB="${HOST_RESERVE_GIB:-26}"
# :220 host-side memory the container needs beyond the GPU budget: three Python
# processes, pinned staging buffers, CPU-side torch, page cache slack.
HOST_SLACK_GIB="${HOST_SLACK_GIB:-10.0}"
# :222 never let the container cgroup cap come within this much of the pool.
OS_RESERVE_GIB="${OS_RESERVE_GIB:-16.0}"
# :235-241 the packed PLE table's size, subtracted from the on-disk checkpoint
# to get GPU-resident weights. Stock/ablit snapshots = 26.82 (measured, drill
# report 2026-09-10); the NVIDIA checkpoint packs 47.68 GiB PLE in
# model-fp8-mtp-ple.safetensors -> start-v030.sh:14 pins 47.68 for it.
PLE_GIB="${PLE_GIB:-47.68}"
# :242-248 GiB of MTP draft weights living inside the checkpoint, loaded only
# when MTP is on (NVIDIA packs 2.34 GiB next to the PLE table); credited back
# when MTP is off, where nothing loads them. 0 = no credit.
MTP_WEIGHTS_GIB="${MTP_WEIGHTS_GIB:-2.34}"
MTP_GIB_ON="${MTP_GIB_ON:-1.49}"    # :509 measured draft-model footprint, MTP on
# Pool floor: 128 GB LPDDR5X shows up as ~121.7 GiB. Below 110 the reserve and
# the weights do not coexist (refuse; do not guess).
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-110}"

compute_budget() {  # sets GPU_MEMORY_UTILIZATION BUDGET_GIB KV_EXPECT_* etc.
    if python3 -c "import sys; sys.exit(0 if $MEM_TOTAL_GIB < $MEM_FLOOR_GIB else 1)"; then
        err "MemTotal ${MEM_TOTAL_GIB%.*} GiB < ${MEM_FLOOR_GIB}: not a 128 GB Spark.
       HOST_RESERVE_GIB=${HOST_RESERVE_GIB} + PLE page cache do not fit here; refusing to launch."
    fi
    local weight_bytes="${BUDGET_WEIGHT_BYTES:-$(du -sb "$MODEL_SNAPSHOT/" -L | cut -f1)}"

    # :506-512 MTP on => pay 1.49 GiB; MTP off => credit the packed draft weights.
    local mtp_gib=0 mtp_off_credit=0
    if [[ "${MTP_NUM_SPECULATIVE_TOKENS:-0}" -gt 0 ]]; then
        mtp_gib="$MTP_GIB_ON"
    else
        mtp_off_credit="$MTP_WEIGHTS_GIB"
    fi
    # :513-516 FP8 halves the main KV (12 full-attn layers, ~84% of bytes/token)
    # but the QSA side/compressor caches stay BF16: the real saving is ~1.7x,
    # not 2x.
    local kv_mult=1.0
    [[ "${KV_CACHE_DTYPE:-auto}" == fp8* ]] && kv_mult=0.58

    # Weight split: one GPU per Spark (TP_SIZE=1 single / 2 dual). The PLE
    # table never lands on the GPU (mmap page cache in the CPU offload worker,
    # engine/ple.sh), so the whole PLE_GIB is host-side on every node:
    # subtract once, THEN split.
    local tp="${TP_SIZE:-1}"
    read -r WEIGHTS_GPU_GIB KV_NEED_GIB BUDGET_GIB DERIVED_GMU KV_EXPECT_GIB \
             KV_EXPECT_TOK BUDGET_CAP_GIB CAP_BINDS <<<"$(python3 -c "
import math
w = ($weight_bytes / 2**30 - $PLE_GIB) / $tp
w -= min($mtp_off_credit / $tp, max(w, 0))
fixed = w + $OVERHEAD_GIB + $mtp_gib
kv_need = ${MAX_MODEL_LEN:-131072} * $KV_BYTES_PER_TOKEN * $kv_mult / 2**30
wish = fixed + max(kv_need, $KV_TARGET_GIB)
cap = $MEM_TOTAL_GIB - $HOST_RESERVE_GIB
budget = min(wish, cap)
gmu = math.floor(budget / $MEM_TOTAL_GIB * 1000) / 1000
budget = gmu * $MEM_TOTAL_GIB
kv_exp = budget - fixed
print(f'{w:.2f} {kv_need:.2f} {budget:.2f} {gmu:.3f} {kv_exp:.2f} '
      f'{int(max(kv_exp, 0) * 2**30 / ($KV_BYTES_PER_TOKEN * $kv_mult))} {cap:.2f} {int(wish > cap)}')")"

    # :539-553 a caller-pinned GMU wins, with the incident warning if it exceeds
    # the host-side cap.
    if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
        warn "  caller-pinned GMU=$GPU_MEMORY_UTILIZATION (derived would be $DERIVED_GMU)"
        local pinned_kv
        read -r BUDGET_GIB KV_EXPECT_GIB KV_EXPECT_TOK <<<"$(python3 -c "
b = $GPU_MEMORY_UTILIZATION * $MEM_TOTAL_GIB
kv = max(b - $WEIGHTS_GPU_GIB - $OVERHEAD_GIB - $mtp_gib, 0)
print(f'{b:.2f} {kv:.2f} {int(kv * 2**30 / ($KV_BYTES_PER_TOKEN * $kv_mult))}')")"
        CAP_BINDS=0
        if python3 -c "import sys; sys.exit(0 if $BUDGET_GIB > $BUDGET_CAP_GIB else 1)"; then
            warn "  pinned budget ${BUDGET_GIB} GiB is ABOVE the host-side cap ${BUDGET_CAP_GIB} GiB"
            warn "  (MemTotal - HOST_RESERVE_GIB=${HOST_RESERVE_GIB}). This is the configuration"
            warn "  that killed three servers on 2026-09-04. You asked for it; the watchdog ends it."
        fi
    else
        GPU_MEMORY_UTILIZATION="$DERIVED_GMU"
    fi

    CONTAINER_MEM_GIB="${CONTAINER_MEM_GIB:-$(python3 -c "print(int($BUDGET_GIB + $HOST_SLACK_GIB))")}"
    MAX_CONTAINER_GIB=$(python3 -c "print(int($MEM_TOTAL_GIB - $OS_RESERVE_GIB))")

    info "  unified pool ............. ${MEM_TOTAL_GIB%.*} GiB total, ${MEM_AVAIL_GIB%.*} GiB available now"
    info "  weights on GPU ........... ${WEIGHTS_GPU_GIB} GiB  (checkpoint/TP${tp} minus ${PLE_GIB} GiB PLE table)"
    if [[ "$mtp_off_credit" != 0 ]]; then info "  MTP draft weights ........ ${mtp_off_credit} GiB  credited back (MTP off: packed in the checkpoint, never loaded)"; fi
    info "  PLE table ................ ${PLE_GIB} GiB  memory-mapped in the CPU offload worker"
    info "  runtime overhead ......... ${OVERHEAD_GIB} GiB"
    if [[ "$mtp_gib" != 0 ]]; then info "  MTP draft model .......... ${mtp_gib} GiB"; fi
    info "  KV needed for ${MAX_MODEL_LEN:-131072} ...... ${KV_NEED_GIB} GiB  (kv dtype ${KV_CACHE_DTYPE:-auto}, mult ${kv_mult})"
    info "  host reserve ............. ${HOST_RESERVE_GIB} GiB  (HOST_RESERVE_GIB) => GPU budget cap ${BUDGET_CAP_GIB} GiB"
    if [[ "$CAP_BINDS" == 1 ]]; then warn "  KV target ${KV_TARGET_GIB} reduced to ${KV_EXPECT_GIB} by HOST_RESERVE_GIB=${HOST_RESERVE_GIB}"; fi
    info "  GPU budget (GMU ${GPU_MEMORY_UTILIZATION}) ... ${BUDGET_GIB} GiB  => ~${KV_EXPECT_GIB} GiB KV (~${KV_EXPECT_TOK} tokens)"
    info "  container cgroup cap ..... ${CONTAINER_MEM_GIB} GiB  (hard ceiling ${MAX_CONTAINER_GIB}; bounds host-side memory only)"

    # Guards, ported from :586-604, plus a consistency gate the monolith never
    # needed: its checkpoints are 98+ GiB, so a bogus PLE_GIB (47.68 against a
    # stock 26.82 table) could not go negative there. Ours must refuse instead
    # of launching a negative GMU.
    if python3 -c "import sys; sys.exit(0 if $WEIGHTS_GPU_GIB < 10 or $GPU_MEMORY_UTILIZATION <= 0 or $GPU_MEMORY_UTILIZATION >= 1 else 1)"; then
        err "Derived budget is nonsense (weights ${WEIGHTS_GPU_GIB} GiB, GMU ${GPU_MEMORY_UTILIZATION}).
       Checkpoint on disk and the recipe's PLE_GIB disagree: stock Mia NVFP4
       packs 26.82 GiB of PLE, the nvidia build 47.68. Fix MODEL_ID/PLE_GIB,
       or measure: du -sb <snapshot>."
    fi
    if python3 -c "import sys; sys.exit(0 if $KV_EXPECT_GIB < $KV_NEED_GIB else 1)"; then
        err "Budget leaves ${KV_EXPECT_GIB} GiB for KV but ${MAX_MODEL_LEN:-131072} tokens need ${KV_NEED_GIB} GiB.
       Lower MAX_MODEL_LEN, keep KV_CACHE_DTYPE=fp8, or buy context deliberately."
    fi
    if [[ "$CONTAINER_MEM_GIB" -gt "$MAX_CONTAINER_GIB" ]]; then
        err "Container cap ${CONTAINER_MEM_GIB} GiB exceeds the hard ceiling ${MAX_CONTAINER_GIB} GiB
       (pool ${MEM_TOTAL_GIB%.*} GiB minus OS_RESERVE_GIB=${OS_RESERVE_GIB}). On unified memory
       this is the line between a killed container and a hung host. Lower the budget."
    fi
    if python3 -c "import sys; sys.exit(0 if $MEM_AVAIL_GIB < $CONTAINER_MEM_GIB + 4 else 1)"; then
        err "Only ${MEM_AVAIL_GIB%.*} GiB available now but the container may use ${CONTAINER_MEM_GIB} GiB.
       Something else is holding memory (docker ps; ps --sort=-rss)."
    fi
}
