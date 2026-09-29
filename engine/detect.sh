#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# engine/detect.sh — topology detection. No side effects, sourceable standalone
# (tests/detect_matrix.sh sources exactly this file).
#
# TWO ways to be a dual pair, tried in order:
#  1. EXPLICIT PAIR (.env or exported): HEAD_IP + WORKER_IP both set. The rank
#     is the node whose local address equals one of them (any interface — the
#     gx10 pair runs 10.200.0.1/30 on enp1s0f0np0, which no /30-probe default
#     would find). A box holding neither address is a config error, never a
#     silent single: refusing beats TP=1 with pair env on.
#  2. IMPLIED /30 (msi single box, or a pair with no .env): an address on
#     192.168.100.0/30 (override the prefix with FABRIC_NET=10.200.0) on an
#     interface that is UP + carrier + has an on-link route to the peer.
#     Convention (sister repos + TensorFold): .1 = rank 0 head (HTTP),
#     .2 = rank 1 headless worker.
# SINGLE is what you get when NEITHER fires: no GPU-count heuristic (every
# GB10 has exactly one GPU, alone or in a pair).
#
# Every external command is read through a variable so this can be unit-tested
# offline with fake ip/nvidia-smi/meminfo (this laptop has no NVIDIA GPU):
#   DETECT_IP_CMD         default `ip`
#   DETECT_NVIDIA_SMI_CMD default `nvidia-smi`
#   DETECT_MEMTOTAL_FILE  default /proc/meminfo
#   FABRIC_NET            default 192.168.100 (the /30's /24 prefix)
DETECT_IP_CMD="${DETECT_IP_CMD:-ip}"
DETECT_NVIDIA_SMI_CMD="${DETECT_NVIDIA_SMI_CMD:-nvidia-smi}"
DETECT_MEMTOTAL_FILE="${DETECT_MEMTOTAL_FILE:-/proc/meminfo}"
FABRIC_NET="${FABRIC_NET:-192.168.100}"

# Loggers, guarded: start.sh supplies coloured ones; standalone sourcing works too.
declare -F info >/dev/null || info() { printf '[INFO]  %s\n' "$*"; }
declare -F warn >/dev/null || warn() { printf '[WARN]  %s\n' "$*"; }
declare -F err  >/dev/null || err()  { printf '[ERR ]  %s\n' "$*" >&2; exit 1; }

# Outputs (consumed by start.sh / budget.sh):
#   TOPO_MODE single|dual   NODE_RANK   0|1 (dual), - (single)
#   NODE_IP HEAD_IP PEER_IP FABRIC_IFACE FABRIC_LINK  up|down|absent
detect_topology() {
    # pair env (path 1) captured BEFORE the reset below wipes it.
    local _head_cfg="${HEAD_IP:-}" _worker_cfg="${WORKER_IP:-}"
    TOPO_MODE=single; NODE_RANK=-; NODE_IP=; HEAD_IP=; PEER_IP=
    FABRIC_IFACE=; FABRIC_LINK=absent; GPUS=0

    GPUS=$($DETECT_NVIDIA_SMI_CMD --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [[ "$GPUS" == 0 ]]; then
        warn "no NVIDIA GPU visible via '$DETECT_NVIDIA_SMI_CMD'"
    elif [[ "$GPUS" != 1 ]]; then
        # GB10 is one GPU per box. More => we are not on a Spark; the memory
        # budget in engine/budget.sh (unified-pool model) does not apply.
        warn "GPUS=$GPUS: this is not a 1-GPU GB10 Spark; unified-memory budget maths may not hold."
    fi

    # MemTotal / MemAvailable: the GPU side of a Spark is budgeted FROM these
    # (engine/budget.sh), so read them here once.
    read -r MEM_TOTAL_GIB MEM_AVAIL_GIB <<<"$(python3 -c "
m = {l.split(':')[0]: int(l.split()[1]) for l in open('$DETECT_MEMTOTAL_FILE') if ':' in l}
g = 1048576
print(m['MemTotal'] / g, m['MemAvailable'] / g)")"

    # --- path 1: explicit pair from .env / exported env ---------------------
    # (IFACE/IB_HCA/IB_GID_INDEX stay operator-pinned from .env; start.sh
    # passes them to NCCL unchanged.)
    if [[ -n "$_head_cfg" && -n "$_worker_cfg" ]]; then
        local myip="" myifc="" ifc a
        while read -r ifc a; do
            a="${a%%/*}"
            if [[ "$a" == "$_head_cfg" || "$a" == "$_worker_cfg" ]]; then
                myip="$a"; myifc="$ifc"; break
            fi
        done <<<"$($DETECT_IP_CMD -o -4 addr show 2>/dev/null \
            | awk '$3 == "inet" { gsub(/:$/, "", $2); print $2, $4 }')"
        if [[ -z "$myip" ]]; then
            err "pair configured (HEAD_IP=$_head_cfg WORKER_IP=$_worker_cfg) but this host holds neither address. This machine is not a pair member — unset the .env or fix the IPs."
        fi
        if [[ "$myip" == "$_head_cfg" ]]; then NODE_RANK=0; PEER_IP="$_worker_cfg"
        else NODE_RANK=1; PEER_IP="$_head_cfg"; fi
        TOPO_MODE=dual; NODE_IP="$myip"; HEAD_IP="$_head_cfg"; FABRIC_IFACE="$myifc"
        # link state: same carrier+route probe, on whichever iface holds it.
        local flags
        flags=$($DETECT_IP_CMD -o link show dev "$myifc" 2>/dev/null || true)
        if [[ "$flags" == *,UP,* && "$flags" == *,LOWER_UP* ]] \
           && $DETECT_IP_CMD -4 route get "$PEER_IP" 2>/dev/null | grep -q "dev $myifc"; then
            FABRIC_LINK=up
        else
            FABRIC_LINK=down
            warn "pair configured, iface $myifc lacks carrier/route to $PEER_IP (peer may boot later)."
        fi
        info "explicit pair: rank $NODE_RANK ($NODE_IP on $FABRIC_IFACE), head $HEAD_IP, link $FABRIC_LINK"
        return 0
    fi

    # --- path 2: implied /30 auto-detect --------------------------------------
    # Candidates: interfaces already addressed inside the fabric /30.
    # (if-guards, never bare `[[ ]] &&` bodies: a loop whose last command is a
    # failed and-or returns non-zero and kills a `set -e` caller.)
    local -a cands=()
    local ifc addr
    while read -r ifc addr; do
        if [[ -n "$ifc" ]]; then cands+=("$ifc|$addr"); fi
    done <<<"$($DETECT_IP_CMD -o -4 addr show 2>/dev/null \
        | awk -v p="${FABRIC_NET}." '$3 == "inet" && index($4, p) == 1 { gsub(/:$/, "", $2); print $2, $4 }')"

    if (( ${#cands[@]} == 0 )); then
        info "no pair config and no address on ${FABRIC_NET}.0/30 -> single Spark (TP=1)"
        return 0
    fi
    if (( ${#cands[@]} > 1 )); then
        warn "${#cands[@]} interfaces addressed on ${FABRIC_NET}.0/30; using ${cands[0]%%|*}"
    fi

    local cidr last
    cidr="${cands[0]#*|}"; cidr="${cidr%%/*}"
    ifc="${cands[0]%%|*}"; last="${cidr##*.}"
    case "$last" in
        1) NODE_RANK=0; HEAD_IP="${FABRIC_NET}.1"; PEER_IP="${FABRIC_NET}.2" ;;
        2) NODE_RANK=1; HEAD_IP="${FABRIC_NET}.1"; PEER_IP="${FABRIC_NET}.2" ;;
        # .0/.3 of a /30 are network/broadcast: someone mis-typed the address.
        *) err "${FABRIC_NET}.0/30 has only .1 and .2 usable; $ifc holds $cidr" ;;
    esac
    TOPO_MODE=dual; NODE_IP="$cidr"; FABRIC_IFACE="$ifc"

    # Link state: flags arrive as `<...,UP,LOWER_UP>` — IFF_UP + IFF_LOWER_UP
    # (carrier) means the 200G cable is in and the peer port is awake. Plus a
    # route to the peer, which must be on-link.
    local flags
    flags=$($DETECT_IP_CMD -o link show dev "$ifc" 2>/dev/null || true)
    if [[ "$flags" == *,UP,* && "$flags" == *,LOWER_UP* ]] \
       && $DETECT_IP_CMD -4 route get "$PEER_IP" 2>/dev/null | grep -q "dev $ifc"; then
        FABRIC_LINK=up
    else
        FABRIC_LINK=down
        warn "fabric addressed on $ifc ($cidr) but link/route to $PEER_IP is down."
        warn "     Still proceeding as DUAL: the peer may boot after this node."
    fi
    info "dual pair: this node rank $NODE_RANK ($cidr on $ifc), head $HEAD_IP, link $FABRIC_LINK"
}
