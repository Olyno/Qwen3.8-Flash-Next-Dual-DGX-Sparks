#!/usr/bin/env bash
# tests/detect_matrix.sh — unit-tests engine/detect.sh against fake environments
# (no GPU, no 200G cable — runs green on this laptop). Fakes answer exactly the
# three calls detect.sh makes:
#   ip -o -4 addr show      -> candidate lines "N: ifc inet A/30 ..."
#   ip -o link show dev X   -> "N: X: <...,UP[,LOWER_UP]> ..."
#   ip -4 route get PEER    -> "... dev X ..." or nothing
# DETECT_NVIDIA_SMI_CMD fakes one GB10; DETECT_MEMTOTAL_FILE a 128G pool.
#
# Matrix (Contract):
#   single fixture   no address on 192.168.100.0/30       -> single, rank -
#   dual .1 fixture  192.168.100.1/30 UP+LOWER_UP+route  -> dual, rank 0, up
#   dual .2 fixture  192.168.100.2/30 UP+LOWER_UP+route  -> dual, rank 1, up
#   .2 no-carrier    UP without LOWER_UP                  -> dual, rank 1, down
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
FIX="$SCRIPT_DIR/.fixtures.$$"
trap 'rm -rf "$FIX"' EXIT
mkdir -p "$FIX/bin"

printf 'MemTotal:       127608704 kB\nMemFree:  40000000 kB\nMemAvailable:  104857600 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n' > "$FIX/meminfo-128"

# mk_fake <name> <addr-line> <flags> <route|none>
mk_fake() {
    local name="$1" addr="$2" flags="$3" route="$4" ifn
    ifn="$(printf '%s' "$addr" | awk '{gsub(/:$/,"",$2); print $2}')"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'case "$1" in\n'
        printf '  -o)\n    case "$2" in\n'
        printf '      -4)   [ "$3" = addr ] && printf "%%s\\n" %q ;;\n' "$addr"
        printf '      link) [ "$3" = show ] && echo "4: reth0: <%s> mtu 9000" ;;\n' "$flags"
        printf '    esac ;;\n'
        if [[ "$route" == route ]]; then
            printf '  -4)   [ "$2" = route ] && echo "peer dev %s src local" ;;\n' "$ifn"
        else
            printf '  -4)   [ "$2" = route ] && exit 1 ;;\n'
        fi
        printf 'esac\nexit 0\n'
    } > "$FIX/bin/$name"
    chmod +x "$FIX/bin/$name"
}

mk_fake ip-single          "1: lo    inet 127.0.0.1/8 scope host lo" "LOOPBACK,UP,LOWER_UP" none
mk_fake ip-dot1            "4: reth0    inet 192.168.100.1/30 brd 192.168.100.3 scope link reth0" "BROADCAST,MULTICAST,UP,LOWER_UP" route
mk_fake ip-dot2            "4: reth0    inet 192.168.100.2/30 brd 192.168.100.3 scope link reth0" "BROADCAST,MULTICAST,UP,LOWER_UP" route
mk_fake ip-dot2-nocarrier  "4: reth0    inet 192.168.100.2/30 brd 192.168.100.3 scope link reth0" "BROADCAST,MULTICAST,UP" route

cat > "$FIX/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == --query-gpu=name ]] && echo "NVIDIA GB10 [DGX Spark]"
exit 0
EOF
chmod +x "$FIX/bin/nvidia-smi"

mk_fake ip-single          "1: lo    inet 127.0.0.1/8 scope host lo" "LOOPBACK,UP,LOWER_UP" none
mk_fake ip-dot1            "4: reth0    inet 192.168.100.1/30 brd 192.168.100.3 scope link reth0" "BROADCAST,MULTICAST,UP,LOWER_UP" route
mk_fake ip-dot2            "4: reth0    inet 192.168.100.2/30 brd 192.168.100.3 scope link reth0" "BROADCAST,MULTICAST,UP,LOWER_UP" route
mk_fake ip-dot2-nocarrier  "4: reth0    inet 192.168.100.2/30 brd 192.168.100.3 scope link reth0" "BROADCAST,MULTICAST,UP" route
mk_fake ip-gx10head        "5: enp1s0f0np0    inet 10.200.0.1/30 brd 10.200.0.3 scope link enp1s0f0np0" "BROADCAST,MULTICAST,UP,LOWER_UP" route
mk_fake ip-gx10worker      "5: enp1s0f0np0    inet 10.200.0.2/30 brd 10.200.0.3 scope link enp1s0f0np0" "BROADCAST,MULTICAST,UP,LOWER_UP" route

pass=0; failn=0
check() {  # check <label> <fake-ip> <want mode|rank|link>
    local label="$1" ipbin="$2" want="$3" got rc=0
    got=$(DETECT_IP_CMD="$FIX/bin/$ipbin" \
          DETECT_NVIDIA_SMI_CMD="$FIX/bin/nvidia-smi" \
          DETECT_MEMTOTAL_FILE="$FIX/meminfo-128" \
          bash -c "source '$REPO/engine/detect.sh'
                   detect_topology >/dev/null 2>&1 || exit 3
                   echo \"\$TOPO_MODE|\$NODE_RANK|\$FABRIC_LINK|\$GPUS\"" 2>/dev/null) || rc=$?
    if (( rc == 0 )) && [[ "$got" == "$want" ]]; then
        echo "PASS: $label -> $got"; pass=$((pass + 1))
    else
        echo "FAIL: $label got [$got] want [$want] (rc=$rc)"; failn=$((failn + 1))
    fi
}

check "single fixture  (no fabric addr)"  ip-single          "single|-|absent|1"
check "dual .1 fixture (head, link up)"   ip-dot1            "dual|0|up|1"
check "dual .2 fixture (worker, link up)" ip-dot2            "dual|1|up|1"
check "dual .2 no-carrier (link down)"    ip-dot2-nocarrier  "dual|1|down|1"

# MemTotal plumbing: the 128 GiB fixture must land ~121.7 (budget.sh gate uses it).
memt=$(DETECT_IP_CMD="$FIX/bin/ip-single" DETECT_NVIDIA_SMI_CMD="$FIX/bin/nvidia-smi" \
       DETECT_MEMTOTAL_FILE="$FIX/meminfo-128" \
       bash -c "source '$REPO/engine/detect.sh'; detect_topology >/dev/null 2>&1; echo \"\$MEM_TOTAL_GIB\"" 2>/dev/null)
if python3 -c "import sys; sys.exit(0 if abs($memt - 121.7) < 0.5 else 1)"; then
    echo "PASS: meminfo fixture plumbed (MEM_TOTAL_GIB=$memt)"; pass=$((pass + 1))
else
    echo "FAIL: MEM_TOTAL_GIB=$memt, expected ~121.7"; failn=$((failn + 1))
fi

# Explicit-pair path (.env HEAD_IP/WORKER_IP — the gx10 layout: fabric on
# 10.200.0.0/30, which the default 192.168.100 probe cannot see).
check_env() {  # check_env <label> <fake-ip> <want>
    local label="$1" ipbin="$2" want="$3"
    local got rc=0
    got=$(DETECT_IP_CMD="$FIX/bin/$ipbin" \
          DETECT_NVIDIA_SMI_CMD="$FIX/bin/nvidia-smi" \
          DETECT_MEMTOTAL_FILE="$FIX/meminfo-128" \
          HEAD_IP=10.200.0.1 WORKER_IP=10.200.0.2 \
          bash -c "source '$REPO/engine/detect.sh'
                   detect_topology >/dev/null 2>&1 || exit 3
                   echo \"\$TOPO_MODE|\$NODE_RANK|\$FABRIC_LINK|\$GPUS\"" 2>/dev/null) || rc=$?
    if (( rc == 0 )) && [[ "$got" == "$want" ]]; then
        echo "PASS: $label -> $got"; pass=$((pass + 1))
    else
        echo "FAIL: $label got [$got] want [$want] (rc=$rc)"; failn=$((failn + 1))
    fi
}
check_env "pair .env: head holds HEAD_IP"    ip-gx10head   "dual|0|up|1"
check_env "pair .env: worker holds WORKER_IP" ip-gx10worker "dual|1|up|1"

echo "detect_matrix: $pass passed, $failn failed"
(( failn == 0 )) || exit 1
