#!/bin/bash
# gb10_guard.sh — the #56824-class startup-collapse guard + forensic sampler for
# the bench box (upstream issue: engine init drives a GB10 unified pool to
# exhaustion during KV-profiling/graph-capture; the host sees NOTHING until
# NV_ERR_NO_MEMORY; the community's only working defense = external 2-3 s
# MemAvailable guard that kills the container before the kernel hits the wall.
# Six msi freezes (09-29/30) journal-end at journald "under memory pressure"
# -> 44 s stall -> silence: consistent with that curve reaching zero = host
# loss. If this guard fires and the box SURVIVES, the class is confirmed and
# the boot becomes a logged failure instead of a hard freeze. If the box dies
# WITH the guard running and a clean thermal trace, it's power/thermal
# hardware — the sampler history is the evidence.
#
# Started by chain_r2 (and by the @reboot selfheal). One instance: pidfile.
R=$HOME/v30_bench
LOG=$R/guard.log
PIDF=$R/guard.pid
[ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null && exit 0
echo $$ > "$PIDF"
trap 'rm -f $PIDF' EXIT
FLOOR=${GUARD_FLOOR_GIB:-6}          # same operating point as the community earlyoom
KILLS=0
while true; do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    tz=$(paste -sd/ /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tr -d '\n')
    gpus=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -c "^v30")
    echo "$(date +%H:%M:%S) avail=${avail}G v30=${gpus} zones=${tz:-none}" >> "$LOG"
    # keep the log bounded (append-only bench logs are a repo law; rotate 2 MB)
    [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 2097152 ] && tail -c 1048576 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
    if [ "$avail" -lt "$FLOOR" ] && [ "$gpus" -gt 0 ] && [ "$KILLS" -lt 3 ]; then
        echo "$(date +%H:%M:%S) GUARD-FIRED avail=${avail}G killing v30 engines (rescue attempt $((++KILLS)))" >> "$LOG"
        docker ps --format "{{.Names}}" | grep "^v30" | xargs -r docker stop -t 10 >> "$LOG" 2>&1
    fi
    sleep 5
done
