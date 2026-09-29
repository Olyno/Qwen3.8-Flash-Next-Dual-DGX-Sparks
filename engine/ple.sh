#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# engine/ple.sh — PLE n-gram table policy for v0.30 (PROVEN mmap contract).
#
# History matters here, so it is spelled out:
#   * v0.30 native EngramConfig offload (PLE_MODE=pinned) holds the 47.7 GiB
#     table as anonymous PINNED RAM -> ~104 GiB non-evictable with weights ->
#     unified-pool exhaustion, kernel hang, no OOM record (msi crashes #3-#5,
#     experiments/v30/README.md). REFUSED on <=128 GB pools below.
#   * The day-0 packed-file idea (VLLM_PLE_PACKED_TABLE_DIR + *.packed_u8 +
#     RW bind) is DEAD design: it re-pinned the mmap through UVA and re-copied
#     47.7 GiB on every boot (hang #8, msi 2026-09-27 ~13:0x). Do not
#     reintroduce it.
#   * The PROVEN path (default here, PLE_MODE=mmap) is
#     files/patch_ple_mmap_v030.py (from files/), which
#     overlays the image's ngram_embedding.py with a torch.from_file map under
#     VLLM_PLE_MMAP_DIR — a subdir of the ALREADY rw-mounted ~/.cache/vllm, per
#     (layer prefix, snapshot fingerprint, tp rank), committed via
#     msync + a .json sidecar fingerprint. The FIRST boot builds it lazily;
#     every later boot maps the existing file straight away (no copy). Pages
#     are file-backed = reclaimable page cache, and VLLM_PLE_MMAP_ADVICE=1
#     madvises them. No build step, no separate bind mount.
#
# Dependency of the mmap path: the patcher must be applied to the PRISTINE
# image ngram_embedding.py AT LAUNCH (extract-from-image + patch; see
# ple_prepare_overlay below and files/NOTES.md).

ple_policy() {  # consumes PLE_MODE, MEM_TOTAL_GIB; sets PLE_OFFLOAD_ENV (+ PLE_* paths)
    PLE_MODE="${PLE_MODE:-mmap}"
    case "$PLE_MODE" in
        mmap)
            # Dir lives under ~/.cache/vllm which start.sh mounts rw wholesale:
            # host $HOME/.cache/vllm -> /root/.cache/vllm. Nothing extra to
            # mount; pre-creating the host dir is all the contract needs
            # (the patcher itself also mkdirs inside the container).
            PLE_MMAP_HOST="$HOME/.cache/vllm/ple_mmap_v030"
            PLE_MMAP_CTR="/root/.cache/vllm/ple_mmap_v030"
            PLE_OFFLOAD_ENV=(-e VLLM_PLE_CPU_OFFLOAD=1
                             -e "VLLM_PLE_MMAP_DIR=$PLE_MMAP_CTR"
                             -e VLLM_PLE_MMAP_ADVICE=1)
            ;;
        pinned)
            # Guard (monolith :396 re-based): the pinned path cannot fit a
            # 128 GB unified pool. ~104 GiB non-evictable pinned footprint
            # (47.7 GiB table + weights + KV) hung msi three times on
            # 2026-09-26. 126 GiB separates a 128G Spark (~121.7 usable)
            # from a genuinely bigger box; below that, refuse.
            if python3 -c "import sys; sys.exit(0 if $MEM_TOTAL_GIB < 126 else 1)"; then
                err "PLE_MODE=pinned refused on this box (${MEM_TOTAL_GIB%.*} GiB pool).
       v0.30 native offload keeps the PLE table in NON-evictable anonymous RAM
       (~104 GiB pinned footprint): it hung the unified pool three times on
       2026-09-26 (experiments/v30/README.md). Use the default PLE_MODE=mmap, or run
       on a >128 GB pool if you must A/B the pinned path."
            fi
            PLE_OFFLOAD_ENV=(-e VLLM_PLE_CPU_OFFLOAD=1)
            ;;
        off)
            PLE_OFFLOAD_ENV=(-e VLLM_PLE_CPU_OFFLOAD=0)
            ;;
        *) err "PLE_MODE must be mmap|pinned|off (got: '$PLE_MODE').
       NB 'packed' (VLLM_PLE_PACKED_TABLE_DIR/*.packed_u8) is a DEAD design:
       UVA re-pin + 47.7 GiB copy every boot -> hang #8. It is now 'mmap'." ;;
    esac
}

ple_prepare_overlay() {  # mmap mode only: extract pristine ngram_embedding.py + patch
    [[ "$PLE_MODE" == mmap ]] || return 0
    local vpkg=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
    local dst="$SCRIPT_DIR/files/v030_ple" patcher="$SCRIPT_DIR/files/patch_ple_mmap_v030.py"
    if [[ ! -f "$patcher" ]]; then
        warn "files/patch_ple_mmap_v030.py absent — mmap PLE path NOT wired."
        warn "     Copy it (files/) before booting; without the"
        warn "     overlay VLLM_PLE_MMAP_DIR is ignored and the boot takes the"
        warn "     pinned path this launcher exists to avoid."
        return 0
    fi
    if [[ ! -f "$dst/ngram_embedding.py" ]]; then
        info "Preparing PLE mmap overlay (extract pristine file from $IMAGE + patch)..."
        mkdir -p "$dst/orig"
        docker run --rm --entrypoint sh "$IMAGE" -c "cat $vpkg/ngram_embedding.py" \
            > "$dst/orig/ngram_embedding.py"
        # The patcher is anchor-checked: it refuses on drift against the image's
        # own file, so a silently-wrong overlay is not possible.
        python3 "$patcher" || err "patch_ple_mmap_v030.py refused the image's ngram_embedding.py (upstream drift)."
    fi
    # The OVERLAY file itself is read-only in the engine; only the mmap DIR
    # needs write access, and that rides the wholesale ~/.cache/vllm mount.
    PLE_OVERLAY=(-v "$dst/ngram_embedding.py:$vpkg/ngram_embedding.py:ro")
    ok "PLE mmap overlay: $dst/ngram_embedding.py"
}
