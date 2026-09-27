# Reorg design — one start.sh for single- and dual-DGX-Spark Qwen3.8-Flash-Next

Status: **design + skeleton** in this scratch dir. Nothing here has run on a
real Spark; the smoke evidence is the offline fake-docker matrix (§Tests).
The goal is the locked spec: exactly one `start.sh` + one `stop.sh`, zero
flags, auto-detecting SINGLE Spark vs one node of a DUAL pair, recipes as
tiny data files, all logic in `engine/*.sh` modules under 150 lines.

```
start.sh  stop.sh                     # the only two entry points
engine/detect.sh                      # topology: who am I (role from local IP)
engine/budget.sh                      # Step-2 memory budget (monolith :496-560)
engine/ple.sh                         # packed-mmap default + pinned refusal guard
engine/quant-detect.sh               # checkpoint-algo vs image-dispatch pre-flight
engine/memwatch.sh                    # watchdog starter (body = files/memwatch.sh)
recipes/prod.conf                     # daily driver (v0.30, K=6, MTP off, packed)
recipes/arm-a4.conf                   # A4 combo ckpt, K=10 + MTP5 drafter
recipes/compare.conf                  # pinned-offload A/B (refused on 128G by design)
recipes/peers.conf.example            # per-operator overrides (copy to peers.conf)
tests/recipe_lint.sh                  # keys/pins validator — green on this laptop
tests/detect_matrix.sh                # detect.sh vs fake ip/nvidia-smi — green
files/NOTES.md                        # runtime deps to drop in (NOT copied: big)
```

## What moved where (old file → new file)

Sources read here on the laptop: `single-spark/` = MiaAI single-Spark sister
repo (AGPL); `pushrepo/` = our dual repo fork (branch experiments/speed);
`pushrepo/tp1/` = the older single-node launcher inside it.

| Old | Where its logic lives now | Port refs |
|---|---|---|
| `single-spark/start.sh` (1299 L monolith) Step-2 | `engine/budget.sh` | constants `:209-248`; formula `:525-537`; pinned-GMU override `:539-553`; guards `:586-604`; WHY-comments `:42-56` |
| `single-spark/start.sh` PLE build | `engine/ple.sh` | build block `:880-888`; refusal guard `:396` (re-based to the pinned-offload rule, spec'd for ≤128G); cache-id `:877-879` |
| `single-spark/start.sh` quant pre-flight | `engine/quant-detect.sh` | `:760-851` incl. the silent-garbage rationale `:843-844` |
| `single-spark/start.sh` watchdog start | `engine/memwatch.sh` | `scripts/start-memwatch.sh:1-31` |
| `single-spark/files/memwatch.sh` (body) | stays a file — `files/NOTES.md` (177 L poll loop, one conceptual unit) | policy header `:6-44` summarized in engine/memwatch.sh |
| `single-spark/start-v030.sh` (22 L env wrapper) | `recipes/prod.conf` | its pins `:11-19` → IMAGE/KV/PLE_GIB=47.68/MTP_WEIGHTS_GIB=2.34/MEMWATCH floors |
| `pushrepo/start.sh` (1142 L dual monolith) TP2 path | `start.sh` dual branch | `--nnodes 2/--master-addr/--master-port` `:813-815`; EP `:817-819`; NCCL/RoCE env `:881-888`; rank0 `:1092-1094`, rank1 `--node-rank 1 --headless` `:1030-1031`; hf-overrides-nesting lesson `:836-842` |
| `pushrepo/start.sh` head-ssh-worker orchestration | **deleted** (see Deletion list) | the spec: same script on both nodes, role from local IP |
| `pushrepo/tp1/start.sh` | superseded by single-spark monolith port | same shape, older numbers |
| `pushrepo/spike_v30/launch_v30.sh` | `recipes/prod.conf` arg fragment + derived-GMU block already in budget.sh | derived block `:18-36`; serve-"$MODEL" lesson `:113-117` of its README |
| `pushrepo/start-fp8.sh / start-lean.sh / start-tp1.sh / start-v030.sh` | recipes (data), near-duplicate launchers die | — |
| `pushrepo/stop.sh` + `single-spark/stop.sh` | `stop.sh` | graceful SIGTERM rationale `:6-9`; pkill `[m]` anchor `:48-51`; shm report-don't-delete `:92-94`; STOP_TIMEOUT validation `:30-37` |
| `pushrepo/files/evict_page_cache.py` | stop.sh calls it | behavior `files/evict_page_cache.py:4-25` |

## Detection as implemented (engine/detect.sh)

* NOT GPU count (every GB10 box = exactly one GPU, pair or not).
* `ip -o -4 addr show` → any address inside **192.168.100.0/30**?
  * none → `single` (TP1).
  * `.1` → `dual` rank 0 (head, serves HTTP); `.2` → `dual` rank 1 (headless).
  * `.0/.3` → refuse (network/broadcast — misconfiguration).
* Link quality: interface flags contain `,UP,` and `,LOWER_UP` (carrier) **and**
  `ip -4 route get <peer>` resolves via that iface → `FABRIC_LINK=up`;
  otherwise `down` + warning but still dual (the peer may boot later — the
  same script must work on whichever node starts first).
* Everything external is a hook (`DETECT_IP_CMD`, `DETECT_NVIDIA_SMI_CMD`,
  `DETECT_MEMTOTAL_FILE`) so tests run offline on a laptop with no NVIDIA GPU.

Budget caveat (honest): in dual, `HOST_RESERVE_GIB=26` stays per-node — each
Spark still runs its own CPU offload worker and its own page-cache pressure.
Only the *weights* split by TP_SIZE=2. That assumption is §Open risks #1.

## Memory-fit arithmetic (our box, 121 GiB usable)

Formula, ported with comments (engine/budget.sh = monolith :496-560):

```
w        = (checkpoint_bytes/2³⁰ − PLE_GIB) / TP_SIZE − MTP-off-credit
fixed    = w + OVERHEAD_GIB(5.6) + MTP_GIB(1.49 if MTP on)
kv_need  = MAX_MODEL_LEN × 29482 × KV_MULT(fp8→0.58) / 2³⁰
budget   = min(fixed + max(kv_need, KV_TARGET_GIB), MemTotal − HOST_RESERVE_GIB)
GMU      = floor(budget/MemTotal × 1000)/1000          # 3 decimals = what vLLM gets
```

Numbers for the prod recipe on a 121.7 GiB pool (real 106 GiB NVIDIA ckpt):
weights on GPU ≈ 55.9−… → derived **GMU ≈ 0.60-0.62**, budget ≈ 73.5 GiB,
KV ≈ 12 GiB ≈ **~750k tokens** at fp8 — the v0.30 config requested
(modelopt NVFP4 + fp8 PLE via packed mmap + fp8-KV). Floor guard:
`MemTotal < 110` → refuse before any launch (non-Spark boxes). The negative
guard (smoke R4) refuses a recipe/ckpt PLE_GIB mismatch instead of launching
a nonsense budget — this class of bug is exactly what the pinned-table crashes
on 2026-09-26 produced on the old launchers.

## Deletion list (once this repo is adopted)

From `pushrepo/` root: `start.sh` (1142 L), `start-fp8.sh`, `start-lean.sh`,
`start-tp1.sh`, `start-v030.sh`, `tp1/start.sh`, `tp1/stop.sh`,
`spike_v30/launch_v30.sh`, `spike_ps/ps_launch.sh`, `spike_hyb/hyb_launch.sh`
— all replaced by start.sh + a recipe. Also dead per spike_v30/README.md
§"Now NATIVE in v0.30": `files/modelopt_patched.py`, `files/ple_offload/`
4-file stack, `hyb_spike/stack_modelopt.py` **after** the first real v0.30
boot confirms native dispatch. Keep: `download.sh`, `check-weights.sh`,
`bench/`, `scripts/` supervision layer (supervise.sh may gain a recipe arg
later — out of scope here), `files/*.py` patches still marked [fp8dense].
From `single-spark/`: whole repo retires once its `files/` payload is dropped
into this repo's `files/` (NOTES.md lists exactly which).

## Tests (run on this laptop — both green)

* `tests/recipe_lint.sh` — data-only (no `$( )`/backticks), known-keys whitelist
  (typo'd keys are silently dead at launch; lint kills that class), IMAGE
  pinned to `@sha256:`, CKPT_SHA256 64-hex, CONTAINER namespaced, PLE_MODE enum.
* `tests/detect_matrix.sh` — 4 topology fixtures + meminfo plumbing, all PASS.
* Offline end-to-end (`/tmp/smoke3.sh`, throwaway): fake docker/ip/nvidia-smi —
  R1 single TP1 launch · R2 dual rank0 (`--nnodes 2 --node-rank 0`, NCCL env) ·
  R3 dual rank1 (`--headless`, exits before health loop) · R4 nonsense-budget
  refusal rc=1 · R5 old-container refusal · R6 arm-a4 (`num_speculative_tokens:5`,
  K=10 in argv) · R7 pinned refused on 121 GiB rc=1.

## Open risks / needs the real boxes

1. **Dual budget model untested.** Per-node HOST_RESERVE=26 + weights/TP2 is
   reasoned, not measured. gx10 pair must boot once with `nvidia-smi dmon` +
   memwatch logs to confirm the 200G TP2+EP path (NCCL IB HCA/GID names are
   `.env.sample` defaults, overridable per recipe).
2. **Pins are placeholders.** IMAGE digest + CKPT_SHA256 were derived
   deterministically (documented in each recipe header) — replace from
   `docker image inspect` / `sha256sum` on the boxes.
3. **files/ payload absent here** (by design, NOTES.md): memwatch.sh,
   evict_page_cache.py, build_ple_packed_table.py + the fp8dense overlay must
   be copied into `files/` before a real boot; start.sh warns loudly instead
   of silently skipping the watchdog, and packed-PLE build fails pre-launch.
4. **v0.30 packed-mmap first-boot behaviour on msi.** launch_v30's crash note
   2 says the UVA dummy-forward touches all 47.7 GiB of the map during
   profiling; the cgroup cap + evict fixes are ported, but the packed path on
   v0.30 (not the patched serve image) still needs one supervised boot.
5. **Dual weights-sync** is out of scope for the zero-flag start.sh: each node
   must already have the checkpoint + packed table (rank 1 refuses without).
   A `sync.sh` helper is the obvious next file, deliberately not invented here.
6. YaRN/`VLLM_ALLOW_LONG_MAX_MODEL_LEN` >262144 paths from the monolith are
   not ported (prod serves 131072); peers.conf.example mentions the knob.
