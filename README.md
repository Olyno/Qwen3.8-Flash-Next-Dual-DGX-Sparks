# Qwen3.8-Flash-Next on DGX Spark — one launcher, single or dual

Serving stack for [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
and our measured-improvement builds of it, on 1 or 2 DGX Spark GB10 boxes, vLLM v0.30.
Exactly two entry points — `./start.sh` and `./stop.sh` — zero required flags: topology,
memory budget, PLE policy and engine args all come from detection plus a small data recipe.
Every performance/quality claim below links to a measured verdict in `docs/verdicts/`
or the crash/port history in `experiments/v30/README.md`. Builds on
[getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark);
FP8-KV kernel work vendored from
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
(AGPL-3.0-or-later).

## Results at a glance (measured, single Spark unless noted)

| Stack | prose tok/s | code tok/s | GPQA | vs |
|---|---|---|---|---|
| v0.30 stock floor (A1) | 17.1 | 17.5 | — | baseline |
| + MTP k=4 drafting (A2) | 21.8 | 25.0 | — | ×1.28 / ×1.43 |
| + FP8-dense hybrid ckpt (A3) | 25.8 | 32.2 | 78.3 % (old image) | +4.1 pp quality vs stock |
| + lean thinking bake (A4 = **prod**) | **30.5** | **35.5** | 82.8 % baked / 81.3 % v0.30 re-gate | **×1.79 / ×2.00**, K6 +4-5 % free |

Reactivity shipped since 09-28: persisted Triton JIT cache, breakable CUDA
graphs off, native 262K context (YaRN 1M retired — acceptance collapse at
long positions). Every number: `docs/verdicts/` (ladder + gate math in
`V30-LANE.md`).

## Hardware

| Kit | What runs | How it is found |
|---|---|---|
| 2× pair, direct link (any subnet) | TP=2 + expert parallel; head serves `:8888`, worker headless | `.env` sets `HEAD_IP`+`WORKER_IP`: the node whose local address equals `HEAD_IP` is rank 0, `WORKER_IP` is rank 1 |
| 2× pair, no `.env`, sister convention | same | address inside `192.168.100.0/30` on a link UP with carrier + on-link route (`.1` head / `.2` worker; prefix overridable `FABRIC_NET=`) |
| 1× DGX Spark GB10 (128 GB unified) | TP=1, serves `:8888` | neither fires → single |

With `.env` naming a pair this host is not in, the launcher REFUSES (a silent
single-TP fallback on a pair box is how the 09-27 gx10 confusion happened).
NCCL knobs for the fabric: `IFACE` (socket ifname), `IB_HCA`, `IB_GID_INDEX` — `.env`, example values in `.env.example`.

Not a GPU-count check on purpose: every GB10 box has exactly one GPU, alone or in a pair.
More than one GPU, or a `MemTotal` under 110 GiB, is not a Spark and the launcher refuses —
the unified-memory budget model below does not apply elsewhere.

## 5-minute prod try

```bash
# 1. Get a checkpoint. Stock from the hub (~126 GiB, on the node you run it):
./download.sh nvidia/Qwen3.8-Flash-Next-NVFP4
#    …or point the recipe (or the env) at a local build — recipes/prod.conf ships
#    MODEL_PATH="$HOME/models/q38-lean-hyb" (lean×hybrid, ~120 GiB, built + verified 09-27).

# 2. Launch. Auto-detects single vs dual; derives the memory budget; starts the watchdog.
./start.sh                      # cold load takes minutes; it follows logs until /health answers

# 3. Use it.
curl -s http://localhost:8888/health
curl -s http://localhost:8888/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}]}'

# 4. Stop (graceful SIGTERM, then page-cache eviction for the next boot).
./stop.sh
```

Dual: run everything from the HEAD. `./start.sh` there ssh's to `WORKER_IP`
(`WORKER_USER`), ships the image (`docker save|load`, first run only), rsyncs
this repo dir + the checkpoint to identical absolute paths on the worker, then
launches rank 1 headless, waits 15 s, launches rank 0 and follows health. The
worker needs only: docker, key-auth ssh from the head, and the same `.env`
(rsynced). `./stop.sh` on the head stops worker-first then head — one command
for both boxes. Ctrl-C during launch only detaches the log follower; the
containers keep running.

## Recipes (`recipes/`)

Plain `KEY=VALUE` data files — no logic, linted by `tests/recipe_lint.sh` (image pinned by
`@sha256` digest, container namespaced, keys whitelisted so typos cannot die silently).
Precedence: exported env > `recipes/$RECIPE.conf` > `recipes/peers.conf` > engine default.

- **`prod.conf` — the locked daily driver.** vLLM `v0.30.0` (digest-pinned) ×
  lean×hybrid checkpoint (`MODEL_PATH`, `FP8DENSE=1` overlay) × K=6 experts ×
  MTP k=4 probabilistic drafting with block rejection and
  `disable_eagle_block_drop` (native key in v0.30 — no patch; k=5 is ILLEGAL
  on QSA: ring `4*ceil((4+k)/4)` must divide the 16-aligned block size — hard
  boot fail) × PLE via mmap × fp8 KV (`fp8_e4m3` via the #55557 backport) ×
  131072 ctx native (no YaRN), lazy load, chunked prefill, `FULL_DECODE_ONLY`
  CUDA graphs + breakable graphs off, modelopt quant, `qwen3` reasoning
  parser + `qwen3_coder` tool parser.
  **Status (09-29, measured + gated):** decode prose 30.5 / code 35.5 tok/s
  single stream = ×1.79 / ×2.00 over the v0.30 floor (A1 17.1–18.8). gpqa200
  re-gate: 81.3 % vs 82.8 % baked (−1.515 pp against a ≤1 pp bar); the A3
  engine-control run resolved it — A3 (no-lean, same stack) scores **identical
  138/163**, so the delta is the v0.30/fp8-KV-era numerics, not the lean bake
  (McNemar p=.33 N.S.) ⇒ **prod stands**. Full ledger: `docs/verdicts/V30-LANE.md`.
- **Bench-arm ladder.** `arm-a1.conf` is the baseline floor (stock wk1 checkpoint, K=6, no
  drafter, bf16 KV, no overlay) — every speed claim above it is paid for against its
  decodebench row. The ladder: **A1 baseline → A2 +MTP k4 → A3 +hybrid → A4 +lean = prod**
  (`arm-a1.conf`..`arm-a4.conf`). Arms use their own `CONTAINER` + port, so an A/B never collides with
  a running prod; launch one with `RECIPE=arm-a4 ./start.sh`.
- **`peers.conf.example`** — copy to `peers.conf` (gitignored) for per-operator overrides:
  a local image digest, a different `MODEL_PATH`, ports, `FABRIC_NET` for non-`192.168.100.x`
  wiring. Single node: nothing needed. Dual: nothing needed either unless the fabric differs.

## What's inside

```
start.sh  stop.sh  download.sh         # the only entry points
engine/     detect.sh · budget.sh · ple.sh · quant-detect.sh · memwatch.sh   # all logic, <150 L each
files/      runtime payload: v0.30 patchers, fp8dense overlay (model/mtp/hyperconnection.py),
            evict_page_cache.py, draft-vocab list + patchers
bench/      decodebench.py · longctx.py · acceptance_sweep.py · mtp_accept.py · reasoning_check.py
tools/bake/   lean checkpoint recipe (bake_lm_head.py + marker_token_map.json)
tools/hybrid/ fp8-dense checkpoint builder + verify (the A3/A4 checkpoint source)
prompts/cod/  chain-of-draft prompts used by the lean/gate measurements
recipes/      prod.conf · pair.conf · arm-a1.conf · arm-a4.conf · peers.conf.example
docs/verdicts/ measured verdicts — index below
experiments/  bench lanes with their own scripts and raw rows:
                v30/        the v0.30 arm launcher + analyzers (README inside =
                            crash-triage history; primary source for the memory model)
                probsparse/ K6 expert-sampling A/B
tests/        recipe_lint.sh · detect_matrix.sh · ssh_env_quoting.sh — offline, no GPU
```

## The memory model (read this before tuning anything)

- **One pool, budgeted from the host side.** GB10 has a single 128 GB LPDDR5X pool
  (~121.7 GiB `MemTotal`) shared by GPU allocations, weights staging, the PLE table and the
  Linux page cache. vLLM fills the GPU side to `GMU × MemTotal`, so an uncapped KV wish eats
  exactly the pages the PLE cache and the driver need. `engine/budget.sh` therefore derives
  GMU from the *host* budget: weights (÷TP) + overhead + MTP + KV need, capped at
  `MemTotal − 26 GiB host reserve`, and the container cgroup cap follows it. In dual, the
  reserve stays per-node — only the weights split.
- **Why pinned PLE offload hangs (and is refused ≤128 GiB).** v0.30's native offload keeps
  the 47.7 GiB n-gram table in anonymous **pinned** RAM: with weights that is ~104 GiB of
  non-evictable footprint in a ~121 GiB pool — the pool exhausts, the kernel hangs, and there
  is no OOM record to show you (three msi crashes on 2026-09-26, `experiments/v30/README.md` +
  `engine/ple.sh`). `PLE_MODE=pinned` exits at the guard by design (A/B it on a >128 GB box only,
  via `recipes/peers.conf` — that is the only place it fits.)
- **The mmap table is the proven contract.** Default `PLE_MODE=mmap`:
  `files/patch_ple_mmap_v030.py` overlays the image's `ngram_embedding.py` with a file-backed
  map under `VLLM_PLE_MMAP_DIR` (inside the already-mounted `~/.cache/vllm`), keyed per
  (layer prefix, snapshot fingerprint, TP rank), committed with `msync` + a sidecar. First
  boot builds it lazily; every later boot maps the existing file straight in — no copy.
  File-backed pages are reclaimable page cache, and `VLLM_PLE_MMAP_ADVICE=1` madvises them,
  so PLE lookups degrade to disk instead of hanging the box. An earlier file design re-pinned
  the map and re-copied the whole table every boot and hung the box again (2026-09-27); it is
  gone and must not be reintroduced.
- **Page cache is a first-class citizen.** On unified memory, a checkpoint left resident from
  a download or a previous launch can push weight loading into a `CUDA out of memory` on an
  otherwise idle box. `stop.sh` releases the checkpoint's own clean pages with
  `posix_fadvise(DONTNEED)` (no root, touches only those files); the system-wide drop still
  needs root: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- **memwatch is the second line of defence.** Because an exhausted pool hangs instead of
  raising OOM, `files/memwatch.sh` polls `MemAvailable`/`MemFree` (floors from the recipe,
  e.g. `MEMWATCH_MIN_GIB=3`) and, on a debounced breach, archives `docker logs --tail 3000`
  then SIGTERMs the container. Timeline log: `logs/memwatch-<container>.log`.

## v0.30 patch deltas — what is still ours, and when each goes away

v0.30 made most of the old serve-image patches dead: modelopt mixed-precision arms
(FP8 per-channel-per-token, FP8 block-scale MoE), FP8 PLE dispatch, native `EngramConfig`
offload, MTP draft quant config and quant-prefix matching are all first-class upstream now
(`experiments/v30/README.md` §"Now NATIVE"). What remains, with delete-conditions:

| Still ours (in `files/`) | Why it exists | DELETE when |
|---|---|---|
| `patch_ple_mmap_v030.py` (mmap PLE) | native offload = pinned RAM; hangs ≤128 GiB pools | upstream's PLE offload can keep the table in evictable file-backed pages (then re-bench vs native pinned via `compare` on a >128 GB box and keep the winner) |
| `patch_qsa_fp8_kv_v030.py` — backport of vllm-project/vllm#55557 (fp8_e4m3 QSA KV) | v0.30 image predates the merge | the image moves to vLLM ≥ 0.31, which ships #55557 natively (stated in the patcher's own docstring) |
| `patch_mtp_draft_vocab_v030.py` + the 47k/65k draft-vocab files | reduced-vocabulary drafting slices the drafter lm_head (1.18 -> 0.22 GiB); coverage decides its worth: en+code 47k is ~flat for English, but fr/de/zh/ja/ru/pt prose is under-covered (fr 69 %) — the vendored per-language 65k files lift it to ~99.5 % (measured +10…+77 % decode by language, `recipes/prod-fr.conf`) | the slice engages only via greedy drafting (`use_local_argmax_reduction` is rejected alongside sampled drafting, PR #71); `prod.conf` keeps probabilistic + full head = unaffected. Dual-lane wiring + fk3-verified 09-29 |
| fp8dense overlay: `model.py` `mtp.py` `hyperconnection.py` (`FP8DENSE=1` mounts) | stock v0.30 still hardcodes `quant_config=None` in `GatedResidual` and misses the MTP HC mixer → FP8 weights load into bf16 Linears | upstream passes the resolved mixed-precision config through the HC mixers + final LM heads for `qwen4_exp` (then a stock NVFP4 checkpoint needs zero mounts, and this row dies with the re-port script `experiments/v30/port_v30.py`) |

The block-drop lever is **not** a patch: `disable_eagle_block_drop` is a native v0.30
speculative-config key (−50 % TTFT, sister-measured; prod ships it in
`SPECULATIVE_CONFIG`).

## Verdicts index (`docs/verdicts/`)

- **`DFLASH.md` — KILLED (day 1.5).** External block-diffusion drafter vs a pre-registered
  bar of 2.9 tok/step prose (this cluster's measured MTP=3 number). The strongest free
  mechanism (ngram lookup) reached 2.69; the trained packaged draft measured 1.37–1.46 and
  sat *at* the no-spec band (16.2–16.8 tok/s); live production MTP=3 drafts τ=2.31 for free.
  Speculation on this kit is capped at the quality of the built-in MTP head's chains.
- **`DDTREE.md` — KILLED (day 0 of the reopen).** The draft's 16-way candidate tree (offline
  oracle 4.77 accepted-len) never reaches the target: this stack verifies single chains, and
  36 of 60 layers are gated-delta-net with one state per request — tree verification is a
  fork-level kernel project, not wiring. Absorbs into the DFlash conclusion.
- **`PROBSPARSE.md` — ADOPTED (serving default, K=6 of 512-routed-10).** Decode
  +4.4–5.0 % (prose/code, ctx 1k; +2.5–7.7 % across all cells), quality statistically null
  on GPQA (+2.6)/MATH (−0.2)/GSM8K (−2 problems at n=100, paired test null). Free gain;
  reversible per boot via `--hf-overrides`, zero patch.
- **`HYBRID-FP8.md` — ADOPTED.** Dense projections re-quantized FP8 per-channel: decode
  **+60–68 %** across all content classes (measured; analytical model said +30–50 % — NVFP4
  dequant cost was undercounted), GPQA 78.3 % (**+4.1 pp**), MATH −0.4 (noise), GSM8K tie.
  Needs the fp8dense loader overlay.
- **`LEAN.md` — ADOPTED (the "thinking changes" axis).** Offline bake of the overthinking
  marker-penalty into `lm_head`: standalone measured GPQA 82.8 % (+8.6 pp, p=.002, with
  −16 % completion tokens); 86.4 % (+12.2 pp, p<.001, −33.6 % tokens) stacked with the
  Chain-of-Draft prompts. In the combo ledger (`HYBRID-FP8.md`) the lean axis is carried at
  +5.6 pp, and the lean×hybrid combo owes its own 3-suite re-gate to arm A4.
- **`V30-LANE.md` — the v0.30 ledger (current).** All A1-A4 speed rows above,
  the MTP/QSA k-legality arithmetic, the nine-hang boot saga, and the closed
  gpqa200 gate: prod combo 81.3 % vs 82.8 % baked (−1.515 pp vs the ≤1 pp
  bar) resolved by the A3 engine-control — same stack without the lean bake
  scores 138/163, **identical to A4** ⇒ engine numerics, not lean ⇒
  **prod.conf confirmed as the product**.

## Troubleshooting

- **`/health` never answers.** `start.sh` follows the container logs while it waits; if you
  detached, `docker logs --tail 200 <container>` (name from the recipe: `vllm-fn-prod`,
  `vllm-fn-a1`, …). Look for: quant-dispatch refusals (checkpoint algo vs image),
  anchor-refused patchers (they refuse on upstream drift instead of silently mis-patching),
  and where weight loading stalled.
- **Box hard-hangs during/after load, no OOM anywhere.** That is the unified pool. Check
  `free -g` before launching, and the memwatch timeline `logs/memwatch-<container>.log`.
  Classic triggers: launching a server while a bulk copy runs (two memory hogs — serialize
  heavy jobs), leftover page cache from a previous boot (`./stop.sh` evicts; or root
  `drop_caches`), a recipe asking for more than the derived budget.
- **`CUDA out of memory` on an idle box.** Almost always checkpoint pages left resident —
  run `./stop.sh` (or `python3 files/evict_page_cache.py <snapshot dir>`) and relaunch.
- **psm_*/sem.mp-* files in `/dev/shm` after a stop.** A SIGKILL'd `--ipc host` container
  leaks its POSIX shm segments until reboot; this is why both `stop.sh` and memwatch stop
  with SIGTERM + grace first. `stop.sh` reports the residue and never deletes it — other
  containers may own those segments.
- **Refusals are features.** `PLE_MODE=pinned` under 126 GiB, `MemTotal < 110`, a taken
  port, a running same-name container, a missing `FP8DENSE` overlay file, fewer than 60 GiB
  free under `~/.cache/vllm` — each exits with the reason and the fix in its message.
- **Dual node won't join.** Verify the 200G link: both nodes need an address in the fabric
  /30 with carrier (`ip -o -4 addr`); NCCL HCA/GID names are recipe-overridable
  (`IB_HCA`, `IB_GID_INDEX` in `peers.conf`) for cross-wired boxes.

---

License AGPL-3.0-or-later. The lean bake's marker-token set and bake script live in
`tools/bake/`; the served checkpoint's own licenses (NVIDIA Open Model License / Qwen
Community License) govern the weights.
