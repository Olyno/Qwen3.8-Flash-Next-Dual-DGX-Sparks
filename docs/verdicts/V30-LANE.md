# vLLM v0.30 lane — measured status (2026-09-27)

The upgrade is an ENABLER, not a win by itself: the control row proves it.

## A1 baseline — v0.30 stock K6, fp8-KV (#55557 backport), mmap-PLE, ctx 114688
decodebench 600 tok, temp 0.6, ~/v30_bench/v30_k6_pass1.txt:

| ctx | prose | code | entropy | copy |
|---|---|---|---|---|
| 1k | 17.1 | 17.5 | 17.5 | 18.8 |
| 100k | 17.4 | 17.3 | 17.3 | 17.7 |

TTFT warm 0.6-2.1 s; prose@100k cold 43 s. Old-image stock was 16.0-17.4 ->
**v0.30 port is speed-neutral** (+2-6 %, within protocol variance). All gains
must come from the arms on top: A2 spec-stack, A3 hybrid, A4 lean (running).

Concurrency (pre-fix ladder): C=1 16.6/stream, C=32 11.1/stream aggregate 3.2,
TTFT@32 91 s. Single Spark saturates ~C=8-16; 30+ seats is a dual-pair story.

## What v0.30 actually bought (verified in source + this lane)
- native `disable_eagle_block_drop` (speculative.py:440; -50 % TTFT, sister-measured)
- MTP probabilistic draft + block rejection as config (the A2-A4 stack)
- `--enable-return-routed-experts` (T1 telemetry: MoE verify cost per K)
- safetensors load strategies (host-spike control at load)
- PLE offload native BUT pinned (~104 GiB unevictable = 7 hangs); fixed by
  the mmap patcher this lane carries (files/patch_ple_mmap_v030.py).

## Engine legality — MTP+QSA block arithmetic (cost us the K5 spec)
Attention block = 3232 (16-aligned, GDN conv page). QSA ring capacity =
4*ceil((4+k)/4). k in {5..8} -> 12, 3232 % 12 = 4 -> HARD BOOT FAIL (observed
A2 09-27 19:44, exit 1). Legal k = {0..4, 9..12, 25..28}. **Product = k=4**
(closest legal to the locked K5 intent; T1 sweep now {1,2,3,4,9} covers the
9-12 plateau properly). recipes/prod.conf carries the full note.

## Boot economics (the nine-hang saga, closed)
mmap table: build-once (+~11 min first boot), committed via msync + fingerprint
sidecar; later boots reuse ("reused file-backed table", pinned=False). v0.30
boots now 13-22 min deterministic. Kernel kit v2 (hung_task_panic@120 + water-
marks + swappiness 30) persists on msi — D-state hangs become 2-min reboots.

## A2-A4 measured rows (09-27 21:12, all k=4 stack, fp8-KV, mmap-PLE)
decodebench 600 tok temp .6, ~/v30_bench/{v30_k6,hyb_v30,lean_hyb_mtp4pb}_pass1.txt:

| class 1k / 100k | A1 stock | A2 +MTP | A3 xhyb | A4 xlean (PRODUCT) |
|---|---|---|---|---|
| prose | 17.1 / 17.4 | 21.8 / 21.1 | 25.8 / 31.3 | **30.5 / 30.7** |
| code  | 17.5 / 17.3 | 25.0 / 22.4 | 32.2 / 32.7 | **35.5 / 35.9** |
| entropy | 17.5 / 17.3 | 24.8 / 24.9 | 43.4 / 30.7 | 34.0 / 36.2 |
| copy  | 18.8 / 17.7 | 45.8 / 44.8 | 66.7 / 62.3 | 65.6 / 61.1 |

- Product vs floor: prose x1.79, code x2.0 measured on v0.30.
- A3 beats old-image hybrid EVERY class (code 32.2 vs 26.1, copy 66.7 vs 29.2):
  the spec stack pays ON TOP of the checkpoint-level hybrid gains.
- entropy@1k A4 dips (34.0 vs A3 43.4): random-content acceptance behaves
  differently under lean's shorter chains; low-value class, noted not gated.
- Lean adds effective-time gain beyond the rate: -10-16 % thinking tokens
  (old-image measured) at the same tok/s.
- Conc rows on MTP arms: n_ok-collapse (C=1 dbg: HTTP 200, zero countable
  deltas) — preemption/abort-frame-under-contention suspected; raw SSE capture
  scheduled at T1 idle gap; concbench2 (--ignore-eos --min-tokens) = D1 repair.
  Does not affect pass1 (sequential bench) validity.


## gpqa200 G1-gate FINAL (05:26) — -1.5 pp: FAIL vs the <=1 pp bar, PENDING engine control
Like-for-like re-anchor (GateReanchor 05:3x): current score.py re-scores BOTH
files byte-identically to their originals (A4 161/198, baked 164/198; md5
match; score.py unchanged on-disk since the baseline run) — the delta is real,
not a scorer artifact. Paired McNemar: discordants 11 vs 8, p = 0.57 —
statistically indistinguishable, but the product rule is the point estimate.
**A4 81.3 % vs 82.8 % = -1.515 pp > 1 pp ⇒ gate NOT passed as written.**
- Token anatomy: A4 mean 7576 / median 5824 / trunc 49 vs baked 7220 / 5174 /
  42 — A4 reasons ~5 % LONGER than its own lean baseline, the opposite of the
  bake's signature (-16 %). Something about the combo (lean bake x FP8-dense x
  fp8-KV x v0.30) is re-lengthening chains; near-cap rows convert to
  truncations (49) which is where the 11 discordant losses plausibly live.
- CONFOUNDER: baseline 82.8 is OLD-IMAGE + bf16-KV. A4 is v0.30 + fp8-KV. The
  gate never isolated engine numerics from the bake. CONTROL RUNNING: A3
  (q38-hyb, no lean) gpqa200 on the IDENTICAL v0.30 fp8-KV stack
  (ride_a3gate.sh, arm a3-gate, chained after T1).
  * A3 lands ~78-79 (old-image value)  -> engine neutral -> the -1.5 pp is
    lean-specific (bake x hyb interaction) -> retune knob D1, ship A3.
  * A3 lands ~81+ (also up ~3 / flat vs its old 78.3) -> engine shift affects
    all arms -> A4 >= A3 + lean token gains -> prod.conf stands as measured.
  (Old-image banked rows for the two: hyb 78.3, lean/baked 82.8.)
- Interim product until the control lands: A3 stack (hyb x MTP4 x fp8KV x
  mmap; 25.8/32.2 prose/code) — every G1 number behind it is measured.
- Gate-history hygiene: the 7 harness-error rows were retried (append-only
  resume), all scored; first-pass 78.8 and pre-correction -4.0 pp figures are
  superseded by this section.

## Provenance footnote (09-29)
All A1-A4 + gate speed rows above were booted at `max_num_batched_tokens=2048`
(the ride scripts' export; `launch_v30.sh`'s own default 8192 never applied).
c=1 steady decode at bench ctx is budget-insensitive (6-row steps, one-chunk
prefill) so the ladder comparisons stand; the one open question is whether the
prod config (8192, recipes/prod.conf) shifts TTFT@1k/100k — first true 8192
row = R1 arm (ride_r1.sh), recorded here when it lands.

## T1 fixed-K speed ladder (wk1 arm, prose@1k tok/s; 09-28 sweep)
K=1 23.5 · K=2 23.9 · K=3 20.9 · K=4 20.2 · K=9 11.9 (copy rises with K:
31.0/34.7/40.7/44.5). Telemetry was lost to the `$KK_metrics` bash bug; T1b
re-dumps /metrics per arm -> t1_analyze_v2 EV verdict (acceptance data needed
to call whether K2/K3 beats K4 for prose EV — the second-box "-6 % prose at
K4" datapoint says check this). Analyzer dry-run 09-29 parses all five rows
(drafts=0 until T1b lands). NOTE the anomaly: this ladder's K1/K2 rows are
ABOVE the A2 K4 row on the same class (20.2 vs A2 21.8 hyb-less 25.8?) —
different checkpoints (wk1 vs hybrid); do not cross-read columns across arms.

## Async scheduling: already ON by default at v0.30 for our stack (09-29)
Image grep (vllm/config/vllm.py:1407-1466): explicit `--async-scheduling` raises
only for non-EAGLE/MTP/draft_model/dspark spec methods; the auto path disables
(with a `warning_once`) only on those methods or `disable_padded_drafter_batch`.
`method='mtp'` is inside `EagleModelTypes` (config/speculative.py:69-71) and the
mp executor returns `supports_async_scheduling()=True` (multiproc_executor.py:558).
The running A3 container's 17 k boot-log lines contain zero "async" strings —
no disable-warning fired ⇒ scheduler already runs AsyncScheduler (max_concurrent
_batches=2). Step-overhead hiding is therefore NOT an open lever; R3's ITL row
and the P1b2 profile will show whether the overlap is effective at k=4.
