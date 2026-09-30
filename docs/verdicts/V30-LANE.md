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

## v0.30.1rc0 is landmine-contaminated — stay on v0.30.0 (09-29)
415-commit diff audit (GitHub compare API): rc0 carries **#55390** ("annotate
MTP draft KV groups positionally" = the −30 % GDN-prefix-TPS regression opener
from hunt-2) but NOT its fix #58368 ⇒ any rc0 build sits inside the known
landmine window. Also inside (skipped/irrelevant): #57885 sparse-meta perf
(MLA indexer/sparse_swa files only — not our QSA builder), #57273 sm_90-only
QSA table (H100 tuning, not SM121), #57396 vocab-mapping CPU-GPU sync removal
(dead code for us: VocabMapping is inactive on the draft-vocab-patch path),
#55867 FP8-TP FlashInfer TRTLLM MoE for this exact model (merged 09-17, so
pre-v0.30.0? — base branch main, cut date 09-21 ⇒ present in our image).
Real upgrade candidates remain v0.31.0rc1+ AFTER the R queue harvests the
#58449 fused-draft port (rc1 already contains #58400+#58368).

## A3 engine-control GATE VERDICT (2026-09-29 13:30, n=163 common-anchor)
A3 = q38-hyb (NO lean bake) on the v0.30 stack (fp8-KV backport, GMU .748,
BATCHED 2048, k=4 block) — the control for the A4 −1.515 pp finding.
Raw: 138/163 valid = 84.66 % of valid; 35 ids error-out (runner HTTP retries;
gap-fill rerun armed). Reanchored to the 163 ids all three arms answered:
  baked(lean,v0.27) 143/163 = 87.73 %
  A4  (lean,v0.30)  138/163 = 84.66 %
  A3  (hyb, v0.30)  138/163 = 84.66 %   ← A3 == A4 to the question
McNemar baked-vs-A3: 11↔6 discordant, p≈0.33 (not significant).
**Pre-registered branch lands: A3 ≥ 81 ⇒ ENGINE SHIFT, not lean penalty.**
The −1.5 pp gate delta tracks the v0.30/fp8-KV era stack (~3 pp on this
anchor set, statistically weak), NOT the marker-penalty bake: identical engine,
opposite bake verdicts, zero separation. Decision: **prod.conf (lean-hyb K4)
stands as the product**; the gate bar FAIL remains as-measured on 198
(81.3 vs 82.8) with the caveat that its cause is engine numerics and the
0.31-era upgrade (native #55557) is now the quality-side lever.
Gap-fill (a3retry) may add up to 35 ids; verdict re-check on its close.

## CTX arm — pre-registered expectations (2026-09-30, runs tonight)
The 1M-retirement verdict (09-29, commit "262K native for the gx10 pair
recipe") was INFERRED (acceptance-position collapse + KV math), never measured
end-to-end. The context A/B arm now measures it on the production stack:
needle retrieval + prefill TTFT + prose decode + per-position acceptance at
60k/200k/500k/950k, native (262K) arm vs YaRN-4.0 (1M) arm.
Expectations written BEFORE data:
  - YaRN arm: recall stays high only while acceptance tau degrades enough to
    cost decode; the 09-27 pair rows (mean acceptance 2.8-3.3 at long ctx vs
    ~3.5-4.0 at short) predict YaRN loses speed, not needles.
  - If YaRN needles FAIL past 262K too, the retirement was right for TWO
    reasons and the question closes.
  - Adoption rule: raise prod context (131K -> 262K) only on a needle PASS at
    250K + decode within 10 % of the 131K row; keep 131K otherwise.

## Field row 2026-09-29 21:2x — the c=8 thrash (gx10, caught live)
User's still-running 1M-YaRN boot at 10 streams: Engine log 155.3 tok/s ->
12.6 -> 11.2 across 30 s with `GPU KV cache usage 92.4 -> 97.6 %` and
`Waiting: 1` — pool saturation + chunked-prefill stealing decode steps:
classic KV-thrash oscillation, not a scheduler bug. Compounding factors,
each measured on our own lane: YaRN seats demand fragmentation headroom the
fp8-KV pool doesn't have; MTP per-position acceptance collapses at long
positions (drafted 18 tok/s -> 8 accepted = half the draft bandwidth spent
for nothing). Fix = the recipe cutover (262K/131K native, no YaRN, prod
stack), staged on the pair 09-30: checkout at fork tip, `.env RECIPE` was
still `lean-stock` (stock checkpoint, old memory model — NOT the product
stack), corrected to `prod`; 120 GiB `q38-lean-hyb` verified byte-present on
both nodes. Awaiting the user's stop/start moment; if thrash persists AFTER
the swap at >95 % pool, reopen as a budget bug.

## 09-30: the box-freeze class — RESOLVED: pinned-PLE boot, not power/thermal/batch
Six freezes (16:50 + five on 09-30) all traced to ONE export bug: the chain's
first arm (depth sweep) exports KV_FP8+MAXLEN but NOT PLE_MMAP, so launch_v30
falls back to native PINNED PLE offload. The pool guard caught it at 5 s
resolution (guard.log): weight load ramps 98->31 G available over ~4 min
(normal lazy read), then the pinned-table phase slams 31->0 G in ONE tick,
driver NV_ERR_NO_MEMORY (the #56824 curve, reproduced). Rescue kills fired
3/3 and the box SURVIVED each at ~4 G available; thermal zones stayed 50-60 C
(power/thermal theory EXONERATED); every arm that sets PLE_MMAP (R1/R2/R5/CTX/
R7/a3retry) never collapsed once. Fixes: PLE_MMAP added to t1b/r3/r4/a6_k/r6
(r3 also lacked KV_FP8 entirely: bf16 KV + pinned = double hit); the morning
batch-size theory was wrong (2048 kept anyway: it is the banked-comparable
setting). Instruments kept: gb10_guard (kill at 6 G + 5 s forensics), gated
@reboot selfheal (sync-then-rearm, throttle 2/h), WoL wake from gx10 (the
10:31/11:40 collapses still hard-hung the box — the guard's kill beats the
kernel's, but a 0 G tick can outrun it; power cycle/WoL remains recovery).


## CTX partial 09-30 14:2x — native arm data + the two harness bugs found
native (262K seat, prod stack, 2048-batch):
  60k: needle 3/3 PASS, TTFT 28.9 s (2,078 tok/s cold prefill), prose 21.3 tok/s
  200k: engine DIED mid-cell — the pool guard FIRED (rescue 1, avail 5 G):
        the #56457-class indexer-workspace cliff reproduced ON OUR STACK AT
        ~200k prefill EVEN WITH SPARSE_MAX_LOGITS_MB=256 (guard curve: avail
        bleeds 17->8 G over ~12 min of chunked prefill, monotonic — the
        fragmentation growth pattern #57105 fixes, and that fix is 0.31-only).
        Guard killed the engine at 5 G and the box survived. OPERATIONAL
        CONSEQUENCE: deep prefills (~200k) are currently unservable on v0.30
        on this box regardless of seat size — until ctx2 repeats it on the
        YaRN arm and R1's 100k cell stays clean, treat ~100-130K as the
        safe prefill ceiling (prod.conf's 131K sits exactly under it).
  500k/950k: REFUSED-NATIVE (expected; logged in ctx_verdict.txt)
yarn 1M: BOOT-TIMEOUT was a pydantic max_model_len refusal (launcher lacked
  VLLM_ALLOW_LONG_MAX_MODEL_LEN — start.sh has it, bench launcher never did).
  Corrected rerun (ctx2: ALLOW_LONG, 90-min window) + t1b telemetry rerun
  serialized on standby; they fire when chain_r2 completes.
T1b all five K arms booted+benchmarked (rc=0) but acceptance dumps = zeros:
  probe called --only (decodebench never had it; --tasks prose is real) —
  argparse exit swallowed by || true. Speed ladder survives intact
  (23.5/23.9/20.9/20.2/11.9); EV needs the per-pos dump → t1b2 tonight.
Implication if ctx2's yarn@200k also guard-fires: 1M-YaRN is not just slow,
it is UNSERVABLE at deep prefill on this engine+box — the retirement
verdict would upgrade from inference to measured-crash evidence.

## v31-lane L0 result (09-30 17:2x, source-tested, no boot yet)
All five patchers handled against v0.31.0rc2 source on the worktree branch
(local, per single-branch law until measured): fp8-KV = native on rc2,
backport refuses as predicted, start.sh version-gates the mount
(KV_PATCH=1|native|auto, gate matrix 19/19); PLE mmap = re-anchored onto
common/ngram_embedding.py — every semantic anchor survives there (fp8
process_weights_after_loading byte-identical), the shard weight_loader loop
lives in the nvidia shim and is patched too; fp8dense trio converted to an
anchor-checked patcher (fusion gate preserved, drift refuses); draft-vocab +
fused-draft re-anchored, compose clean incl. stacked fp8dense→draft-vocab.
v0.30 tests still green, recipe_lint PASS (new KV_PATCH/IMAGE_SERIES keys).
Remaining before any L1 boot: the notes/gitignore polish + an rc2 image that
exists (build it ourselves or wait for the v0.31.0 artifact — decision then).

## #59432 zero-fill skip: NOT free on our stack (image-verified 09-30 19:3x)
The upstream rationale ("caches that read-before-write keep their zeros via
needs_kv_cache_zeroing") inverts for us: our config has 36 gated-delta-net
layers → `has_mamba_layers` True → `needs_kv_cache_zeroing` True on every
boot — the startup memset is load-bearing here, not slack. The only way the
backport pays is block-scope (zero the mamba/ring groups, skip a uniform-
precision attention group), which is finer than the issue's platform flag.
VERDICT: parked, evidence recorded; the boot-pool win we already took is the
guard + mmap PLE. (If a future image narrows the flag per-group, reopen.)

## Gate verdict RE-CHK after gap-fill (2026-09-30 19:42, rc=0, +23 ids)
Common-answer set grew 163 -> 186 ids. On it:
  baked (lean, old image)   161/186 = 86.56 %
  A3 (hyb,  v0.30 stack)    155/186 = 83.33 %   own valid n=186 (fully filled)
  A4 (lean, v0.30 stack)    154/186 = 82.80 %
The decisive pair is A3-vs-A4 (identical engine, opposite bake): 155 vs 154,
McNemar 10-vs-9 discordant, p≈1.0 — the lean bake's gate cost is ZERO, now on
a 14 % bigger anchor set. The baked-vs-A4 gap (−3.8 pp) remains the
engine-era shift (baked-vs-A4 pair: 11-vs-4, p=.118). Verdict UNCHANGED and
strengthened: prod.conf stands; quality lever = the v0.31 ladder, not a
lean retune. tool: tools/gate_reanchor.py (replays this from the scored
files; sign test reproduces the banked .33).

## R7 dynamic arm: DSD-on-MTP confirmed dead on v0.30 (boot-fail evidence, 20:2x)
The concurrency-scheduled-depth boot died at CUDA-graph capture, every time
(three identical traces): v1/worker/gpu/cudagraph_utils.py:777
prepare_inputs_to_capture -> input_batch.py:129 make_dummy ->
`assert 0 < num_reqs <= num_tokens`. Same family as upstream #58692 (DSD +
MTP on the V2 runner crashing the speculator's capture sizing) — our config
reproduces it as a dummy-batch zero-token tier (the K=0 range and/or the
derived draft query lengths), pre-registered outcome "boot-fail IS the
answer (R4 precedent)". DSD adoption for MTP: impossible on v0.30, awaiting
upstream capture fixes (#49652/#56136 open). The paired static boot proceeds
regardless — it is our first true 32-seat census row (G4 baseline under the
new image guard), and R7's band is decided: NO (engine refuses).

## Context verdict, v0.30 side (ctx2, 09-30 22:58): 200k prefill PASSES with the #57105 backport
The registered rule was "native@200k passes needles + flat pool → 262K is
shippable, ship cap-64". It fired. With SPARSE_MAX_LOGITS_MB=64 AND the
#57105 worst-case-workspace backport mounted (QSA_RESERVE=1):
  60k:  needles PASS, TTFT 28.2 s (≈2,130 tok/s prefill)
  200k: needles PASS, TTFT 84.3 s, decode 28.0 tok/s — guard never fired,
        the whole arm ran clean where run-1 bled 17→8 G and died at cap-256
  500k/950k: refused by the 262K cap (by design; that refusal is data)
Attribution: two knobs shipped together (smaller chunks + single worst-case
reservation); the fragmentation mechanism is #57105's own, cap-64 was the
issue's tested lever. v0.31 carries #57105 natively — another L1 argument.
YaRN-1M arm boots at 23:0x; cells follow (~3 h). The 262K bump for prod
lands AFTER the yarn cells + r1b confirm, with the engine-cap note updated.
R7 census note: the 32-seat rows came back zero-token (late-first-token
streams outliving the bench window under pool pressure) — the row is NOT
the concurrency answer; r1b's pre-death census attempt + a redo decide it.
R6 (DeepGEMM): 6/10 text-identical, logprob max|Δ| 0.0065 — review tomorrow;
null-text rows and truncation need disambiguating before any corruption claim.
