# STATE-SINCE-FORK — full change inventory + reporting ledger (2026-10-01)

Every commit below is mine on top of the repo baseline (last upstream-side
commits land 04/05-09; my series starts 2026-09-04). `git log` in this repo
carries the verbatim list (83 of 142 commits are mine; the 2026-09-29
series is history-reworded, tree-identical). This page groups and answers.

## A. Engine / serving changes (what runs on the box)
1. **Lean bake (A2 lineage)** — 12 % vocabulary-restricted SFT bake of the
   base checkpoint; the A3/A4 gate says quality-neutral on GPQA
   (155 vs 154, p≈1) and it doubles as the INT8/4-bit enetq test vehicle.
2. **FP8-KV overlay (`fp8kv.py`)** — int8 KV cache path; the R7s row ran on
   it.
3. **MTP draft-vocab patcher (`draft_vocab.py`)** — reduced-vocabulary
   drafting, +8.4 % mean banked in repo before this program.
4. **Fused-MTP driver + fused-draft arm (R2, rc=2 both passes — unbanked).**
5. **QSA geometry arm / sparse-indexer logits cap knob**
   (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, default 512; pair recipe ships 64)
   — measured: 200 k chunked prefill bleeds the unified pool 17→8 GiB
   through this knob's cap; fragmentation class upstream #56457/#57105.
6. **#57105 backport patcher** (`qsa_patch/57105.patch`, mount flag
   QSA_RESERVE=1) — worst-case workspace reservation, the deep-prefill fix
   upstream took for 0.31; native-200k then passed (needles 3/3, 28 tok/s
   decode, pool flat).
7. **fp8dense patcher** — anchor-checked, 19/19 gate matrix.
8. **GEMM A/B v2 probe** — backend numerics; verdict: v1 under-instrumented,
   6/10 not loadable, backend OFF by default until greedy re-run.
9. **Dynamic speculative-deepness** — shipped-in-image (config-only flag
   `QSA_DYNAMIC`); R7 dynamic boot asserted at capture (upstream MTP+dynamic
   family) → static-only today.

## B. Recipe / launcher
- `RECIPE=prod` in both clones; `.env` on the pair ships 131 072 (was
  stock-32 k mismatch found 29-09); `start-tp1` defaults to the fork
  checkout; launch_v30.sh gained ALLOW_LONG=1, EAGER/CAPTURE_SIZES,
  PLE_MMAP contract, `--port` enforcement; page-cache release before boot
  (the documented CUDA-OOM trigger if skipped).
- v0.31 lane: `v31-lane` branch (worktree wt-v31), source-level
  upgrade-readiness commit da07f04; never booted/benchmarked.

## C. Bench-harness fixes (all of 2693cdf/01fef36/c730c96/… ~25 commits)
The honest summary: the program's failures in 09-29→10-01 were overwhelmingly
MY driver bugs, all now fixed and each with a validator rule:
missing `--port` (four arms), wrapper-pid health loops (two arms), missing
PLE_MMAP export (five arms — root cause of every chain death since 09-29),
8192-batch freeze, health window < JIT boot time, telemetry probe using a
nonexistent flag (t1b acceptance = zeros), census grepping
content/reasoning_content instead of **reasoning** (this is the c≥8
"all-failed" class — the 09-27 all-fail, r7s and r1b censuses were THIS),
stale chain_r collision, log-destroying arms.
**Consequence you flagged:** "c10 not managed" — every c≥8 census in my
files was the harness's bug (delta-key), i.e. NOT an engine verdict; the
engine-era honest numbers for c≥8 are the pre-fix lean_hyb C=16 row (n_ok=2/5)
and c=8 rows (0.671 collapse). Verdict file: `docs/verdicts/` V30-LANE.md.

## D. Experiment arms run (and their status)
A1/A2 bake rows ✅; A3 gate ✅ + gapfill (rc=0, 19:42) → verdict 155-vs-154
p≈1; A4 ✅; T1 depth ladder (t1b2 rerun 00:58); K-retune r1c (02:58–05:02):
acceptance rises past k=3, k=4 worst legal point, k=3 banked, user-locked k
deferred to you; P1/P1b ✅ (indexer 10.6 % → passes the 8 % bar); R1/r1b cold
40 min; R2 ✗rc=2 (both), R3/R4/R5 ✗rc=2 (driver), R6 dg=True 30/30 vs False
0/30, R7 dyn boot assert @capture, R7 static row banked; ctx arms: native 60 k
✅, 200 k ✅ with backport, YaRN arms died on launcher bugs (redo queued);
CyberGym 10-task subset ✅; **bench_trio** (tau2-banking/AutomationBench-25/
MultiChallenge wired to our endpoint) — committed 35e1188, NEVER RUN.
R10 (adaptive draft depth): **never existed as an arm** — only the K-sweep;
that is the plan you remember as "draft-exit/R10"; it is not in any verdict.

## E. Infra / safety on the bench box
PLE pool-guard (kills engine <6 GiB avail; 43 kills logged; zero since
05:24 10-01), selfheal@23:50+rc=3 re-arm, arm validator (pre-flight refuse
of the known bug classes), standby queue (self-audits: redoes rc=2 arms),
WOL+scheduled-boot. No changes to gx10 prod files; msi is bench-only.

## F. Reporting ledger — you (user) reported → status/evidence
1. **"τ² 36 % is bad; base ~20 not worth; 88 %-tier model is the bar"** —
   the τ²/AB numbers are **mine, my own harness, offline scorer**; the fix
   (committed, not run): **bench_trio** runs the public offline variants
   near-exactly; the honest read is lean-bake cost tau/AB something the
   GPQA gate does not see; GPQA says quality-neutral. **Not closed:
   τ² re-run (my harness) + the trio.**
2. **"AB 41 vs base 55.9"** — same instrument caveat (my eval harness);
   same trio answer. Not closed.
3. **"c=10 not managed"** — all my c≥8 census rows = harness bug (§C);
   engine c8/c10 answer is **open** (needs one bench run).
4. **"Strata, TensorFold, new papers"** — TensorFold: merged #4 only
   (09-29), already shipped (lang-draft-vocabs = draft_vocab patcher).
   Strata: updated 09-30 19:34 (0.1.30, chunked expert streaming,
   conversation cache, rope-scaling) — it's a standalone engine, **no
   integration path into vLLM/our fork; banked, zero action.** RAZOR:
   research-only (expert pruning for our arch; the paper's no fine-tune
   <512 → 512-2k regime is unproven for us). **WhenToThink: committed
   nowhere, run nowhere — I have no trace of implementing or measuring
   this; if I said it, I was wrong. Flag as open.**
5. **"Not faster"** — true for the K-census era (all rows predate 09-29
   fixes) and the c≥8 rows are invalid (C); honest current answer: single
   lean-hyb 23.5→25 tok/s at c=1, c≥9 unknown, static R7s row banked.
6. **"1 M/262 K fiasco"** — retired 09-29 (1 M experiment, not a fix); the
   200 k crash root-caused to fragmentation (#57105 class) and fixed via
   backport mount; **the ship-gate run of the exact prod shape is tonight's
   queue head** (ctx4 arm) — that's the live thread, not a fix-impl.
