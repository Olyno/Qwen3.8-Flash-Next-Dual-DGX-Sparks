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
- gpqa200 G1-gate (prod.conf pass/fail) running, ETA ~00:30. P1+T1 chain armed
  unattended (chain.sh on box, 7 h ceilings, panic-kit live).
