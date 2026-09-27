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

## Queue state at write time
A1 banked; A2 (k=4 spec stack) riding; A3/A4+gpqa200/P1/T1 queued under
ChainSupervisor (manual-ride protocol). Rows land here as commits.
