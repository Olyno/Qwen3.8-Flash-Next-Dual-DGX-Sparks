# Load time

Every experiment pays this: a boot is 12–13 minutes, and almost all of it is weight loading. This file records what that time is spent on, what did not work, and what the default now is.

## The checkpoint and the ceiling

| | value |
|---|---|
| on disk | 124 GB, ext4 on `/dev/nvme0n1p2` |
| tensors | **299,545** across 11 shards |
| tensor kinds | `weight`, `weight_scale`, `weight_scale_2`, `input_scale` — four per expert per layer |
| resident after load | 77.04 GiB |
| raw read, direct I/O | **3.4 GB/s** (512 MiB in 0.156 s) |
| reported load | **713–727 s** → 170 MB/s effective |

The disk could deliver the whole checkpoint in about 40 s. The gap is not the filesystem.

## Where the time goes

Sampled during a boot with `vmstat` and `ps`:

- `VLLM::Worker_EP` at **101% CPU** — exactly one core saturated.
- iowait **0**, `/proc/pressure/io` `some`/`full` both **0.00**.
- The rest of the machine idle, `UVM GPU1 BH` at 11%.

So the loader is **CPU-bound**: roughly 2.4 ms per tensor (727 s ÷ 299,545), not I/O-bound. That single measurement invalidates the obvious fixes — parallelising reads helps nothing, and it says where a real fix has to act: fewer tensors, or more cores inside the per-tensor loop.

It also explains a hazard: any loader that buffers whole shards on this device competes with the 77 GiB already resident in 121 GiB of unified memory.

## Arms measured

| loader | load | KV after boot | 18-task ruler | verdict |
|---|---|---|---|---|
| `safetensors` + `lazy` (previous default) | 713 s, 727 s, 683 s | 4.25–5.32 GiB | 15, 15, 14, 13, 13 | working baseline |
| `runai_streamer` | **281 s, 326 s** | **8.54–10.21 GiB** (388,174 tokens) | 15, 14 | **adopted** |
| `fastsafetensors` | — | — | — | **rejected: rebooted the host** |

- **`runai_streamer` is 2.2× faster** on the same checkpoint, and the streaming phase itself runs at 8,000–28,000 tensors/s across two cores where the default does ~400/s.
- **It also leaves more memory for KV** — 8.5–10.2 GiB against 4.3–5.3 GiB, i.e. 388,174 tokens of cache instead of ~180–240k. Available KV is computed as `total × utilization − peak_allocated`, so this is the streamer retaining fewer host buffers, not loading fewer weights (77.04 GiB in both cases).
- **`fastsafetensors` took the machine down.** It bulk-copies shards on the device; with 77 GiB of weights resident and a largest shard of 51 GB, the host stopped answering ssh within two minutes and came back after a reboot. No load-time number is worth that.

## Does a loader change the model?

It must not, and this is where the measurement itself had to be fixed first.

The KL yardstick compares teacher-forced distributions against a baseline capture, with a "self-noise floor" taken from two live passes of the same weights. On this stack that floor is **0.909–0.913 top-1 agreement**: two passes of the *same* model disagree on ~9% of positions through batching and kernel nondeterminism alone.

Two consequences, both now handled:

1. **The default gate was unattainable.** It demanded 99.5% top-1 while the metric's own floor sits at 91%, so it failed identical models — including a replay of a capture against the endpoint that produced it. `bench/kl_yardstick.py` now relaxes the gate to the floor (minus two points of pass-pair slack) and says so in its output (`cc57053`).
2. **Greedy text is not reproducible either.** The same boot, same prompts, temperature 0, twice: different output. So "identical bytes" is not an available test here; distribution distance and task scores are, and both place `runai_streamer` inside the noise band (mean KL 0.123–0.131 against a floor of 0.118–0.143).

The honest summary: the loader transports the same files and the model's behaviour is unchanged at every resolution we can measure. It is not bit-identical proof, and nothing on this stack can currently provide that.

## What is left

The remaining ~300 s is still one core walking 300k tensors. The levers that would move it, none tried yet:

- **Fewer tensors**: pack the four per-expert tensors into one, cutting the loop ~4×; requires a repacked checkpoint and a loader that understands it.
- **`sharded_state`**: vLLM's own pre-sharded format, which exists to skip the name-by-name mapping.
- **Don't boot**: the engine supports `enable_sleep_mode`, so a process can be parked and woken instead of restarted. For experiment iteration that is worth more than any loader.
