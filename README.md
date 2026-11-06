# Qwen3.8-Flash-Next on DGX Spark

Serve [Qwen3.8-Flash-Next](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
on a single DGX Spark, or across a 2-node cluster, with vLLM: expert
parallelism, MTP speculative decoding, PLE offload, up to 262K native context.
Topology is `NODES=1|2` in `.env`; TP follows the node count.

Forked from [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
(AGPL-3.0-or-later), itself based on
[getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark).

## Layout

```
start.sh          thin entrypoint: recipe → engine/* → boot
stop.sh           tear down the server (and the worker, when NODES=2)
download.sh       fetch HF weights onto the head (HF recipes only)
engine/           the boot pipeline, one file per step (config, cache, pair,
                  verify, patches, prepare, args, launch) + recipe.py loader
recipes/          serving config: prod.yaml (default) and mia.yaml (vendor reference)
patches/          runtime patchers applied to the vLLM image (bind-mounted)
overlays/         opt-in vLLM overlays (fp8dense, qsa_gb10, qwen4_exp)
vocab/            reduced MTP draft vocabularies
scripts/          operator tools (check-weights.sh, verify-weights.py, nfs-share.sh, ...)
bench/            benchmarks
tests/            CPU-only test suite
docs/             notes
```

Config is split in two, on purpose:

- **`.env`** — machine truth only: `NODES` (1 = single Spark, 2 = head +
  worker), cluster IPs, interface, InfiniBand, secrets. Copy `.env.example`
  and edit. Never holds serving knobs.
- **`recipes/<name>.yaml`** — the complete serving config: model, image, KV
  dtype, MTP, ports. A recipe wins over `.env` for every key it sets, so a
  stale `.env` value can never silently take effect. `RECIPE=<name> ./start.sh`
  selects one; the default is `prod`.

| Recipe | Model | Image | Use |
|---|---|---|---|
| `prod` (default) | local `~/models/Qwen3.8-Flash-Next-NVFP4-lean`, published as [Olyno/Qwen3.8-Flash-Next-NVFP4-lean](https://huggingface.co/Olyno/Qwen3.8-Flash-Next-NVFP4-lean) | `vllm/vllm-openai:v0.32.0` | our serving config |
| `mia` | `nvidia/Qwen3.8-Flash-Next-NVFP4` (HF cache, stock weights) | `vllm/vllm-openai:qwen38-flash-next` | vendor reference, day-0 defaults |

All new optimizations land in `prod.yaml`.

## Prerequisites

- One DGX Spark (NODES=1), or two wired over ConnectX with passwordless ssh
  from head to worker (NODES=2). Docker on each.
- `.env` filled in (`cp .env.example .env`): `NODES`, `HEAD_IP`, and for
  NODES=2 also `WORKER_IP`, `IFACE`, `IB_HCA`, `IB_GID_INDEX` — see the
  comments in the file.
- For `prod`: the local checkpoint at the recipe's `model_path`
  (default `~/models/Qwen3.8-Flash-Next-NVFP4-lean`) on the head.
- For `mia`: `./download.sh` once (fetches the nvidia checkpoint onto the head).

## Quick start

```bash
./start.sh                 # default: patch → launch (syncs weights to the worker first when NODES=2)
RECIPE=mia ./start.sh      # vendor reference: download (if needed) → sync → patch → launch

curl http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"Hello"}]}'
```

`./stop.sh` stops the server (and the worker, when NODES=2).

## Flags

```
./start.sh --no-download  # skip HF download (weights already cached on head)
./start.sh --no-launch    # download + sync only, don't start vLLM
./start.sh --launch       # skip download/sync; patch + launch
./start.sh --nfs          # share the head cache over NFS instead of rsync
./start.sh --no-nfs       # force rsync even if the recipe sets nfs_share
```

## Local checkpoints (`model_path`)

A recipe with `model_path: <dir>` serves a plain checkpoint directory — no HF
repo, no download. The dir is verified (every shard in the safetensors index
must exist) and bind-mounted at `/model` in the container (on NODES=2 it is
also rsynced to the worker's `~/models/<name>` once). `model_path` is mutually
exclusive with `nfs_share` and `ABLIT=1`.

## NFS weight sharing (optional, HF recipes, NODES=2 only)

Off by default: each node keeps its own checkpoint copy, rsynced from the head
once. With NFS the head exports its HF cache over ConnectX and the worker
mounts it read-only — no worker copy.

| | rsync (default) | NFS |
|---|---|---|
| Worker disk | ~126 GiB | none |
| First launch | one full copy, then free | no copy |
| Every cold start | reads local disk | streams ~126 GiB over ConnectX |
| Head must stay up | only to launch | **for the whole serving run** |

Enable per run with `--nfs`, or `nfs_share: "true"` in a recipe.
`./stop.sh --nfs` also tears the share down. **Do not stop the `vllm-fn-nfs`
container while vLLM is loading or running.**

## Abliterated checkpoint (`ABLIT=1`)

`ABLIT=1 ./start.sh` serves the gated Keys house QSA L3–47 checkpoint
([drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47)).
Safety refusals are removed; MTP, PLE, experts and the chat template stay
stock. Requires `HF_TOKEN` in `.env` and accepting the terms on that page,
then `ABLIT=1 ./download.sh`.

## Scripts

| Script | Purpose |
|--------|---------|
| `download.sh` | fetch weights onto the head (`ABLIT=1` for the gated Keys snapshot, `--fp8` for official FP8, `org/repo` for anything else) |
| `start.sh` / `stop.sh` | boot / tear down the pair |
| `scripts/check-weights.sh` | presence + size on both nodes; `--verify` hashes every file against the HF manifest, `--dry-run` plans, `--manifest FILE` verifies offline |
| `scripts/verify-weights.py` | the per-file SHA-256 verifier behind `--verify` |
| `scripts/memwatch.sh` | kill the container if host MemAvailable runs low |
| `scripts/build_draft_vocab.py` | build a reduced MTP draft vocabulary from a corpus |
| `bench/` | decode/prefill/long-context/MTP-acceptance benchmarks |

## Measured

Single box, `tensor_parallel_size = 1`, the committed recipe. Each row is reproducible from `bench/`, and each is written up in [`docs/serving-experiments.md`](docs/serving-experiments.md) with its method, its scope and its negative results.

| What | Result |
|---|---|
| Decode, prose | 27.8 / 30.2 / 28.1 tok/s at 1k / 32k / 65k |
| Decode, code | 37.0 / 39.7 / 38.6 tok/s |
| Decode, copy-heavy | 61.7 / 65.7 / 62.0 tok/s |
| Decode step rate, copy | 13.5-14.2 steps/s at 1k (70-74 ms/step), the row that repeats to 2% |
| Real session, 24 spread turns (every prefix cold) | **13.4 s per turn** = 9.7 s prefill + 3.7 s decode (321 s, 301k prompt tokens); was 23.0 s before the prefill prefetch shipped |
| Real session, 8 consecutive turns | **2.5 s per turn warm** (95% of the prompt cached, prefill 0.47 s) against 6.8 s on a cold first pass |
| First prompt of a new content region (before the prefetch) | +13.5 to +24 s once (490-714 tok/s against a 2,271 tok/s steady cold rate); five regions in one boot each paid it. With the prefetch, the same 40-turn window has no cold turn above 7.9 s |
| Real session, 24 never-sent consecutive turns | **7.1 s per turn** = 3.5 s prefill + 3.6 s decode (170 s, 320k prompt tokens, 46% cached); one first-touch turn at 11.8 s prefill |
| Real session, 40 consecutive turns | 9.6 s per turn; **6 turns paid the cold penalty, 28% of the wall**; at the steady cold rate the same forty would cost 275 s instead of 383 s |
| Cold prefill with the prefetch on (same 40-turn window) | wall 284.1 → **233.3 s**, prefill −22.5%, the three 250-1,000 tok/s turns 45.6 → 20.7 s, worst prefill 17.1 → 7.9 s, major faults 121k → 40k |
| TTFT | 0.53 s / 11.3 s / 22.9 s |
| Concurrent streams | 59.9 tok/s aggregate at 4 streams, 95.1 at 8 |
| Prefix cache, repeated prompt | 98.9% of the prompt served from cache |
| Prefix cache, five-turn session | 64% overall, 80% by the fifth turn |
| Tool use, 12 scored tasks | 10 of 12 |
| Reasoning and long-context, 12 tasks | 12 of 12 |
| Knowledge, 600 questions | 31.7% accuracy, 63.9% hallucination |

Read the prose row with care: on identical content it has swung between 15.5 and
25.5 tok/s across runs — including 15.5 then 19.0 on two consecutive runs of the
same boot — with step rate moving by a third. Copy repeats to 2%, so a
configuration claim rests on copy steps/s or on a paired A/B, never on a single
prose sample.

Every shipped mechanism has its source recorded in [`docs/provenance.md`](docs/provenance.md), including the work that was evaluated and deliberately rejected.

Two-rank numbers are deliberately absent, and they are not extrapolations of these: step time, TTFT and concurrency all change with the second rank, while token counts do not. `docs/serving-experiments.md` states which side of that line each measurement sits on.

## How this recipe compares to the vendor one

`recipes/mia.yaml` is the vendor contract (MiaAI-Lab) shipped as-is; `recipes/prod.yaml` is this repo's default. What differs, and what the differences buy, in two tables — configuration first, then measured effect.

| | `mia` (vendor contract) | `prod` (this repo) |
|---|---|---|
| topology | 2 nodes, TP=2 | 1 node, TP=1; at `NODES=2` the same recipe derives 262,144 context, TP=2 and 8 seqs |
| checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` | local NVFP4 lean bake |
| `max_model_len` | 262,144 | 65,536 |
| `gpu_memory_utilization` | 0.80 | 0.72 |
| `max_num_seqs` | 8 | 4 |
| MTP drafts | 3 | 4 |
| PLE n-gram table | resident on device | host mmap over ATS |
| FP8 draft head | off | **on** |
| FP8 target head | off | **on** |
| reduced draft vocab (47k) | on | on |

Measured on this box as A/B against the same recipe with the mechanism off, so the deltas hold regardless of topology. ms/step is the currency: under speculation `tok/s = steps/s × tokens/step`, and tokens/step moves with the *text*, not the configuration.

| Change over the vendor contract | Measured | Written up in |
|---|---|---|
| FP8 target lm_head | **−4.0% ms/step** on copy (14.2 against 13.5-13.7 steps/s, no overlap); p99 KL 1.660 under a 1.730 self-noise floor | [serving-experiments](docs/serving-experiments.md) |
| FP8 draft head | **−9.3 to −10.1%** ms/step at 1k, −4.9 to −9.3 at 32k, −3.1 to −5.7 at 65k | [bench-gates](docs/bench-gates.md) |
| Adaptive verify width | **−3.1 to −4.1%** ms/step in every bucket | [bench-gates](docs/bench-gates.md) |
| Draft head read per draft step | **1.18 GiB → 0.22 GiB** with the 47k slice | boot log, [serving-experiments](docs/serving-experiments.md) |
| `runai_streamer` loader | load **281-326 s**, against 683-727 s for the previous loader | [load-time](docs/load-time.md) |
| PLE prefill prefetch (one-box only) | cold-prefix replay **23.0 → 13.4 s per turn**; mean prefill 19.4 → 9.71 s (647 → 1,294 tok/s); worst cold prefill 17.09 → 7.93 s; major faults 121,360 → 40,026 on the same content | [serving-experiments](docs/serving-experiments.md) |
| PLE prefill prefetch reader pool (16 → 32) | a 13k never-sent prompt **51.85 → 8.42 s** on the shipped default (1,574 tok/s, 69.0 MiB/s of PLE read), faults 98,132 → 3,565 | [serving-experiments](docs/serving-experiments.md) |
| QSA draft-metadata overlays (fused multi-step draft metadata + free-list block-id sort) | `short_code` **38.2-39.4 → 48.4 tok/s** (+25%) and `mixed` 39.2 → 44.7-45.1, at unchanged copy steps/s; booted off, the engine falls back to rebuilding metadata eagerly. The two are flipped together, so the gain is not attributed individually | [serving-experiments](docs/serving-experiments.md) |

The last row is a debt this recipe pays and the vendor one does not: at one node the
PLE n-gram table has to live off-device, so every genuinely new prompt faults its
rows off disk. The prefetch is the remedy inside that constraint; the two-rank
shape keeps the table on device and has no such cost to remove.

Two things this table cannot say, and does not pretend to. It cannot give a vendor-vs-ours **absolute** throughput row on the vendor's own topology, because their figures are two-rank and their recipe OOMs at one node — the two-rank record stays absent until the pair is measured. What it can do, and now does, is reproduce the vendor contract's *settings* on this lane at one node and measure against that:

| Condition | `prod` | Vendor settings on this lane | Difference |
|---|---|---|---|
| `short_code` | 48.4 tok/s | 40.3-43.8 | +10 to +20% |
| `long_prose` | 26.5-26.6 | 24.1-24.5 | +8% |
| `mixed` | 44.7-45.1 | 33.6-36.5 | +23 to +34% |
| decode 1k, prose | 26.8-27.8 | 25.6 | +5 to +8% |
| decode 1k, code | 35.6-37.7 | 37.4 | parity |
| decode 1k, copy | 63.5-64.8 | 54.7 | +16 to +18% |

The arm flips seven keys to the vendor values (MTP depth 4→3, no FP8 draft or target head, no adaptive verify width, no prefetch, no fused draft metadata, no free-list sort) and changes nothing else; booted 509 s, health 200. Copy loses purely on tokens/step (3.87 against 4.5) because the k=3 chain caps it, while prose gains tokens/step but pays per-step latency once the overlays are removed. Every number above is one box, one lane, one checkpoint — not the vendor's deployment.

The upstream repo also publishes its own single-node figures (`docs/HANDOFF-single-spark.md` §0: one Spark, port 8888, MTP 3, temperature 0), which is the only same-topology row in the baseline that needs no reproduction here:

| Single node, temp 0 | this recipe | upstream TP=1 record |
|---|---|---|
| prose | **26.8-27.8 tok/s** | 24.9 tok/s |
| code | **35.6-37.7 tok/s** | 35.1 tok/s |
| KV pool | 446-481k tokens at `gpu_memory_utilization` 0.72 | 685,825 tokens at GMU 0.813 |

So the recipe leads the baseline on its own single-node ground, on a smaller memory budget. It is not a controlled A/B, and the difference runs the honest way: that record is an earlier lane on the `local-inference-lab` checkpoint, whose PLE table is 4-bit (26.9 GiB) against the 8-bit 51.2 GiB table this box streams, so the baseline moved fewer bytes per step. Same topology, lighter weights, slower decode. The vendor's two-node row (prose 57.6 / code 71.5 at S=1) stays the number the dual deployment has to beat, and it stays unmeasured here until the second node is online.

Two rules hold the recipe to that table, and both are checked by booting, not by reasoning: **the shipped recipe must beat the vendor contract** (the arm above; code is the one condition at parity), and **every change must beat the recipe it lands on** (its own A/B, copy steps/s or tokens/step, never prose tok/s alone). A switch that cannot show a gain does not ship. That rule has already cost this log one correction: the QSA draft-metadata overlays were read as neutral and switched off, which cost `short_code` 48.4 → 38.2-39.4 tok/s; they were restored and boot-verified, and the copy-acceptance figure they were judged on turned out to be a high draw (reproduced spread 91.2-97.0 %). And a faster step is not automatically a better model: each mechanism above passed a distribution gate first (p99 KL inside the self-noise floor), and a mechanism that fails it is recorded as rejected in [bench-gates](docs/bench-gates.md) instead of shipping.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
bash tests/test_check_weights.sh
```

CPU-only; no GPU, image, or network needed (one suite wants `torch` and skips
without it).

## Credits

Full attribution — every upstream PR backport, ported overlay and vendored
file, with licenses — lives in [NOTICE](NOTICE); license texts that must ship
with copied code are in [licenses/](licenses). The short version:

| | |
|---|---|
| Upstream repository | [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) (AGPL-3.0-or-later) |
| Base deployment (via the above) | [getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark) |
| FP8 KV cache kernels, draft-vocabulary builder | [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) (AGPL-3.0-or-later) |
| FP8 draft head, determinism knobs | [sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark) (MIT) |
| QSA fused draft, RoPE clamp, loader cache drop | [myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe) (MIT) and its Apache-2.0 vLLM fork |
| vLLM PR backports under `patches/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0) — [#53388](https://github.com/vllm-project/vllm/pull/53388), [#53899](https://github.com/vllm-project/vllm/pull/53899), [#57128](https://github.com/vllm-project/vllm/pull/57128), [#58040](https://github.com/vllm-project/vllm/pull/58040), [#58449](https://github.com/vllm-project/vllm/pull/58449); authors in NOTICE. Backports superseded by native vLLM v0.32.0 delivery were removed |
| FP8-KV approach (via MiaAI-Lab) | [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX) (Apache-2.0) |
| Concurrency / prefill benchmarks | [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) |
| Reduced draft vocabulary technique | [FR-Spec](https://arxiv.org/abs/2502.19797) |
| Omniscience benchmark data (`bench/data/`) | [ArtificialAnalysis/AA-Omniscience-Public](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public) (Apache-2.0) |
| Model | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| Abliteration splice (`ABLIT=1`) | **Keys (drowzeys)** — gated; house QSA `o_proj` L3–47 on nvidia NVFP4 |
| Decision proxy (`proxy/score_proxy.py`) | [autotrust/GEV-26B-Decide](https://huggingface.co/autotrust/GEV-26B-Decide) (confidence-gated escalation) and [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) (scoring prompt rules) |

## License

**AGPL-3.0-or-later** (this repository only) — see [LICENSE](LICENSE), matching
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
from which `patches/patch_qsa_fp8_kv.py` and `scripts/build_draft_vocab.py` are
vendored. Those files are AGPL-3.0-or-later, so the repository that carries
them has to be too. See [NOTICE](NOTICE) for the full attribution list.

vLLM remains Apache-2.0; the container image and the model checkpoint are
governed by their own upstream terms, and nothing here relicenses them. Files
that carry an `SPDX-License-Identifier` header keep the license of their
origin — see each file.

**The abliterated checkpoint**, served only when you opt in with `ABLIT=1`, is
gated on Hugging Face behind its own responsible-use agreement, with its
licence inherited from `nvidia/Qwen3.8-Flash-Next-NVFP4` (NVIDIA Open Model
License + Qwen Community License 1.0). This repository ships a flag that can
serve those weights. It does not redistribute them and does not relicense them.
