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
| `prod` (default) | local checkpoint (`model_path`, no download) | `vllm/vllm-openai:v0.30.0` | Olyno/Qwen-3.8-Flash-Next (prod) |
| `mia` | `nvidia/Qwen3.8-Flash-Next-NVFP4` (HF cache) | `vllm/vllm-openai:qwen38-flash-next` | vendor reference, day-0 defaults |

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

## Opt-in: ReplaySSM-GDN spec decode (`replayssm_gdn`)

v0.30 lane only. Ports the GDN variant of upstream
[vllm#47576](https://github.com/vllm-project/vllm/pull/47576): MTP verify
reconstructs each window from an fp32 checkpoint plus a small circular ring of
per-token d/k/g vectors instead of writing the full GDN state after every
accepted token; the checkpoint is rewritten only when the ring fills. The
mamba page grows to a 5-tuple that is ~2x smaller than the baseline spec page,
and block-keyed ring cursors live in the GDN metadata builder.

Requires MTP (`mtp_num_speculative_tokens > 0`, already the default) and
is mutually exclusive with `lazy_gdn`. `replayssm_gdn_buffer_len` (default 16)
is the ring history per block and must be `>= 1 + k`.

```bash
REPLAYSSM_GDN=true ./start.sh --launch   # or replayssm_gdn: true in the recipe
python3 bench/sweep.py                   # A/B against the stock page
```

Overlay mechanics: `patches/replayssm_gdn/apply_patch.py` rewrites 6 stock
files against extracted originals (fail-closed anchors) and mounts them plus
two vendored Triton files; everything is gated on `VLLM_REPLAYSSM_GDN=1` at
runtime. `tests/test_replayssm_gdn_v030.py` covers the patcher.

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
| K3 lazy GDN commit, FP8 draft head, determinism knobs | [sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark) (MIT) |
| QSA fused draft, RoPE clamp, loader cache drop, TP=2 skinny-GEMM rows | [myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe) (MIT) and its Apache-2.0 vLLM fork |
| vLLM PR backports under `patches/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0) — [#47576](https://github.com/vllm-project/vllm/pull/47576), [#53388](https://github.com/vllm-project/vllm/pull/53388), [#53899](https://github.com/vllm-project/vllm/pull/53899), [#55557](https://github.com/vllm-project/vllm/pull/55557), [#57097](https://github.com/vllm-project/vllm/pull/57097), [#57105](https://github.com/vllm-project/vllm/pull/57105), [#57128](https://github.com/vllm-project/vllm/pull/57128), [#58040](https://github.com/vllm-project/vllm/pull/58040), [#58114](https://github.com/vllm-project/vllm/pull/58114), [#58449](https://github.com/vllm-project/vllm/pull/58449), [#58957](https://github.com/vllm-project/vllm/pull/58957), [#58961](https://github.com/vllm-project/vllm/pull/58961), [#59632](https://github.com/vllm-project/vllm/pull/59632), [#59753](https://github.com/vllm-project/vllm/pull/59753), [#59945](https://github.com/vllm-project/vllm/pull/59945); authors in NOTICE |
| FP8-KV approach (via MiaAI-Lab) | [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX) (Apache-2.0) |
| Concurrency / prefill benchmarks | [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) |
| Reduced draft vocabulary technique | [FR-Spec](https://arxiv.org/abs/2502.19797) |
| Omniscience benchmark data (`bench/data/`) | [ArtificialAnalysis/AA-Omniscience-Public](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public) (Apache-2.0) |
| Model | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| Abliteration splice (`ABLIT=1`) | **Keys (drowzeys)** — gated; house QSA `o_proj` L3–47 on nvidia NVFP4 |

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
