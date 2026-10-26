# Provenance

Where each shipped mechanism comes from, and what was evaluated and deliberately not shipped. Attribution for copied or vendored code lives in `NOTICE`; this file is the map from a mechanism to the work that proposed it, so a reader can check the claim rather than take it.

Every measurement referenced here is written up in `docs/serving-experiments.md`.

## Checkpoint lineages

Three NVFP4 builds of this model circulate, and they are not interchangeable as
measurement subjects:

| lineage | PLE dtype declaration | role here |
|---|---|---|
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | absent from `text_config`; declared only in `quantization_config.config_groups` | base of the served checkpoint (`…-NVFP4-lean`), whose `REPRODUCTION.md` starts from `serve/serve_tp1_nvidia.sh` |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `float8_e4m3fn` | default `SRC_REPO` of `scripts/fp8dense/build.sh`; the snapshot the FP8-dense converter was written and measured against |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | `nvfp4` | not present here |

Two consequences. A number produced on one lineage is not a number about another:
the FP8-dense hybrid (`q38-lean-hyb`) is built from the RadixArk snapshot by
`scripts/fp8dense/build.sh`, so its results belong to that artifact, not to the
served lean bake. And because the served checkpoint does not declare its PLE
dtype at all, `scripts/detect_ple_dtype.py` recovers it from the config group
that lists the table and passes it back as `--hf-overrides`; the boot line
"PLE table dtype not declared by checkpoint — overriding to float8_e4m3fn" is
that path firing.

## Adopted

| Mechanism | Where it lives | Source |
|---|---|---|
| Chain-of-draft prior in the template (marker penalty baked into the weights) | `chat_template_cod.jinja`, the checkpoint | own work, no paper |
| Stable prompt head: volatile blocks trail the conversation | `proxy/score_proxy.py` | [Don't Break the Cache](https://arxiv.org/abs/2601.06007) (agentic prompt-caching evaluation) and Luce-Org/lucebox's prompt-layout note |
| Prefix keepalive during agent pauses | `CacheKeepalive` in `proxy/score_proxy.py` | [Keeping the Cache Warm Pays](https://arxiv.org/abs/2607.19214) |
| Tolerant tool-call parse, schema-typed coercion, unclosed-tail repair, close-thinking retry | `proxy/score_proxy.py` | Luce-Org/lucebox `tool_parser.cpp`, zolotukhin/zinc `tool_format.zig`, uv-genai/colibri, MistVVK/XeStrata — reimplemented, no code copied |
| Duplicate tool-output collapse | `collapse_duplicate_tool_output` | Luce-Org/lucebox, ashhart/TensorFold — idea only |
| Per-request token accounting | `proxy/score_proxy.py` | own work |
| Tool-use task harness and scorer | `bench/tool_tasks.py` | own work |
| Reasoning and long-context set, token totals | `bench/reasoning_check.py` | own work |
| Pinned policy skill | `skills/ponytail`, `SkillIndex` | [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail), MIT, vendored verbatim |

The serving configuration itself (MTP depth, FP8 draft head, adaptive verify width, FP8 KV, PLE offload, expert parallelism) is assembled from vLLM and the upstream forks listed in `NOTICE`; the specific pull requests backported into the image are itemised there.

## Evaluated, not adopted

| Work | What it proposes | Why it is not here |
|---|---|---|
| [Marconi](https://arxiv.org/abs/2411.19379) | hybrid-aware prefix-cache admission and eviction | our working set fits the cache pool, so admission policy has nothing to decide; projected ~0% |
| [STEPQuant](https://arxiv.org/abs/2609.38169) | 6-bit delta-rule recurrent state | bf16 already takes half the state; 6-bit would force the decode path off the fused CUDA MTP kernel for a pool we cannot fill at this concurrency |
| [SketchSSM](https://arxiv.org/abs/2609.33051) | approximate state reads through a compact sketch | an approximation that trades accuracy by construction, and at batch ≤4 the state read is under 1% of a step |
| [SuffixReplay](https://arxiv.org/abs/2609.33477) | suffix replay instead of recurrent-state checkpoints | needs a buffered-write substrate that does not exist here; stock align-mode caching already covers the continuation case |
| [Speculative actions](https://arxiv.org/abs/2510.04371) | predict tool arguments and start the tool during decode | needs a tool-execution plane in the server, not a proxy; the tool term is real but this is a serving-engine change |
| [Parallel tool execution during decode](https://arxiv.org/abs/2603.18897) | same family, measured at −43.5% completion time | same blocker |

## Not in this repository

A wider survey of the field — several hundred papers and the candidate mechanisms extracted from them — was used to decide the table above. It is a working document, not part of the repository, because most of it is not shipped and a reader cannot verify a list of things this code does not contain. What is shipped is in the first table; what was rejected is in the second, with the reason, so the decisions stay checkable without it.
