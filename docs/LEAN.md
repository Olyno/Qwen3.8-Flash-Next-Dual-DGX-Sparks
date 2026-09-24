# lean — the overthinking-baked checkpoint

`Qwen3.8-Flash-Next-NVFP4-lean` is a variant of `nvidia/Qwen3.8-Flash-Next-NVFP4`
carrying one offline weight edit: the overthinking-marker logit penalty of
arXiv 2606.00206 (penalty strength λ = 2.0 over a 49-token filler/hedging
marker set, mined from this model's own chains of thought) baked into the
`lm_head` output-projection rows. Leaner reasoning comes out of the weights —
no serving flags, no decode loops, no client changes; any vLLM host serves the
folder as-is.

The repository adds:

1. **The bake recipe** — `tools/bake/bake_lm_head.py` + `tools/bake/marker_token_map.json`
   reproduce the checkpoint folder from the upstream
   `nvidia/Qwen3.8-Flash-Next-NVFP4` weights.
2. **`start-lean.sh`** — serves it on the 2-node cluster recipe (TP2+EP+MTP):
   stages `$HOME/models/Qwen3.8-Flash-Next-NVFP4-lean` into the head's
   HuggingFace cache as a local pseudo-repo, then launches `start.sh`
   unchanged. On a single Spark the test/experiment path is
   `MODEL_SOURCE=/path/to/checkpoint ./start-tp1.sh`.
3. **Chain-of-Draft prompts** (`prompts/cod/*`; Zhou et al., arXiv 2502.18600) —
   a client-side few-shot recipe that stacks on top of the baked weights for
   the best measured configuration; the weights stay untouched.

## Measured (this checkpoint, matched decode settings, McNemar tests vs the stock base)

| configuration | GPQA-Diamond (198) | MATH-500 | GSM8K (100) |
|---|---|---|---|
| base | 74.2 % | 88.8 % | 96.0 % |
| baked (`start-lean.sh`) | 82.8 (+8.6, p=.002, −16 % tokens) | 88.4 (−0.4 n.s., −9.6 %) | 97.0 (+1.0 n.s., +11.6 %) |
| baked + Chain-of-Draft prompts | **86.4 (+12.2, p<.001, −33.6 %)** | 88.7 (−0.1 n.s., −31.5 %) | 96.0 (0, −20.8 %) |

(Δacc vs base with McNemar p-value; % tokens = change in mean completion
tokens; n.s. = not significant. Same protocol, temperature 0.6, per-item seed.)

## Using the Chain-of-Draft prompts

For hard reasoning traffic, build each request as: `system = prompts/cod/sys.txt`,
then 2–4 exemplar turns (problem → short draft → final answer, from
`prompts/cod/<dataset>.json`), then the real question. GSM8K traffic may skip
the injection (measured parity there; prompt-free serving is simpler). The
full study behind these choices — 31 measured configurations, including the
DEER and REFRAIN decoding-loop methods and their failure modes — is
documented in `MODEL_CARD.md` and `benchmark_final_table.md` shipped inside
the checkpoint folder.
