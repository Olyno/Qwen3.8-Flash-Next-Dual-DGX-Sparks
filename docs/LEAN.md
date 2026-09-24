# lean — the overthinking-baked model (P1) and the P1×P2 combo

## What `lean` adds over upstream
1. **Baked P1 checkpoint**: the logit penalty from arXiv 2606.00206 (λ=2.0 over a 49-token
   overthinking-marker set) is folded into `lm_head` weights — no serving flags, no loops.
   Any vLLM host serves the folder `Qwen3.8-Flash-Next-NVFP4-lean` as-is.
   - reproduce your own folder: `tools/bake/bake_lm_head.py` + `marker_token_map.json`
     (`--lambda 2.0`), against upstream `nvidia/Qwen3.8-Flash-Next-NVFP4`;
   - `start-lean.sh` serves it (MODEL_SOURCE-aware, kv-cache auto for QSA, native 262K default).
2. **P1×P2 combo (best measured config)**: baked weights **plus** Chain-of-Draft exemplar
   prompts (`prompts/cod/*`). Client injects the fewshot turns; weights stay untouched.

## Measured (this checkpoint, matched plane, McNemar vs base)
| config | gpqa200 | math500 | gsm8k100 |
|---|---|---|---|
| base | 74.2 % | 88.8 % | 96.0 % |
| baked-P1 only (`start-lean.sh`) | 82.8 (+8.6, p=.002, −16 % tok) | 88.4 (−0.4 n.s., −9.6 %) | 97.0 (+1.0 n.s., +11.6 %) |
| baked-P1 + CoD prompts (P1×P2) | **86.4 (+12.2, p<.001, −33.6 %)** | 88.7 (−0.1 n.s., −31.5 %) | 96.0 (0, −20.8 %) |

## Using the P1×P2 prompts
For hard reasoning traffic, build the request as: `system = prompts/cod/sys.txt`, then
2–4 exemplar turns (problem→draft→answer pairs from `prompts/cod/<dataset>.json`), then the
real question. GSM8K traffic may skip it (combo = parity there, prompt-free is simpler).
Full method matrix (31 arms incl. DEER/REFRAIN and their failure modes): lab repo
`~/Qwen38-overthinking-lab/docs/REPORT.md` §Verdict.
