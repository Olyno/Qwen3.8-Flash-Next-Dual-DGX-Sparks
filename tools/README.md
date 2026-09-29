# tools/ — one-off construction utilities

Not part of the boot path. `start.sh` never calls anything here; these build
or select artifacts that then live outside the repo (checkpoints, token maps).

## bake/ — lean checkpoint construction

Bakes the overthinking-marker logit penalty into `lm_head.weight`
(arXiv:2606.00206-style minimum-norm row surgery). Product lineage:
`docs/verdicts/LEAN.md`; the shipped `q38-lean-hyb` checkpoint carries the
result of this pipeline.

| file | one-liner |
|---|---|
| `bake/bake_lm_head.py` | rewrites the marker rows of `lm_head.weight` from calibration hidden states (`G=E[hhᵗ]`, exact mean penalty), copies every other shard byte-verbatim; run inside the vLLM image, see its docstring for the `docker run` form |
| `bake/select_token_map.py` | picks refined-vs-original marker set for the bake using the pre-registered rule (acc-neutral floor, max token reduction) over an overthinking-lab `results/` tree; needs `OVERTHINKING_LAB_DIR` |
| `bake/marker_token_map.json` | the chosen marker-token map (ids → penalty rows) fed to `bake_lm_head.py --token-map` |
| `bake/chosen_lambda.json` | the selected penalty strength + the selection rule + the bypass-bake decision record |

## Related scripts that live in `files/` (not tools/)

| file | one-liner |
|---|---|
| `../files/fp8dense/make_fp8_dense_checkpoint.py` | builds the NVFP4-experts + FP8-per-channel-dense hybrid checkpoint (streaming, CPU, hard-links expert/PLE shards) |
| `../files/fp8dense/verify_fp8_dense_checkpoint.py` | GPU-free verifier for a hybrid snapshot: index/weight-map consistency, F8_E4M3 + F32-scale pairing, hard-link inode equality, sampled dequant error |
| `../files/fp8dense/compute_quant_stats.py` | per-tensor relative-RMSE of every FP8 tensor vs the bf16 source |
| `../files/fp8dense/test_quant_config_resolution.py` | CPU-only check that the overlaid modelopt resolves the right quant method per layer prefix |
| `../files/build_draft_vocab.py` | builds a reduced MTP draft vocabulary from a corpus (the shipped one is `files/draft_vocab_en_code_47k.txt`) |
