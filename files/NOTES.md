# files/ — runtime dependency payloads (audit 2026-09-27)

This dir holds the host-side files `start.sh` / `stop.sh` / `engine/*.sh`
bind-mount or exec at launch. The old "copy these in from the single-spark
sister repo before a real boot" pattern is **gone**: every payload the
launcher needs is now IN-TREE. The "absent — copy it" warnings in
`engine/memwatch.sh`, `engine/ple.sh` and `stop.sh` are defensive guards for
stripped installs, not a standing to-do.

## Inventory (each row: who loads it)

| file | loaded by | what for |
|---|---|---|
| `memwatch.sh` | `engine/memwatch.sh` → `nohup bash` | host-MemAvailable watchdog; kills the container before a unified-pool hang |
| `evict_page_cache.py` | `stop.sh` (+ `experiments/v30/launch_v30.sh`) | `posix_fadvise(DONTNEED)` on the checkpoint tree so the next boot doesn't CUDA-OOM on an "idle" box |
| `patch_ple_mmap_v030.py` | `engine/ple.sh` (`ple_prepare_overlay`) | overlays the image's `ngram_embedding.py` with the mmap PLE table path; output dir `files/v030_ple/` is launch-generated, gitignored |
| `patch_qsa_fp8_kv_v030.py` | `start.sh` (when `KV_CACHE_DTYPE=fp8*`) | backports vllm#55557 fp8_e4m3 QSA KV onto v0.30 `nvidia/qsa.py` + `ops/qsa.py`; output dir `files/v030_fp8kv/` is launch-generated, gitignored. Delete-condition: image moves to vLLM 0.31+ |
| `model.py` `mtp.py` `hyperconnection.py` | `start.sh` (when `FP8DENSE=1`) | the 3-source fp8dense overlay mounted ro over `vllm/models/qwen4_exp/nvidia/` — needed ONLY for hybrid / lean-hyb checkpoints; stock nvidia ckpts boot without mounts |
| `patch_mtp_draft_vocab.py` | `patch_mtp_draft_vocab_v030.py` (imports its blocks) + `experiments/v30/launch_v30.sh` via `$OV` | TP-aware reduced-vocab MTP head splice (`VLLM_MTP_DRAFT_VOCAB`). Not wired by consolidated `start.sh` yet — the v0.30 delta keeps its delete-condition with the spike lane |
| `patch_mtp_draft_vocab_v030.py` | `experiments/v30/launch_v30.sh` | applies the blocks above onto the v0.30-portable `mtp.py` |
| `build_draft_vocab.py` | operator (`.env.example`, README recipe) | corpus → frequency-ranked draft-id list |
| `draft_vocab_en_code_47k.txt` | `MTP_DRAFT_VOCAB` knob | shipped 47,149-id code-tuned vocabulary (vendored, +13.1 % single-Spark measured); the one tracked exception to `files/draft_vocab_*.txt` |
| `test_draft_vocab.py` | manual: mount into the image (`python3 /tmp/t.py`) | CPU test of the draft-vocab shard slicing + cross-rank argmax — run after touching the patcher |
| `fp8dense/` | operator / `tools/README.md` pointers | hybrid-checkpoint pipeline: `build.sh` orchestrates `make_fp8_dense_checkpoint.py`; `verify_fp8_dense_checkpoint.py` (GPU-free verifier), `compute_quant_stats.py` (per-tensor RMSE), `test_quant_config_resolution.py` (in-image CPU quant-method resolution check). `build.sh` resolves the snapshot through `../resolve_snapshot.py` (kept; also used by `download.sh`) |

## Deleted from files/ in the consolidation (2026-09-27)

- Pre-v0.30 patcher era: `patch_qsa_fp8_kv.py`, `patch_ple_layer.py`,
  `patch_ple_offload*.py`, `patch_modelopt_*.py`, `patch_checkpoint_config.py`,
  `ple_layer_patched.py`, `ple_offload/*`, `overlay/*` (diff-era), `qsa_gb10/*`,
  `detect_ple_dtype.py`, `build_ple_packed_table.py`.
- `files/nfs-share.sh` — orphan: its header promised "Sourced by start.sh and
  check-weights.sh", but the consolidated `start.sh` never sources it (weights
  ride the direct `$HF_CACHE_DIR` bind-mount; dual = same script both nodes) and
  `check-weights.sh` itself was deleted. Zero refs in start.sh / stop.sh /
  engine/ / recipes/ / tests/ / bench/ / tools/ / experiments/. The old
  single-spark README copy survives under `docs/legacy/` for archaeology.

Generated-at-launch dirs (`v030_ple/`, `v030_fp8kv/`, `dvdraft/`) are
gitignored; wipe them freely — they rebuild from the patchers above.
| `patch_qsa_fused_draft_v030.py` | experiments R2 arm (`ride_r2.sh`, worktree) | backport of vllm#58449: the QSA builder declares `supports_draft_decode_metadata_update` + in-place `update_draft_decode_metadata`, flipping the speculator's fused multi-step draft loop ON (v0.30's consumer is present; our builder didn't declare support -> boot log "falling back to rebuilding attention metadata", k whole-model metadata rebuilds per round). Output dir `files/v030_fused/` is bench-staged, gitignored. DELETE when the image ships #58449 merged |
