#!/usr/bin/env python3
"""Port the [fp8dense overlay] + packed-PLE-table onto vLLM v0.30.0 qwen4_exp.

Semantics:
  * HC Linears / final mixer / lm_heads receive the resolved mixed-precision
    quant_config (same as the serve-image overlays).
  * NEW: the MTP HC mixer gets draft_vllm_config.quant_config. v0.30 declares
    mtp.*.hyper_connection_mixer FP8_PER_CHANNEL_PER_TOKEN in the checkpoint;
    without this the mixer stays unquantized and its FP8 weight load fails.
  * VLLM_PLE_PACKED_TABLE_DIR selects an mmap-backed PLE table (page-cache on
    GB10 unified memory instead of ~104 GiB anonymous pinned RAM). Checkpoint
    shard copies still flow through the normal loader (writing the same bytes
    into the map), so semantics are unchanged if the file exists.

Anchors are unique-count checked; the script refuses to write on drift.
"""
import os

SRC = os.path.expanduser("~/upgrade/v30/src")
OUT = os.path.expanduser("~/upgrade/v30/overlay")


def load(name):
    return open(os.path.join(SRC, name)).read()


def save(name, s):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write(s)


def sub(s, old, new, n=1):
    c = s.count(old)
    if c != n:
        raise SystemExit(f"anchor count {c} != {n}:\n{old[:200]}")
    return s.replace(old, new)


# ---------------------------------------------------------------- hyperconnection
hc = load("hyperconnection.py")
hc = sub(
    hc,
    """        use_combine: bool = True,
        prefix: str = "",
    ) -> None:""",
    """        use_combine: bool = True,
        prefix: str = "",
        quant_config=None,  # [fp8dense overlay]
    ) -> None:""",
)
hc = hc.replace(
    """                params_dtype=config.params_dtype,
                quant_config=None,""",
    """                params_dtype=config.params_dtype,
                quant_config=quant_config,  # [fp8dense overlay]""",
)
hc = hc.replace(
    """            params_dtype=config.params_dtype,
            quant_config=None,""",
    """            params_dtype=config.params_dtype,
            quant_config=quant_config,  # [fp8dense overlay]""",
)
assert hc.count("quant_config=quant_config,  # [fp8dense overlay]") == 3, "hc linear arms"
save("hyperconnection.py", hc)

# ---------------------------------------------------------------- model.py
m = load("model.py")
m = sub(
    m,
    """        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )""",
    """        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
            quant_config=quant_config,  # [fp8dense overlay]
        )""",
)
m = sub(
    m,
    """        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )""",
    """        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
            quant_config=quant_config,  # [fp8dense overlay]
        )""",
)
m = sub(
    m,
    """            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )""",
    """            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
                quant_config=vllm_config.quant_config,  # [fp8dense overlay]
            )""",
)
m = sub(
    m,
    """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )""",
    """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,  # [fp8dense overlay]
            prefix=maybe_prefix(prefix, "lm_head"),
        )""",
)
save("model.py", m)

# ---------------------------------------------------------------- mtp.py
t = load("mtp.py")
t = sub(
    t,
    """        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )""",
    """        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            quant_config=draft_vllm_config.quant_config,  # [fp8dense overlay]
        )""",
)
t = sub(
    t,
    """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )""",
    """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,  # [fp8dense overlay]
                    prefix=maybe_prefix(prefix, "lm_head"),
                )""",
)
save("mtp.py", t)

# ---------------------------------------------------------------- ngram_embedding.py
# NOT patched by string-splice here anymore. The proven v0.30 mmap path lives
# in files/patch_ple_mmap_v030.py (port of the single-spark repo's patcher,
# anchors verified against pristine image sources by that lane). port_v30.py
# only keeps model.py/mtp.py/hyperconnection.py splices.
# Boot #8 (09-27 ~13:0x) proved the old hand-rolled _PlePackedTableEmbedding
# still hung: it UVA-registered the mmap AND re-copied all 47.7 GiB of shards
# per boot. The sister patcher reads rows over ATS from a shared MAP_ANON-free
# file and skips shard copies once the persistent table is committed.
save("ngram_embedding.py", n)
print("ported:", sorted(os.listdir(OUT)))
