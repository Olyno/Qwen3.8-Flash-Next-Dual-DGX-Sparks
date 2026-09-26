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
n = load("ngram_embedding.py")
n = sub(n, "import torch\n", "import os\nimport torch\n", 1)
n = sub(
    n,
    """        embedding_cls = (
            Qwen4ExpPLEPinnedHostEmbedding
            if engram_config is not None and engram_config.cpu_offload
            else Qwen4ExpPLEDeviceEmbedding
        )""",
    """        if os.environ.get("VLLM_PLE_PACKED_TABLE_DIR"):
            embedding_cls = _PlePackedTableEmbedding
        else:
            embedding_cls = (
                Qwen4ExpPLEPinnedHostEmbedding
                if engram_config is not None and engram_config.cpu_offload
                else Qwen4ExpPLEDeviceEmbedding
            )""",
)
n = sub(
    n,
    """        self.ngram_embedding = embedding_cls(""",
    """        _pt_kwargs = {}
        if embedding_cls is _PlePackedTableEmbedding:
            _pt_kwargs = {"_packed_prefix": embedding_prefix}
        self.ngram_embedding = embedding_cls(""",
)
n = sub(
    n,
    """            data_parallel_rank=data_parallel_rank,
        )
        weight = self.ngram_embedding.weight""",
    """            data_parallel_rank=data_parallel_rank,
            **_pt_kwargs,
        )
        weight = self.ngram_embedding.weight""",
)

# mmap-backed pinned-table variant: reuses the UVA lookup of the pinned class.
# Defined at module end (after Qwen4ExpNGramEmbedding, which resolves it lazily
# through the module globals at construction time).
n += '''

class _PlePackedTableEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """Pinned-host PLE whose CPU buffer is an mmap of a pre-packed FP8 table.

    VLLM_PLE_PACKED_TABLE_DIR holds "<prefix>.ngram_embedding.packed_u8"
    (produced by the repo's build_ple_packed_table.py). On GB10 unified memory
    this trades ~104 GiB of non-evictable anonymous pinned RAM for a
    page-cache-backed map; lookup and dequant semantics are identical.
    """

    def __init__(self, *args, _packed_prefix: str = "", **kwargs) -> None:
        # Set before super().__init__ runs: allocate_embedding_weight is called
        # from the base constructor, so object.__setattr__ is required.
        object.__setattr__(self, "_packed_prefix", _packed_prefix)
        super().__init__(*args, **kwargs)

    def _find_packed_file(self) -> str | None:
        dir_ = os.environ["VLLM_PLE_PACKED_TABLE_DIR"]
        want = self._packed_prefix + ".packed_u8"
        try:
            entries = os.listdir(dir_)
        except OSError:
            return None
        # Match by suffix: the checkpoint-side table was named under the old
        # runtime prefix ("language_model.model..."), the v0.30 prefix may lack
        # the "language_model." root.
        for e in entries:
            if e.endswith(want):
                return os.path.join(dir_, e)
            tail = want[len("model.") :] if want.startswith("model.") else want
            if e.endswith(tail):
                return os.path.join(dir_, e)
        return None

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        path = self._find_packed_file() if dtype == torch.float8_e4m3fn else None
        if path is None:
            # Not FP8, or no table: normal pinned allocation + row copy.
            return super().allocate_embedding_weight(
                num_embeddings, embedding_dim, dtype
            )
        import mmap

        expected = num_embeddings * embedding_dim
        size = os.path.getsize(path)
        if size != expected:
            raise ValueError(
                f"packed PLE table {path} is {size} bytes, expected {expected}"
            )
        fd = os.open(path, os.O_RDWR)
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        weight = (
            torch.frombuffer(mm, dtype=torch.uint8)
            .view(num_embeddings, embedding_dim)
            .view(dtype)
            .requires_grad_(False)
        )
        object.__setattr__(self, "_packed_mmap", mm)  # keep the mapping alive
        return weight
'''
save("ngram_embedding.py", n)
print("ported:", sorted(os.listdir(OUT)))
