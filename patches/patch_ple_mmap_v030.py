#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""PLE mmap offload overlay for the vLLM v0.32.0 lane (v030 lane, PLE_OFFLOAD).

v0.32's PLE CPU offload parks the n-gram table in pinned anonymous host
memory (~32 GiB/node). This overlay reroutes the pinned-host embedding class
to a file-backed table (VLLM_PLE_MMAP_DIR) read over ATS and persisted across
launches, so a restart skips the multi-hundred-second re-gather. v0.32 split
the implementation: models/qwen4_exp/common/ngram_embedding.py holds the real
Qwen4ExpPLEPinnedHostEmbedding (the overlay target) and the nvidia module is
a thin wrapper whose load_weights tracks shard completion for the commit.
Both files are patched and mounted. The subclass keeps v0.32's native
supports_prefetch, so the mmap table gets the side-stream prefetch for free;
the redundant _row_bytes/_block_d plumbing the v0.30 overlay added itself is
now stock and only the mmap branch is layered on.

Inputs:  <dir>/orig/ngram_common.py and <dir>/orig/ngram_nvidia.py
         (start.sh extracts both from the image)
Outputs: <dir>/ple_common_v032.py and <dir>/ple_nvidia_v032.py
argv[1] overrides the base dir (used by the test); default
patches/v030_ple.

VLLM_PLE_MMAP_DIR  directory for the persistent tables (unset = pinned host)
VLLM_PLE_MMAP_ADVICE madvise advice for the mapping (default 1 = MADV_RANDOM).
    Nonzero also arms the unified-memory page-cache hygiene set: POSIX_FADV_RANDOM
    on a dedicated read fd for the table, taken before any gather read, and a
    MADV_POPULATE_READ pre-fault of an already-complete (ready) mapping so the
    first serve starts deterministic. 0 disables the whole set (kill-switch).
    Idea credit: NVIDIA/flare-local-inference (Apache-2.0), commit 5dd6769
    ignition/preparation/host_memory.py (readahead probe: a 256 MiB read under
    default advice retained 8 MiB, under RANDOM 0); idea-level, own implementation.

The eager prefetch hook (start_prefetch) additionally warms the page-cache
pages of each batch's rows through coalesced preads planned by
_ple_gather_plan (offset-sorted rows, same-page / near-gap rows folded into
one range) -- design credit NVIDIA/flare-local-inference (Apache-2.0)
helios/src/engram/reader.rs:437-490, idea-level, own implementation. The
gather itself stays GPU/ATS-driven; see the ponytail note on _ple_mmap_warm.
"""
import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _ple_gather_plan(offsets, row_bytes, page_size=4096):
    """Sort row file offsets and coalesce rows whose pages run together.

    Rows sharing a 4KiB page, or separated by a gap smaller than one page,
    fold into one read range; duplicate offsets fold trivially. Returns
    [(start, length, [row_index, ...])] ordered by ascending start and
    non-overlapping, each range covering every row it lists; row_index is
    the position in the input list. Pure -- stdlib only -- so it unit-tests
    offline (tests/test_ple_gather_coalesce.py). Design credit:
    NVIDIA/flare-local-inference helios/src/engram/reader.rs coalesced
    reader (idea-level, own implementation).
    """
    if row_bytes <= 0:
        raise ValueError("row_bytes must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    plan = []
    for i in sorted(range(len(offsets)), key=lambda i: (offsets[i], i)):
        start = offsets[i]
        end = start + row_bytes
        if plan:
            prev = plan[-1]
            same_page = start // page_size == (prev[1] - 1) // page_size
            if same_page or start - prev[1] < page_size:
                if end > prev[1]:
                    prev[1] = end
                prev[2].append(i)
                continue
        plan.append([start, end, [i]])
    return [(start, end - start, rows) for start, end, rows in plan]

HELPERS = inspect.getsource(_ple_gather_plan) + "\n\n" + '''@triton.jit
def _lookup_ple_rows_from_host_addr_kernel(
    weight_addr,
    ids_ptr,
    output_ptr,
    row_bytes,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_B: tl.constexpr,
):
    weight_ptr = weight_addr.to(tl.pointer_type(tl.uint8))
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_B)
    store_mask = offsets < row_bytes
    values = tl.load(
        weight_ptr + local_idx * row_bytes + offsets,
        mask=store_mask & in_range,
        other=0,
    )
    tl.store(output_ptr + row_id * row_bytes + offsets, values, mask=store_mask)


_PLE_MMAP_FORMAT = "ple-mmap-v1"

_MADV_POPULATE_READ = 23  # linux; EINVAL on <5.14 kernels is logged, not fatal


def _ple_advice_value() -> int:
    """VLLM_PLE_MMAP_ADVICE: madvise advice number; 0 disables all hygiene."""
    try:
        return int(os.environ.get("VLLM_PLE_MMAP_ADVICE", "1"))
    except ValueError:
        return 1


def _ple_mmap_sidecar(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def _ple_fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ple_checkpoint_identity() -> dict:
    model_config = get_current_vllm_config().model_config
    model = str(model_config.model_weights or model_config.model)
    snapshot = None
    if os.path.isdir(model):
        model = os.path.realpath(model)
        snapshot = os.path.basename(model)
    else:
        from huggingface_hub import try_to_load_from_cache

        try:
            cached = try_to_load_from_cache(
                model, "config.json", revision=model_config.revision
            )
        except ValueError:
            cached = None
        if isinstance(cached, str):
            snapshot = os.path.basename(os.path.dirname(cached))
    return {"model": model, "revision": model_config.revision, "snapshot": snapshot}


def _ple_mmap_location(prefix: str, rank: int) -> tuple[str, dict] | None:
    root = os.environ.get("VLLM_PLE_MMAP_DIR", "").strip()
    if not root:
        return None
    identity = _ple_checkpoint_identity()
    key = identity["snapshot"] or hashlib.sha256(identity["model"].encode()).hexdigest()
    os.makedirs(root, exist_ok=True)
    name = f"{prefix.replace('/', '_')}.{key[:12]}.etp{rank}.bin"
    return os.path.join(root, name), identity


def _ple_mmap_fingerprint(identity: dict, shape: tuple[int, int], dtype) -> dict:
    return {
        "format": _PLE_MMAP_FORMAT,
        **identity,
        "shape": list(shape),
        "dtype": str(dtype),
    }


def _ple_mmap_matches(path: str, fingerprint: dict, nbytes: int) -> bool:
    try:
        with open(_ple_mmap_sidecar(path)) as handle:
            stored = json.load(handle)
        return stored == fingerprint and os.path.getsize(path) == nbytes
    except (OSError, ValueError):
        return False


def _ple_mmap_open(
    path: str, shape: tuple[int, int], dtype, fingerprint: dict
) -> tuple[torch.Tensor, bool, int]:
    nbytes = shape[0] * shape[1] * dtype.itemsize
    ready = _ple_mmap_matches(path, fingerprint, nbytes)
    if not ready:
        sidecar = _ple_mmap_sidecar(path)
        if os.path.exists(sidecar):
            os.unlink(sidecar)
            _ple_fsync_dir(os.path.dirname(path))
        with open(path, "ab") as handle:
            handle.truncate(nbytes)
    # Unified-memory page-cache hygiene (one switch, VLLM_PLE_MMAP_ADVICE;
    # idea credit NVIDIA/flare-local-inference@5dd6769 host_memory.py).
    # FADV_RANDOM on a dedicated read-only fd before any gather read: under
    # default advice the kernel reads ahead around the sparse n-gram rows
    # and that readahead comes out of the same pool the GPU allocates from
    # on GB10. The fd stays open for the table's lifetime so the eager-path
    # warm reads reuse it. The mmap itself is governed by madvise, not
    # fadvise, so both are set: fadvise for fd reads, MADV_RANDOM for faults.
    read_fd = os.open(path, os.O_RDONLY)
    advice = _ple_advice_value()
    if advice:
        os.posix_fadvise(read_fd, 0, nbytes, os.POSIX_FADV_RANDOM)
    raw = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    if advice:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        start = raw.data_ptr() & ~(mmap.PAGESIZE - 1)
        length = raw.data_ptr() + nbytes - start
        rc = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), advice)
        logger.info("PLE mmap %s: madvise(%d) rc=%d", path, advice, rc)
        if ready:
            # Deterministic first touch: pre-fault a reused table now.
            # Decode measured not fault-bound, so this is predictability,
            # not throughput. Skipped on a fresh table -- the loader writes
            # every page shortly after anyway.
            rc = libc.madvise(
                ctypes.c_void_p(start), ctypes.c_size_t(length),
                _MADV_POPULATE_READ,
            )
            logger.info("PLE mmap %s: MADV_POPULATE_READ rc=%d", path, rc)
    return raw.view(dtype).view(*shape), ready, read_fd


def _ple_mmap_commit(path: str, weight: torch.Tensor, fingerprint: dict) -> None:
    nbytes = weight.numel() * weight.element_size()
    start = weight.data_ptr() & ~(mmap.PAGESIZE - 1)
    length = weight.data_ptr() + nbytes - start
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.msync(ctypes.c_void_p(start), ctypes.c_size_t(length), 4):
        raise OSError(ctypes.get_errno(), f"msync failed for {path}")
    sidecar = _ple_mmap_sidecar(path)
    with open(sidecar + ".tmp", "w") as handle:
        json.dump(fingerprint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(sidecar + ".tmp", sidecar)
    _ple_fsync_dir(os.path.dirname(path))

'''

EDITS_COMMON = [
    (
        "from abc import ABC, abstractmethod\n",
        "import ctypes\nimport hashlib\nimport json\nimport mmap\nimport os\n"
        "from abc import ABC, abstractmethod\n",
    ),
    (
        "    requires_device_loading: bool = False\n",
        "    requires_device_loading: bool = False\n"
        "\n"
        "    def process_weights_after_loading(self, layer: nn.Module) -> None:\n"
        "        commit = getattr(layer, \"commit_mmap_table\", None)\n"
        "        if commit is not None:\n"
        "            commit()\n",
    ),
    (
        "            raise ValueError(\"FP8 PLE checkpoint is missing its global scale\")\n",
        "            raise ValueError(\"FP8 PLE checkpoint is missing its global scale\")\n"
        "        super().process_weights_after_loading(layer)\n",
    ),
    (
        "class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):\n",
        HELPERS.lstrip("\n") + "\nclass Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):\n",
    ),
    (
        "        if not is_uva_available():\n"
        "            raise RuntimeError(\"Engram CPU offload requires UVA support\")\n",
        "        location = _ple_mmap_location(prefix, get_etp_group().rank_in_group)\n"
        "        self._mmap_path, self._mmap_identity = location or (None, None)\n"
        "        self._mmap_ready = False\n"
        "        self._mmap_fd: int | None = None\n"
        "        self._mmap_shards: set[int] = set()\n"
        "        if self._mmap_path is None and not is_uva_available():\n"
        "            raise RuntimeError(\"Engram CPU offload requires UVA support\")\n",
    ),
    (
        "        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)\n"
        "        self._row_bytes = self.embedding_dim * self.weight.element_size()\n"
        "        self._block_d = triton.next_power_of_2(self._row_bytes)\n"
        "        self._prefetch_stream: torch.cuda.Stream | None = None\n",
        "        if self._mmap_path is None:\n"
        "            self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)\n"
        "        else:\n"
        "            self._uva_weight = None\n"
        "            logger.info(\n"
        "                \"PLE %s: %s file-backed table %s (%.2f GiB), read over ATS\",\n"
        "                prefix,\n"
        "                \"reused\" if self._mmap_ready else \"building\",\n"
        "                self._mmap_path,\n"
        "                self.weight.numel() * self.weight.element_size() / 2**30,\n"
        "            )\n"
        "        self._lookup_device = (\n"
        "            torch.device(\"cuda\", torch.cuda.current_device())\n"
        "            if self._uva_weight is None\n"
        "            else self._uva_weight.device\n"
        "        )\n"
        "        self._row_bytes = self.embedding_dim * self.weight.element_size()\n"
        "        self._block_d = triton.next_power_of_2(self._row_bytes)\n"
        "        self._prefetch_stream: torch.cuda.Stream | None = None\n"
        "        self._block_b = triton.next_power_of_2(self._row_bytes)\n",
    ),
    (
        "                    self._prefetch_stream = torch.cuda.Stream(\n"
        "                        device=self._uva_weight.device\n"
        "                    )\n"
        "                    buffer = torch.empty(\n"
        "                        self._prefetch_rows,\n"
        "                        self._num_ngram_heads,\n"
        "                        self.embedding_dim,\n"
        "                        dtype=self.weight.dtype,\n"
        "                        device=self._uva_weight.device,\n"
        "                    )\n",
        "                    self._prefetch_stream = torch.cuda.Stream(\n"
        "                        device=self._lookup_device\n"
        "                    )\n"
        "                    buffer = torch.empty(\n"
        "                        self._prefetch_rows,\n"
        "                        self._num_ngram_heads,\n"
        "                        self.embedding_dim,\n"
        "                        dtype=self.weight.dtype,\n"
        "                        device=self._lookup_device,\n"
        "                    )\n",
    ),
    (
        "        \"\"\"Allocate the complete PLE weight directly in pinned CPU memory.\"\"\"\n"
        "        return torch.empty(\n",
        "        \"\"\"Allocate the complete PLE weight directly in pinned CPU memory.\"\"\"\n"
        "        if self._mmap_path is not None:\n"
        "            self._mmap_fingerprint = _ple_mmap_fingerprint(\n"
        "                self._mmap_identity, (num_embeddings, embedding_dim), dtype\n"
        "            )\n"
        "            weight, self._mmap_ready, self._mmap_fd = _ple_mmap_open(\n"
        "                self._mmap_path,\n"
        "                (num_embeddings, embedding_dim),\n"
        "                dtype,\n"
        "                self._mmap_fingerprint,\n"
        "            )\n"
        "            return weight\n"
        "        return torch.empty(\n",
    ),
    (
        "    def _lookup(\n        self,\n        input_ids: torch.Tensor,\n",
        "    def commit_mmap_table(self) -> None:\n"
        "        if self._mmap_path is None or self._mmap_ready:\n"
        "            return\n"
        "        expected = getattr(self, \"_mmap_expected_shards\", None)\n"
        "        if expected is None or len(self._mmap_shards) != expected:\n"
        "            logger.warning(\n"
        "                \"PLE %s: loaded %d/%s shards, table not persisted\",\n"
        "                self._mmap_path,\n"
        "                len(self._mmap_shards),\n"
        "                expected,\n"
        "            )\n"
        "            return\n"
        "        _ple_mmap_commit(self._mmap_path, self.weight, self._mmap_fingerprint)\n"
        "        self._mmap_ready = True\n"
        "        logger.info(\"PLE %s: built persistent table\", self._mmap_path)\n"
        "\n"
        "    _PLE_MMAP_WARM_MAX = 1 << 23\n"
        "\n"
        "    def _ple_mmap_warm(self, ids: torch.Tensor) -> None:\n"
        "        \"\"\"Warm the page-cache pages this batch's rows live on.\n"
        "\n"
        "        Coalesced host preads over _ple_gather_plan ranges on the\n"
        "        table's read fd, so the ATS gather the kernel below issues\n"
        "        hits resident pages and first-touch is deterministic.\n"
        "\n"
        "        ponytail: this warms, it does not stage. The gather itself\n"
        "        stays GPU-driven (Triton over ATS inside _lookup): a host\n"
        "        staging gather would replace a kernel the graph path depends\n"
        "        on, and start_prefetch (eager_break_during_capture) is the\n"
        "        only hook where host row work is capture-safe -- so under\n"
        "        graph replay this never runs. The pread payload is\n"
        "        discarded, so no pinned staging buffer is allocated\n"
        "        (pinning buys a discarded read nothing); ranges over\n"
        "        _PLE_MMAP_WARM_MAX are skipped. Decode measured not\n"
        "        fault-bound: this is deterministic first-touch, not\n"
        "        bandwidth.\n"
        "        \"\"\"\n"
        "        values = ids.reshape(-1).to(\"cpu\").tolist()\n"
        "        row_bytes = self._row_bytes\n"
        "        lo = self.shard_indices.org_vocab_start_index\n"
        "        hi = self.shard_indices.org_vocab_end_index\n"
        "        offsets = [(v - lo) * row_bytes for v in values if lo <= v < hi]\n"
        "        if not offsets:\n"
        "            return\n"
        "        for start, length, _rows in _ple_gather_plan(offsets, row_bytes):\n"
        "            if length > self._PLE_MMAP_WARM_MAX:\n"
        "                continue\n"
        "            try:\n"
        "                os.pread(self._mmap_fd, length, start)\n"
        "            except OSError:\n"
        "                return\n"
        "\n"
        "    def _lookup(\n        self,\n        input_ids: torch.Tensor,\n",
    ),
    (
        "        active_output = buffer[: gathered_ids.shape[0]]\n"
        "        prefetch_stream.wait_stream(torch.cuda.current_stream())\n",
        "        if self._mmap_fd is not None and _ple_advice_value():\n"
        "            self._ple_mmap_warm(gathered_ids)\n"
        "        active_output = buffer[: gathered_ids.shape[0]]\n"
        "        prefetch_stream.wait_stream(torch.cuda.current_stream())\n",
    ),
    (
        "            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](\n"
        "                self._uva_weight,\n",
        "            if self._uva_weight is None:\n"
        "                _lookup_ple_rows_from_host_addr_kernel[(flat_ids.numel(),)](\n"
        "                    self.weight.data_ptr(),\n"
        "                    flat_ids,\n"
        "                    output.view(torch.uint8),\n"
        "                    self._row_bytes,\n"
        "                    self.shard_indices.org_vocab_start_index,\n"
        "                    self.shard_indices.org_vocab_end_index,\n"
        "                    BLOCK_B=self._block_b,\n"
        "                )\n"
        "                return output\n"
        "            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](\n"
        "                self._uva_weight,\n",
    ),
]

EDITS_NVIDIA = [
    (
        "                embedding.weight.weight_loader(\n"
        "                    embedding.weight,\n"
        "                    loaded_weight,\n"
        "                    checkpoint_start=checkpoint_start,\n"
        "                )\n",
        "                if getattr(embedding, \"_mmap_path\", None) is not None:\n"
        "                    embedding._mmap_shards.add(shard_index)\n"
        "                    embedding._mmap_expected_shards = (\n"
        "                        embedding.org_vocab_size + shard_size - 1\n"
        "                    ) // shard_size\n"
        "                if not getattr(embedding, \"_mmap_ready\", False):\n"
        "                    embedding.weight.weight_loader(\n"
        "                        embedding.weight,\n"
        "                        loaded_weight,\n"
        "                        checkpoint_start=checkpoint_start,\n"
        "                    )\n",
    ),
]


def patch(src: str, edits, name: str) -> str:
    if "_PLE_MMAP_FORMAT" in src:
        sys.exit(f"{name}: orig is already patched")
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(f"{name}: anchor {i} not unique/missing (count={count}):\n{old[:180]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"{name}: patched source does not parse: {exc}")
    return src


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "v030_ple")
    jobs = (
        ("orig/ngram_common.py", "ple_common_v032.py", EDITS_COMMON,
         "ngram_embedding.py (common)"),
        ("orig/ngram_nvidia.py", "ple_nvidia_v032.py", EDITS_NVIDIA,
         "ngram_embedding.py (nvidia)"),
    )
    for orig_rel, out_rel, edits, name in jobs:
        orig = os.path.join(base, orig_rel)
        if not os.path.isfile(orig):
            sys.exit(f"ERROR: missing {orig} (start.sh extracts it from the image)")
        with open(orig) as handle:
            src = patch(handle.read(), edits, name)
        out = os.path.join(base, out_rel)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as handle:
            handle.write(src)
        print(f"patched {out}")


if __name__ == "__main__":
    main()
