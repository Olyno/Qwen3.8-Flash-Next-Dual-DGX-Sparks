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
    Nonzero also arms the read-side hygiene set: POSIX_FADV_RANDOM on a
    dedicated read fd for the table, taken before any gather read. 0 disables
    the whole set (kill-switch). A reused table is NOT pre-faulted at boot:
    with the page cache evicted, a full-table populate re-reads ~48 GiB from
    disk and stalls the boot for tens of minutes, which defeats the reuse
    path; rows fault in on demand instead (decode measured not fault-bound).
    Idea credit: NVIDIA/flare-local-inference (Apache-2.0), commit 5dd6769
    ignition/preparation/host_memory.py (readahead probe: a 256 MiB read under
    default advice retained 8 MiB, under RANDOM 0); idea-level, own implementation.
VLLM_PLE_STAGED_LOOKUP (default 0) host-gathers each batch's rows through a
    pinned staging buffer and H2Ds them, so the GPU never dereferences the
    file-backed mapping (faulting it through UVM, and reading wrong bytes under
    reclaim pressure). Per-replay staging hangs off CudaGraphManager.run_fullgraph,
    the one host hook a FULL graph replay still passes through; start_prefetch
    only stages on eager/prefill steps and returns untouched under capture.
    Off keeps the device kernel and is byte-identical to the stock path.

The eager prefetch hook (start_prefetch) additionally warms the page-cache
pages of each batch's rows through coalesced preads planned by
_ple_gather_plan (offset-sorted rows, same-page / near-gap rows folded into
one range) -- design credit NVIDIA/flare-local-inference (Apache-2.0)
helios/src/engram/reader.rs:437-490, idea-level, own implementation. With the
staged lookup on, those same preads keep their payload instead of discarding
it.
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


STAGED_HELPERS = '''_PLE_STAGE_READ_MAX = 1 << 23  # 8 MiB per staged pread into the scratch buffer

_PLE_LOOKUP_PATHS_LOGGED: set[str] = set()


def _ple_staged_lookup_enabled() -> bool:
    """VLLM_PLE_STAGED_LOOKUP: gather PLE rows through a pinned staging buffer.

    Off by default -- the device kernel keeps reading the file-backed mapping
    over ATS. On, the host preads each batch's rows into pinned memory and
    copies them to the device, so the GPU never dereferences the mapping.
    """
    value = os.environ.get("VLLM_PLE_STAGED_LOOKUP", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _ple_replay_lane() -> bool:
    """Whether the configured cudagraph mode has FULL graphs to hook.

    The per-replay staging call hangs off CudaGraphManager.run_fullgraph, so
    only a mode with FULL graphs can feed decode rows; anything else must
    keep the device kernel.
    """
    try:
        mode = get_current_vllm_config().compilation_config.cudagraph_mode
    except Exception:
        return False
    return "FULL" in str(getattr(mode, "name", mode)).upper()


def _ple_byte_view(buffer):
    """Flat writable byte view of a contiguous buffer (tensor view or bytearray)."""
    view = memoryview(buffer)
    if view.format != "B" or view.ndim != 1:
        view = view.cast("B")
    return view


def _ple_scatter_read(
    fd, stage_view, scratch_view, rows, start, offsets, slots, row_bytes
):
    """Read one coalesced row group into scratch and scatter it into stage."""
    length = offsets[rows[-1]] + row_bytes - start
    got = os.preadv(fd, [scratch_view[:length]], start)
    if got != length:
        raise OSError(
            f"PLE staged lookup: short read at {start}: {got}/{length} bytes"
        )
    for row in rows:
        src = offsets[row] - start
        dst = slots[row] * row_bytes
        stage_view[dst : dst + row_bytes] = scratch_view[src : src + row_bytes]


def _ple_stage_rows(
    fd, ids, row_bytes, vocab_start, vocab_end, stage, scratch,
    read_max=_PLE_STAGE_READ_MAX,
):
    """Gather one batch's PLE rows into ``stage``, in ids order.

    ``ids`` is the flat row-id sequence of the batch. In-shard ids are read
    from the table fd in file-offset order (``_ple_gather_plan`` coalesces
    neighbouring rows), at most ``read_max`` bytes per pread through
    ``scratch``, and scattered back to their ids-order slots. Ids outside
    ``[vocab_start, vocab_end)`` are zero-filled -- the same semantics as
    the masked load in ``_lookup_ple_rows_from_host_addr_kernel``.
    ``stage`` and ``scratch`` are writable contiguous byte buffers (numpy
    views of pinned CPU tensors in the embedding, bytearrays in tests).

    Host-side only. The caller must first drain the previous step's H2D out
    of ``stage``: the blocking ids D2H before this call is that barrier.
    """
    zero = bytes(row_bytes)
    stage_view = _ple_byte_view(stage)
    scratch_view = _ple_byte_view(scratch)
    if len(scratch_view) < row_bytes:
        raise ValueError("PLE staging scratch is smaller than one row")
    read_max = max(row_bytes, min(read_max, len(scratch_view)))
    offsets = []
    slots = []
    for slot, value in enumerate(ids):
        if vocab_start <= value < vocab_end:
            offsets.append((value - vocab_start) * row_bytes)
            slots.append(slot)
        else:
            begin = slot * row_bytes
            stage_view[begin : begin + row_bytes] = zero
    for start, _length, rows in _ple_gather_plan(offsets, row_bytes):
        group = []
        group_start = start
        for row in rows:
            offset = offsets[row]
            if group and offset + row_bytes - group_start > read_max:
                _ple_scatter_read(
                    fd, stage_view, scratch_view, group, group_start,
                    offsets, slots, row_bytes,
                )
                group = []
                group_start = offset
            group.append(row)
        if group:
            _ple_scatter_read(
                fd, stage_view, scratch_view, group, group_start,
                offsets, slots, row_bytes,
            )
    return len(ids)

'''

HELPERS = (
    inspect.getsource(_ple_gather_plan)
    + "\n\n"
    + '''@triton.jit
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
    + "\n\n"
    + STAGED_HELPERS
)

STAGED_INIT_TAIL = '''        self._num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * self.embedding_dim
        self._staged_lookup = (
            self._mmap_path is not None
            and _ple_staged_lookup_enabled()
            and _ple_replay_lane()
            and self.etp_data_parallel_size == 1
            and self._prefetch_rows > 0
        )
        self._stage_buffer: torch.Tensor | None = None
        self._stage_scratch: torch.Tensor | None = None
        if self._mmap_path is not None and _ple_staged_lookup_enabled():
            if not _ple_replay_lane():
                logger.warning(
                    "PLE %s: staged lookup not armed (the cudagraph mode has "
                    "no FULL graphs to hook); the device kernel reads the "
                    "mapping",
                    self._mmap_path,
                )
            elif self.etp_data_parallel_size != 1 or self._prefetch_rows <= 0:
                logger.warning(
                    "PLE %s: staged lookup not armed (ETP spans DP or no "
                    "prefetch rows); the device kernel reads the mapping",
                    self._mmap_path,
                )
        if (
            self._mmap_path is not None
            and self._mmap_path not in _PLE_LOOKUP_PATHS_LOGGED
        ):
            _PLE_LOOKUP_PATHS_LOGGED.add(self._mmap_path)
            logger.info(
                "PLE mmap lookup path: %s",
                "staged pinned gather"
                if self._staged_lookup
                else "device kernel over ATS",
            )
        if self._staged_lookup:
            # Allocated up front: a graph capture must not create streams or
            # allocate buffers. max_total_tokens * heads rows at row_bytes is
            # the hard per-step bound (8192 * 16 * 160 bytes = 20 MiB).
            self._stage_buffer = torch.empty(
                self._prefetch_rows * self._num_ngram_heads,
                self._row_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self._stage_scratch = torch.empty(
                _PLE_STAGE_READ_MAX, dtype=torch.uint8, device="cpu",
                pin_memory=True
            )
            self._prefetch_stream = torch.cuda.Stream(device=self._lookup_device)
            # Zeros, not empty: the capture-time forward reads this buffer
            # before the first replay stages real rows, and finite dummy
            # embeddings keep NaN/Inf out of the capture pass.
            self._prefetch_buffer = torch.zeros(
                self._prefetch_rows,
                self._num_ngram_heads,
                self.embedding_dim,
                dtype=self.weight.dtype,
                device=self._lookup_device,
            )
'''

STAGED_METHODS = '''    _PLE_MMAP_WARM_MAX = 1 << 23

    def _ple_mmap_warm(self, ids: torch.Tensor) -> None:
        """Gather or warm this batch's rows through coalesced host preads.

        Off (default): warm the page-cache pages the rows live on so the
        ATS gather inside _lookup hits resident pages; the payload is
        discarded. On (VLLM_PLE_STAGED_LOOKUP): keep the payload -- read
        every row into the pinned staging buffer in ids order, zero-fill
        ids this ETP rank does not own, and let _lookup copy the buffer to
        the device. The GPU then never dereferences the file-backed
        mapping, so a fault or a reclaim cannot change the bytes it reads.

        Ranges over _PLE_MMAP_WARM_MAX are skipped in the warm-only path;
        the staged path splits them instead, since a skipped range would
        leave stale rows in the output.
        """
        if torch.cuda.is_current_stream_capturing():
            # Host reads cannot run under capture: the ids D2H needs a
            # stream sync, which capture forbids, and a capture-time gather
            # could only feed capture-time rows to every replay.
            return
        if self._staged_lookup:
            self._stage_rows(ids.reshape(-1).to("cpu"))
            return
        values = ids.reshape(-1).to("cpu").tolist()
        row_bytes = self._row_bytes
        lo = self.shard_indices.org_vocab_start_index
        hi = self.shard_indices.org_vocab_end_index
        offsets = [(v - lo) * row_bytes for v in values if lo <= v < hi]
        if not offsets:
            return
        for start, length, _rows in _ple_gather_plan(offsets, row_bytes):
            if length > self._PLE_MMAP_WARM_MAX:
                continue
            try:
                os.pread(self._mmap_fd, length, start)
            except OSError:
                return

    def disable_staged_lookup(self, reason: str) -> None:
        """Fall back to the device kernel for this table."""
        if self._staged_lookup:
            logger.warning(
                "PLE %s: staged lookup disabled (%s); the device kernel "
                "reads the file-backed mapping",
                self._mmap_path,
                reason,
            )
        self._staged_lookup = False

    def _stage_rows(self, ids_cpu: torch.Tensor) -> int:
        """Gather this batch's rows into the pinned staging buffer.

        ``ids_cpu`` must come from a blocking D2H on the current stream:
        that copy drains the stream past the previous step's H2D out of
        this single pinned region, which is what makes reusing it safe.
        """
        stage = self._stage_buffer
        scratch = self._stage_scratch
        if stage is None or scratch is None or self._mmap_fd is None:
            raise RuntimeError("PLE staged lookup has no staging buffers")
        return _ple_stage_rows(
            self._mmap_fd,
            ids_cpu.reshape(-1).tolist(),
            self._row_bytes,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            stage.numpy(),
            scratch.numpy(),
        )

    def _upload_staged(
        self, rows: int, output: torch.Tensor | None = None
    ) -> None:
        """Copy pinned staged rows into the device lookup target.

        Non-blocking H2D on the current stream. The next staging step's
        blocking ids D2H orders after it, so the pinned source is not
        overwritten while the copy is in flight.
        """
        stage = self._stage_buffer
        if stage is None:
            raise RuntimeError("PLE staged lookup has no staging buffer")
        if rows > stage.shape[0]:
            raise RuntimeError(
                f"PLE staged lookup holds {stage.shape[0]} rows, but the "
                f"batch needs {rows}"
            )
        if output is None:
            prefetch = self._prefetch_buffer
            if prefetch is None:
                raise RuntimeError("PLE staged lookup has no prefetch buffer")
            output = prefetch[: rows // self._num_ngram_heads]
        output.view(torch.uint8).reshape(-1).copy_(
            stage[:rows].reshape(-1), non_blocking=True
        )

    def stage_rows_to_device(self, ids: torch.Tensor) -> None:
        """Stage device-computed ids and upload them to the prefetch buffer.

        The blocking ids D2H here is the load-bearing barrier: it cannot
        pass the previous step's H2D out of the same pinned staging buffer,
        so the host may overwrite that buffer for this step.
        """
        self._stage_rows(ids.reshape(-1).to("cpu"))
        self._upload_staged(ids.numel())

    def can_stage(self, num_tokens: int) -> bool:
        """Whether the staging buffers cover a captured batch of this size."""
        stage = self._stage_buffer
        prefetch = self._prefetch_buffer
        if stage is None or prefetch is None or self._mmap_fd is None:
            return False
        return (
            num_tokens > 0
            and num_tokens * self._num_ngram_heads <= stage.shape[0]
            and num_tokens <= prefetch.shape[0]
        )
'''

EDITS_COMMON = [
    (
        "from abc import ABC, abstractmethod\n",
        "import ctypes\nimport hashlib\nimport json\nimport mmap\nimport os\n"
        "from abc import ABC, abstractmethod\n"
        "from vllm.config import get_current_vllm_config\n",
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
        "        self._num_ngram_heads = num_ngram_heads\n"
        "        self._output_dim = num_ngram_heads * self.embedding_dim\n",
        STAGED_INIT_TAIL,
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
        + STAGED_METHODS
        + "\n"
        "    def _lookup(\n        self,\n        input_ids: torch.Tensor,\n",
    ),
    (
        "    @eager_break_during_capture\n"
        "    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:\n"
        "        \"\"\"Gather ETP IDs and launch their UVA lookup on the side stream.\"\"\"\n"
        "        buffer = self._prefetch_buffer\n",
        "    @eager_break_during_capture\n"
        "    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:\n"
        "        \"\"\"Gather ETP IDs and launch their UVA lookup on the side stream.\"\"\"\n"
        "        if self._staged_lookup and torch.cuda.is_current_stream_capturing():\n"
        "            # The replay hook stages this step's rows. Host work here\n"
        "            # would need a capture-stream sync (illegal under capture)\n"
        "            # and could only feed capture-time rows to every replay.\n"
        "            return\n"
        "        buffer = self._prefetch_buffer\n",
    ),
    (
        "        active_output = buffer[: gathered_ids.shape[0]]\n"
        "        prefetch_stream.wait_stream(torch.cuda.current_stream())\n",
        "        if self._mmap_fd is not None and (\n"
        "            self._staged_lookup or _ple_advice_value()\n"
        "        ):\n"
        "            self._ple_mmap_warm(gathered_ids)\n"
        "        active_output = buffer[: gathered_ids.shape[0]]\n"
        "        prefetch_stream.wait_stream(torch.cuda.current_stream())\n",
    ),
    (
        "        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)\n"
        "        embeddings = self._lookup(gathered_ids)\n",
        "        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)\n"
        "        if self._staged_lookup:\n"
        "            self._stage_rows(gathered_ids.reshape(-1).to(\"cpu\"))\n"
        "        embeddings = self._lookup(gathered_ids)\n",
    ),
    (
        "            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](\n"
        "                self._uva_weight,\n",
        "            if self._uva_weight is None:\n"
        "                if self._staged_lookup:\n"
        "                    # The pinned staging buffer already holds this\n"
        "                    # batch's rows (eager warm or the replay hook);\n"
        "                    # never dereference the file-backed mapping.\n"
        "                    self._upload_staged(flat_ids.numel(), output)\n"
        "                    return output\n"
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
    (
        "        prefetch_stream = self._prefetch_stream\n"
        "        if prefetch_stream is None:\n"
        "            raise RuntimeError(\"pinned PLE finalize requires a prior "
        "start_prefetch\")\n"
        "        torch.cuda.current_stream().wait_stream(prefetch_stream)\n",
        "        prefetch_stream = self._prefetch_stream\n"
        "        if prefetch_stream is None:\n"
        "            raise RuntimeError(\"pinned PLE finalize requires a prior "
        "start_prefetch\")\n"
        "        if not (self._staged_lookup\n"
        "                and torch.cuda.is_current_stream_capturing()):\n"
        "            # Under capture nothing was enqueued on the side stream (the\n"
        "            # staged path returns early); waiting there would bind the\n"
        "            # graph to an uncaptured stream and invalidate the capture.\n"
        "            torch.cuda.current_stream().wait_stream(prefetch_stream)\n",
    ),
]

STAGED_NVIDIA_HOOK = '''_STAGED_LAYERS: list = []
_STAGED_HOOK_INSTALLED = False


def _install_staged_lookup_hook(quiet: bool = False) -> bool:
    """Wrap CudaGraphManager.run_fullgraph with the per-replay staging call.

    The base-class method is wrapped, so ModelCudaGraphManager inherits the
    hook without another core file mounted. Idempotent, and gated on
    VLLM_PLE_STAGED_LOOKUP: off, it returns False and leaves vLLM's class
    untouched. Failures leave the staged path disarmed (the caller falls
    back to the device kernel) instead of risking stale rows. quiet
    suppresses the import-failure warning for the module-import attempt,
    which the layer constructor retries once all vLLM modules are loaded.
    """
    global _STAGED_HOOK_INSTALLED
    if not _ple_staged_lookup_enabled():
        return False
    if _STAGED_HOOK_INSTALLED:
        return True
    try:
        import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils

        CudaGraphManager = cudagraph_utils.CudaGraphManager
        target_cls = getattr(cudagraph_utils, "ModelCudaGraphManager", None)
    except Exception as exc:
        if not quiet:
            logger.warning(
                "PLE staged lookup disabled: cannot import CudaGraphManager (%s)",
                exc,
            )
        return False
    original = getattr(CudaGraphManager, "run_fullgraph", None)
    if original is None:
        logger.warning(
            "PLE staged lookup disabled: CudaGraphManager.run_fullgraph is "
            "missing"
        )
        return False
    if getattr(original, "_ple_staged_lookup_hook", False):
        _STAGED_HOOK_INSTALLED = True
        return True

    @functools.wraps(original)
    def run_fullgraph(self, desc):
        # The speculator's manager inherits this method too; only the
        # target's graphs consume the PLE rows. layer.stage_registered
        # ignores token counts this model never captured, so a layout
        # without ModelCudaGraphManager only costs a no-op walk.
        if target_cls is None or isinstance(self, target_cls):
            for ref in list(_STAGED_LAYERS):
                layer = ref()
                if layer is not None:
                    layer.stage_registered(desc.num_tokens)
        return original(self, desc)

    run_fullgraph._ple_staged_lookup_hook = True
    CudaGraphManager.run_fullgraph = run_fullgraph
    _STAGED_HOOK_INSTALLED = True
    logger.info("PLE staged lookup: per-replay staging hook installed")
    return True


def _register_staged_layer(layer) -> bool:
    """Register a layer for per-replay staging, pruning dead references."""
    if not _STAGED_HOOK_INSTALLED:
        return False
    live = [ref for ref in _STAGED_LAYERS if ref() is not None]
    if not any(ref() is layer for ref in live):
        live.append(weakref.ref(layer))
    _STAGED_LAYERS[:] = live
    return True


if _ple_staged_lookup_enabled():
    _install_staged_lookup_hook(quiet=True)'''

STAGED_NVIDIA_INIT = '''        if self.ngram_embedding.supports_prefetch:
            # The side-stream lookup outlives eager-break args, whose
            # graph-pool storage later segments may reuse.
            self._prefetch_ids = torch.empty(
                max_total_tokens, self.ngram_heads, dtype=torch.long
            )
        self._stage_views = None
        self._stage_sizes: set[int] = set()
        if (
            isinstance(self.ngram_embedding, Qwen4ExpPLEPinnedHostEmbedding)
            and getattr(self.ngram_embedding, "_staged_lookup", False)
        ):
            if not self._prefetch_ids.is_cuda:
                self.ngram_embedding.disable_staged_lookup(
                    "the prefetch ids buffer is not on the GPU"
                )
            elif not (
                _STAGED_HOOK_INSTALLED or _install_staged_lookup_hook()
            ):
                self.ngram_embedding.disable_staged_lookup(
                    "the per-replay staging hook is unavailable"
                )
            elif not _register_staged_layer(self):
                self.ngram_embedding.disable_staged_lookup(
                    "the layer could not register for replay staging"
                )
        weight = self.ngram_embedding.weight'''

STAGED_NVIDIA_START = '''        embedding = self.ngram_embedding
        if not embedding.supports_prefetch:
            return
        if getattr(embedding, "_staged_lookup", False):
            if torch.cuda.is_current_stream_capturing():
                # The replay hook gathers this graph's rows. Host work here
                # would need a capture-stream sync (illegal under capture)
                # and could only feed capture-time rows to every replay.
                if not _STAGED_HOOK_INSTALLED:
                    raise RuntimeError(
                        "PLE staged lookup: the replay hook is not installed"
                    )
                if (
                    self._stageable_views(
                        input_ids, query_start_loc, ngram_context
                    )
                    and input_ids.numel() <= self._prefetch_ids.shape[0]
                    and embedding.can_stage(input_ids.numel())
                ):
                    self._remember_stage_views(
                        input_ids, query_start_loc, ngram_context
                    )
                    return
                embedding.disable_staged_lookup(
                    "the captured inputs are outside the staging assumptions"
                )
        ngram_ids = self.compute_ngram_ids('''

STAGED_NVIDIA_METHODS = '''    @staticmethod
    def _stageable_views(
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> bool:
        """Whether the captured views match the replay staging assumptions."""
        return (
            input_ids.dim() == 1
            and input_ids.numel() > 0
            and input_ids.is_contiguous()
            and query_start_loc.is_contiguous()
            and query_start_loc.numel() >= 2
            and ngram_context.dim() == 2
            and ngram_context.is_contiguous()
        )

    def _remember_stage_views(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Keep the widest captured static buffers for replay staging.

        The runner rewrites these buffers in place before each replay, so
        holding the capture-time views (not their contents) is what lets
        the host recompute ids for the current step.
        """
        cached = self._stage_views
        if cached is None or input_ids.numel() > cached[0].numel():
            self._stage_views = (input_ids, query_start_loc, ngram_context)
        self._stage_sizes.add(input_ids.numel())

    def stage_registered(self, num_tokens: int) -> None:
        """Stage one FULL-graph replay's rows from the captured views.

        Runs before the graph replay on the caller's stream: ids D2H, host
        gather into pinned memory, H2D into the prefetch buffer, replay.
        Token counts this model never captured (the speculator's graphs)
        are ignored: a target graph always replays a captured descriptor,
        so skipping cannot leave a consuming graph with stale rows.
        """
        embedding = self.ngram_embedding
        if not getattr(embedding, "_staged_lookup", False):
            return
        views = self._stage_views
        if views is None or num_tokens not in self._stage_sizes:
            return
        input_ids, query_start_loc, ngram_context = views
        if not 0 < num_tokens <= input_ids.numel():
            raise RuntimeError(
                f"PLE staged lookup: replay needs {num_tokens} tokens, the "
                f"captured views hold {input_ids.numel()}"
            )
        if num_tokens > self._prefetch_ids.shape[0]:
            raise RuntimeError(
                f"PLE staged lookup: {num_tokens} tokens exceed the "
                f"{self._prefetch_ids.shape[0]}-row prefetch buffer"
            )
        ids = self.compute_ngram_ids(
            input_ids[:num_tokens],
            query_start_loc,
            ngram_context,
            output=self._prefetch_ids[:num_tokens],
        )
        embedding.stage_rows_to_device(ids)

'''

EDITS_NVIDIA = [
    (
        "from collections.abc import Iterable\n\nimport torch\n",
        "import functools\nimport weakref\n"
        "from collections.abc import Iterable\n\nimport torch\n",
    ),
    (
        "    Qwen4ExpPLEUnquantizedEmbeddingMethod,\n"
        ")\n"
        "from ..common.ops.ple import ple_ngram_ids\n",
        "    Qwen4ExpPLEUnquantizedEmbeddingMethod,\n"
        "    _ple_staged_lookup_enabled,\n"
        ")\n"
        "from ..common.ops.ple import ple_ngram_ids\n",
    ),
    (
        "    \"Qwen4ExpNGramEmbedding\",\n"
        "]\n"
        "\n"
        "\n"
        "class Qwen4ExpNGramEmbedding(nn.Module):\n",
        "    \"Qwen4ExpNGramEmbedding\",\n"
        "]\n"
        "\n"
        + STAGED_NVIDIA_HOOK
        + "\n"
        "\n"
        "\n"
        "class Qwen4ExpNGramEmbedding(nn.Module):\n",
    ),
    (
        "        if self.ngram_embedding.supports_prefetch:\n"
        "            # The side-stream lookup outlives eager-break args, whose\n"
        "            # graph-pool storage later segments may reuse.\n"
        "            self._prefetch_ids = torch.empty(\n"
        "                max_total_tokens, self.ngram_heads, dtype=torch.long\n"
        "            )\n"
        "        weight = self.ngram_embedding.weight\n",
        STAGED_NVIDIA_INIT + "\n",
    ),
    (
        "        embedding = self.ngram_embedding\n"
        "        if not embedding.supports_prefetch:\n"
        "            return\n"
        "        ngram_ids = self.compute_ngram_ids(\n",
        STAGED_NVIDIA_START + "\n",
    ),
    (
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        "        \"\"\"Load hash buffers and checkpoint-split embedding rows.\"\"\"\n",
        STAGED_NVIDIA_METHODS
        + "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        "        \"\"\"Load hash buffers and checkpoint-split embedding rows.\"\"\"\n",
    ),
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
