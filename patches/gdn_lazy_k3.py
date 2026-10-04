# K3 GDN MTP decode with lazy state commit (plan section 4 K3, L3; finding F08).
#
# Carried verbatim (MIT) from the sfxnz 2x-DGX-Spark recipe
# (inferences-engines/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark,
# docker/v030/gdn_lazy.py); design and evidence in its docker/v030/K3.md.
# patch_gdn_lazy_v030.py embeds this file at the end of the patched
# qwen_gdn_linear_attn.py, so it must stay a plain module: no __future__
# import, every top-level name prefixed _gdl / gdl / GDL.
#
# Source of truth for the kernels. docker/v030/apply_gdn_lazy_overlay.py embeds
# this file verbatim at the end of the v0.30.0 qwen_gdn_linear_attn.py overlay
# (docker/v030/gdn_lazy_linear_attn.py), so it must stay a plain module: no
# __future__ import, every top-level name prefixed _gdl / gdl / GDL.
# Design, slot scheme and evidence: docker/v030/K3.md.
#
# Stock v0.30 verify step (fused_gdn_decode_post_conv_mtp_kernel<float,3,true>,
# grid (req, HV), 256 threads): read the committed fp32 state from
# state_indices[req, a-1], run the T <= k+1 verify tokens, store the state after
# EVERY token t to state_indices[req, t] (1 read + T writes of 64 KiB per head).
#
# K3, same grid, one kernel per layer per step:
#   lazy row  : load S_c from column 0, replay the n <= T_prev accepted records of
#               the previous step from the ring, store S_c ONCE to column 0, run
#               the T verify tokens producing outputs only, append T records to
#               the ring (1 read + 1 write + ~4 KiB of ring traffic per head).
#   eager row : exactly the stock layout (after replaying any pending records):
#               candidate t stored to column t. Used when a 1600-token align
#               boundary is within 2k+1 tokens (so the V2 align pre-copy and
#               checkpoint copy only ever see the stock layout), when the ring
#               column is NULL, or when lazy is off but pending rings exist.
#   ring      : the exact stock-kernel INPUTS of each verify token, BF16 (post-
#               conv k row, v row, raw a, raw b; 516 B per head per token), held
#               in the tail (rows 120..127) of the head's region of the COLUMN-1
#               state slot, which the lazy layout never uses for state.
#   header    : int32 per (layer, state block): number of pending records whose
#               committed base S_c sits in that block (column 0). 0 = stock
#               layout. Written by a separate tiny kernel after the decode
#               kernel so every CTA of the step reads the same value.
#   fixup     : before ANY stock reader of the state (mixed batches, non-spec
#               rows, prefills), _gdl_materialize replays pending rings into the
#               exact stock layout (candidate t -> column t) and zeroes headers;
#               prefilling rows only zero headers (stale ring of a dead request).
#
# Numerics: the per-token update, l2norm, gating, butterfly reductions (xor
# 16..1 order, rebuilt as a split tree), gated RMSNorm and bf16 roundings follow
# the stock SASS op-for-op (FTZ inline PTX: mul/add/sub/fma .ftz, ex2.approx.ftz,
# rcp.approx.ftz, rsqrt.approx.ftz, libdevice log1pf). Target class A0 vs the
# stock CUDA kernel; the load-time self-test REQUIRES bitwise equality on GPU
# (outputs every step + committed state), else K3 stays off (fail closed).
# Replay reuses the same jit functions as the in-step update, so lazy == eager
# bitwise by construction (proven on CPU under TRITON_INTERPRET in
# tests/test_gdn_lazy.py, where the PTX helpers fall back to plain ops).
#
# Default OFF: VLLM_QWEN38_GDN_LAZY=1 on BOTH ranks, CUDA, fp32 SSM state,
# K=V=128, W=k+1 in 2..8, mamba_cache_mode none/align, and the self-test passed.
import logging
import os

import torch

try:
    import triton
    import triton.language as tl

    _GDL_HAS_TRITON = True
except Exception:  # noqa: BLE001 - no triton means stock path
    _GDL_HAS_TRITON = False

GDL_ENV = "VLLM_QWEN38_GDN_LAZY"
GDL_DIM = 128  # K = V = 128 (stock kernel requirement)
GDL_BV = 64  # V rows per chunk (stock kChunkV is 32; 64 x 8 warps measured best on GB10, evidence/k3)
GDL_TMAX = 8  # stock kMaxMtpTokens
# Ring at the tail of each head's 64 KiB region of the column-1 slot, in bf16
# elements: TMAX records of [k(128) | v(128)], then TMAX a, then TMAX b.
GDL_HEAD_BF16 = GDL_DIM * GDL_DIM * 2
GDL_RING_BF16 = GDL_TMAX * 2 * GDL_DIM + 2 * GDL_TMAX
GDL_RING_OFF = GDL_HEAD_BF16 - GDL_RING_BF16  # 30704 -> fp32 row 119.9 (last chunk)
GDL_LOG2E = 1.4426950216293334961  # float32(log2 e), the stock __expf constant
GDL_MODE_AUTO = 0
GDL_MODE_EAGER = 1
# Reduction implementation (see _gdl_red32): 0 = warp reduce (GPU default),
# 1 = explicit split tree (CPU default; GDL_RED=1 forces it on GPU).
GDL_RED_ENV = "VLLM_QWEN38_GDN_LAZY_RED"

_gdl_logger = logging.getLogger("vllm.qwen38.gdn_lazy")
_GDL_STATE = {
    "env": os.environ.get(GDL_ENV, "0").strip().lower(),
    "failed": False,  # self-test failed: never start a lazy row
    "tested": set(),  # variant keys that passed the self-test
    "used": False,  # a lazy row may exist: fixups must run from now on
    "detail": "",
}


def gdl_enabled() -> bool:
    return _GDL_STATE["env"] in ("1", "force") and _GDL_HAS_TRITON


def _gdl_interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def _gdl_red() -> int:
    v = os.environ.get(GDL_RED_ENV)
    if v in ("0", "1"):
        return int(v)
    return 1 if _gdl_interpret() else 0


if _GDL_HAS_TRITON:
    from triton.language.extra import libdevice as _gdl_libdevice

    # ---------------------------------------------------------------- FTZ ops
    # GPU: inline PTX with .ftz, exactly the stock SASS (FMUL/FADD/FFMA.FTZ,
    # MUFU.EX2/RCP/RSQ) and never contracted or reassociated by LLVM.
    # INTERP (CPU tests): plain ops; lazy vs eager stays bitwise because both
    # paths call the same helpers.
    @triton.jit
    def _gdl_mul(x, y, INTERP: tl.constexpr):
        if INTERP:
            return x * y
        else:
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "mul.ftz.f32 $0, $1, $2;", "=f,f,f", [x, y],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_add(x, y, INTERP: tl.constexpr):
        if INTERP:
            return x + y
        else:
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "add.ftz.f32 $0, $1, $2;", "=f,f,f", [x, y],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_sub(x, y, INTERP: tl.constexpr):
        if INTERP:
            return x - y
        else:
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "sub.ftz.f32 $0, $1, $2;", "=f,f,f", [x, y],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_fma(x, y, z, INTERP: tl.constexpr):
        if INTERP:
            return x * y + z
        else:
            x, y = tl.broadcast(x, y)
            x, z = tl.broadcast(x, z)
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "fma.rn.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f", [x, y, z],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_ex2(x, INTERP: tl.constexpr):
        if INTERP:
            return tl.exp2(x)
        else:
            return tl.inline_asm_elementwise(
                "ex2.approx.ftz.f32 $0, $1;", "=f,f", [x],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_rcp(x, INTERP: tl.constexpr):
        if INTERP:
            return 1.0 / x
        else:
            return tl.inline_asm_elementwise(
                "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_rsq(x, INTERP: tl.constexpr):
        if INTERP:
            return 1.0 / tl.sqrt(x)
        else:
            return tl.inline_asm_elementwise(
                "rsqrt.approx.ftz.f32 $0, $1;", "=f,f", [x],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_log1p(x, INTERP: tl.constexpr):
        if INTERP:
            return tl.log(1.0 + x)
        else:
            # libdevice log1pf with nvvm-reflect FTZ: same polynomial as the
            # stock SASS; its 3 non-ftz ops only differ on subnormal inputs,
            # which ex2.approx.ftz never produces.
            return _gdl_libdevice.log1p(x)

    @triton.jit
    def _gdl_c(x, C: tl.constexpr):
        """fp32 constant broadcast to x's shape (inline asm needs tensors)."""
        return tl.full(x.shape, C, tl.float32)

    @triton.jit
    def _gdl_bf16(x, INTERP: tl.constexpr):
        """fp32 -> bf16 round-to-nearest-even (F2FP.BF16.F32.PACK_AB). The CPU
        interpreter truncates, so there the rounding is done on the bits."""
        if INTERP:
            b = x.to(tl.uint32, bitcast=True)
            b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
            return b.to(tl.float32, bitcast=True).to(tl.bfloat16)
        else:
            return x.to(tl.bfloat16)

    # ---------------------------------------------------------- reductions
    @triton.jit
    def _gdl_addc_gpu(a, b):
        return tl.inline_asm_elementwise(
            "add.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
            dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_addc_cpu(a, b):
        return a + b

    @triton.jit
    def _gdl_red32(x, INTERP: tl.constexpr, RED: tl.constexpr):
        """[R, 32] -> [R] in the warp_reduce_sum order.
        RED=0: tl.reduce over axis 1. With the natural layout of a row-major
          [R, 128] tile (4 contiguous k per lane, 32 lanes per row) the lane axis
          holds one element per lane and Triton lowers the reduce to shfl.bfly
          16, 8, 4, 2, 1 - the stock order. Layout is the compiler's choice, so
          the self-test's bitwise gate is what proves it (fail closed).
        RED=1: explicit split tree (order guaranteed by construction; slow on
          GPU, used by the CPU tests and as a harness cross-check)."""
        if RED == 1:
            return _gdl_bfly(x, INTERP)
        else:
            if INTERP:
                return tl.reduce(x, 1, _gdl_addc_cpu)
            else:
                return tl.reduce(x, 1, _gdl_addc_gpu)

    @triton.jit
    def _gdl_bfly(x, INTERP: tl.constexpr):
        """[R, 32] -> [R]: the warp_reduce_sum xor-butterfly (16, 8, 4, 2, 1)
        as an explicit split tree. fp add is commutative, so lane j's result
        (and every lane's) equals this tree bit for bit."""
        R: tl.constexpr = x.shape[0]
        x = tl.permute(tl.reshape(x, [R, 2, 16]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        x = tl.permute(tl.reshape(x, [R, 2, 8]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        x = tl.permute(tl.reshape(x, [R, 2, 4]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        x = tl.permute(tl.reshape(x, [R, 2, 2]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        lo, hi = tl.split(x)
        return _gdl_add(lo, hi, INTERP)

    @triton.jit
    def _gdl_quad(v):
        """[128] -> 4 x [1, 32], v_i[j] = v[i*32 + j] (lane j holds dims
        j, j+32, j+64, j+96: the stock l2norm / RMSNorm mapping)."""
        t = tl.reshape(tl.permute(tl.reshape(v, [4, 32]), (1, 0)), [32, 2, 2])
        e, o = tl.split(t)
        v0, v2 = tl.split(e)
        v1, v3 = tl.split(o)
        return (tl.reshape(v0, [1, 32]), tl.reshape(v1, [1, 32]),
                tl.reshape(v2, [1, 32]), tl.reshape(v3, [1, 32]))

    @triton.jit
    def _gdl_lane4(x):
        """[R, 128] -> 4 x [R, 32], x_i[r, l] = x[r, 4l + i] (stock k_base =
        lane * 4 mapping of the state update)."""
        R: tl.constexpr = x.shape[0]
        e, o = tl.split(tl.reshape(x, [R, 32, 2, 2]))
        x0, x2 = tl.split(e)
        x1, x3 = tl.split(o)
        return x0, x1, x2, x3

    @triton.jit
    def _gdl_unlane4(x0, x1, x2, x3):
        R: tl.constexpr = x0.shape[0]
        t = tl.join(tl.join(x0, x2), tl.join(x1, x3))
        return tl.reshape(t, [R, 128])

    @triton.jit
    def _gdl_sumsq(v, INTERP: tl.constexpr, RED: tl.constexpr):
        """sum of squares of a [128] vector in the stock order -> [1]."""
        v0, v1, v2, v3 = _gdl_quad(v)
        s = _gdl_fma(v0, v0, tl.zeros_like(v0), INTERP)
        s = _gdl_fma(v1, v1, s, INTERP)
        s = _gdl_fma(v2, v2, s, INTERP)
        s = _gdl_fma(v3, v3, s, INTERP)
        return _gdl_red32(s, INTERP, RED)

    @triton.jit
    def _gdl_dot4(w0, w1, w2, w3, h0, h1, h2, h3, INTERP: tl.constexpr,
                  RED: tl.constexpr):
        """per-row dot(h[r, :], w) in the stock order: per lane
        fma(w_i, h_i, .) for i = 0..3 from 0, then the butterfly. -> [R]"""
        acc = _gdl_fma(w0, h0, tl.zeros_like(h0), INTERP)
        acc = _gdl_fma(w1, h1, acc, INTERP)
        acc = _gdl_fma(w2, h2, acc, INTERP)
        acc = _gdl_fma(w3, h3, acc, INTERP)
        return _gdl_red32(acc, INTERP, RED)

    # --------------------------------------------------------- token pieces
    @triton.jit
    def _gdl_qk(q_raw, k_raw, scale, INTERP: tl.constexpr, RED: tl.constexpr):
        """l2norm(q) * scale and l2norm(k) in lane4 layout ([1, 32] each)."""
        sq = _gdl_sumsq(q_raw, INTERP, RED)
        sk = _gdl_sumsq(k_raw, INTERP, RED)
        qs = _gdl_mul(_gdl_rsq(_gdl_add(sq, _gdl_c(sq, 1.0e-6), INTERP), INTERP),
                      scale, INTERP)
        ks = _gdl_rsq(_gdl_add(sk, _gdl_c(sk, 1.0e-6), INTERP), INTERP)
        q = _gdl_mul(q_raw, qs, INTERP)
        k = _gdl_mul(k_raw, ks, INTERP)
        q0, q1, q2, q3 = _gdl_lane4(tl.reshape(q, [1, 128]))
        k0, k1, k2, k3 = _gdl_lane4(tl.reshape(k, [1, 128]))
        return q0, q1, q2, q3, k0, k1, k2, k3

    @triton.jit
    def _gdl_k(k_raw, INTERP: tl.constexpr, RED: tl.constexpr):
        sk = _gdl_sumsq(k_raw, INTERP, RED)
        ks = _gdl_rsq(_gdl_add(sk, _gdl_c(sk, 1.0e-6), INTERP), INTERP)
        k = _gdl_mul(k_raw, ks, INTERP)
        return _gdl_lane4(tl.reshape(k, [1, 128]))

    @triton.jit
    def _gdl_gates(a_val, b_val, alog, dtb, INTERP: tl.constexpr):
        """decay = __expf(-__expf(A_log) * softplus(a + dt_bias)),
        beta = 1 / (1 + __expf(-b)); stock SASS order, all [1]."""
        x = _gdl_add(a_val, dtb, INTERP)
        ex = _gdl_ex2(_gdl_mul(x, _gdl_c(x, 1.4426950216293334961), INTERP), INTERP)
        sp = tl.where(x > 20.0, x, _gdl_log1p(ex, INTERP))
        e = _gdl_ex2(_gdl_mul(alog, _gdl_c(alog, 1.4426950216293334961), INTERP), INTERP)
        t = _gdl_mul(e, sp, INTERP)
        decay = _gdl_ex2(_gdl_mul(t, _gdl_c(t, -1.4426950216293334961), INTERP), INTERP)
        eb = _gdl_ex2(_gdl_mul(b_val, _gdl_c(b_val, -1.4426950216293334961), INTERP),
                      INTERP)
        beta = _gdl_rcp(_gdl_add(eb, _gdl_c(eb, 1.0), INTERP), INTERP)
        return decay, beta

    @triton.jit
    def _gdl_update(h0, h1, h2, h3, k0, k1, k2, k3, v, decay, beta,
                    INTERP: tl.constexpr, RED: tl.constexpr):
        """One delta-rule token on a [BV, 128] tile held as 4 x [BV, 32].
        h *= decay; hk = h.k; delta = (v - hk) * beta; h += k * delta."""
        h0 = _gdl_mul(h0, decay, INTERP)
        h1 = _gdl_mul(h1, decay, INTERP)
        h2 = _gdl_mul(h2, decay, INTERP)
        h3 = _gdl_mul(h3, decay, INTERP)
        hk = _gdl_dot4(k0, k1, k2, k3, h0, h1, h2, h3, INTERP, RED)
        delta = _gdl_mul(beta, _gdl_sub(v, hk, INTERP), INTERP)
        # expand_dims keeps delta's slice layout's parent (the tile's blocked
        # layout); tl.reshape produced a #linear layout that dragged the updated
        # tile - and the output dot's tl.reduce - into a register-sequential
        # (non-butterfly) reduction on sm_121: 1-ulp output misses vs stock.
        d = tl.expand_dims(delta, 1)
        h0 = _gdl_fma(k0, d, h0, INTERP)
        h1 = _gdl_fma(k1, d, h1, INTERP)
        h2 = _gdl_fma(k2, d, h2, INTERP)
        h3 = _gdl_fma(k3, d, h3, INTERP)
        return h0, h1, h2, h3

    @triton.jit
    def _gdl_load1(ptr):
        return tl.load(ptr + tl.zeros([1], dtype=tl.int32)).to(tl.float32)

    @triton.jit
    def _gdl_replay_one(h0, h1, h2, h3, ring_rec, ring_ab, t, rows, alog, dtb,
                        INTERP: tl.constexpr, TMAX: tl.constexpr, RED: tl.constexpr):
        k_raw = tl.load(ring_rec + tl.arange(0, 128)).to(tl.float32)
        v = tl.load(ring_rec + 128 + rows).to(tl.float32)
        a_val = _gdl_load1(ring_ab + t)
        b_val = _gdl_load1(ring_ab + TMAX + t)
        k0, k1, k2, k3 = _gdl_k(k_raw, INTERP, RED)
        decay, beta = _gdl_gates(a_val, b_val, alog, dtb, INTERP)
        return _gdl_update(h0, h1, h2, h3, k0, k1, k2, k3, v, decay, beta, INTERP, RED)

    @triton.jit
    def _gdl_row_plan(si_row, W, T, acc, hdr_ptr, seq_len, BS, MODE: tl.constexpr,
                      TMAX: tl.constexpr):
        """Per-request decisions shared by the decode and header kernels.
        Returns (valid, src_slot, n_replay, lazy, col0, ring_slot)."""
        col0 = tl.load(si_row)
        ring_slot = tl.load(si_row + 1, mask=W > 1, other=0)
        r = tl.load(hdr_ptr + col0, mask=col0 > 0, other=0)
        pending = r > 0
        acc_ok = (acc > 0) & (acc <= W)
        stock_src = tl.load(si_row + acc - 1, mask=acc_ok, other=0)
        src = tl.where(pending, col0, stock_src)
        n_rep = tl.where(pending, tl.minimum(tl.maximum(acc, 1), r), 0)
        valid = (src > 0) & (T <= TMAX)
        if MODE == 0:
            lazy = ring_slot > 0
        else:
            lazy = ring_slot < 0  # never
        nc = seq_len - T
        lazy = lazy & ((BS <= 0) | ((nc // tl.maximum(BS, 1))
                                    == ((nc + 2 * W - 1) // tl.maximum(BS, 1))))
        return valid, src, n_rep, lazy, col0, ring_slot

    # ------------------------------------------------------------- kernels
    @triton.jit
    def _gdl_decode_kernel(
        mixed_ptr, stride_mixed,
        a_ptr, stride_a, b_ptr, stride_b,
        alog_ptr, dtb_ptr,
        si_ptr, stride_si,
        cu_ptr, acc_ptr, seqlen_ptr,
        state_ptr, ring_ptr, stride_slot, stride_slot_ring,
        hdr_ptr,
        gate_ptr, stride_gate,
        nw_ptr, out_ptr,
        H, HV, W, BS, scale, eps,
        G: tl.constexpr, SIGMOID: tl.constexpr, MODE: tl.constexpr,
        INTERP: tl.constexpr, TMAX: tl.constexpr, BV: tl.constexpr,
        RING_OFF: tl.constexpr, RED: tl.constexpr,
    ):
        req = tl.program_id(0)
        hv = tl.program_id(1)
        bos = tl.load(cu_ptr + req)
        eos = tl.load(cu_ptr + req + 1)
        T = eos - bos
        if T <= 0:
            return
        acc = tl.load(acc_ptr + req)
        seq_len = tl.load(seqlen_ptr + req)
        si_row = si_ptr + req.to(tl.int64) * stride_si
        valid, src, n_rep, lazy, col0, ring_slot = _gdl_row_plan(
            si_row, W, T, acc, hdr_ptr, seq_len, BS, MODE, TMAX)
        cols = tl.arange(0, 128)
        out_row = out_ptr + (bos.to(tl.int64) * HV + hv) * 128
        if valid == 0:
            for t in range(0, T):
                tl.store(out_row + t * HV * 128 + cols,
                         tl.zeros([128], dtype=tl.float32).to(tl.bfloat16))
            return
        kh = hv // G
        alog = _gdl_load1(alog_ptr + hv)
        dtb = _gdl_load1(dtb_ptr + hv)
        head_off = hv.to(tl.int64) * 128 * 128
        src_base = state_ptr + src.to(tl.int64) * stride_slot + head_off
        col0_base = state_ptr + col0.to(tl.int64) * stride_slot + head_off
        ring_head = (ring_ptr + ring_slot.to(tl.int64) * stride_slot_ring
                     + hv.to(tl.int64) * (2 * 128 * 128) + RING_OFF)
        ring_ab = ring_head + TMAX * 256
        rr = tl.arange(0, BV)
        for c in tl.static_range(128 // BV):
            rows = c * BV + rr
            tile_off = rows[:, None] * 128 + cols[None, :]
            h0, h1, h2, h3 = _gdl_lane4(tl.load(src_base + tile_off))
            for t in range(0, n_rep):
                h0, h1, h2, h3 = _gdl_replay_one(
                    h0, h1, h2, h3, ring_head + t * 256, ring_ab, t, rows,
                    alog, dtb, INTERP, TMAX, RED)
            if not INTERP:
                tl.debug_barrier()  # ring reads before any column-1 store
            if lazy:
                tl.store(col0_base + tile_off, _gdl_unlane4(h0, h1, h2, h3))
            for t in range(0, T):
                tok = bos + t
                mrow = mixed_ptr + tok.to(tl.int64) * stride_mixed
                q_raw = tl.load(mrow + kh * 128 + cols).to(tl.float32)
                k_raw = tl.load(mrow + H * 128 + kh * 128 + cols).to(tl.float32)
                v = tl.load(mrow + 2 * H * 128 + hv * 128 + rows).to(tl.float32)
                a_val = _gdl_load1(a_ptr + tok.to(tl.int64) * stride_a + hv)
                b_val = _gdl_load1(b_ptr + tok.to(tl.int64) * stride_b + hv)
                q0, q1, q2, q3, k0, k1, k2, k3 = _gdl_qk(q_raw, k_raw, scale, INTERP, RED)
                decay, beta = _gdl_gates(a_val, b_val, alog, dtb, INTERP)
                h0, h1, h2, h3 = _gdl_update(h0, h1, h2, h3, k0, k1, k2, k3, v,
                                             decay, beta, INTERP, RED)
                o = _gdl_dot4(q0, q1, q2, q3, h0, h1, h2, h3, INTERP, RED)
                tl.store(out_row + t * HV * 128 + rows, _gdl_bf16(o, INTERP))
                if not lazy:
                    dst = tl.load(si_row + t)
                    if dst > 0:
                        dst_base = (state_ptr + dst.to(tl.int64) * stride_slot
                                    + head_off)
                        tl.store(dst_base + tile_off, _gdl_unlane4(h0, h1, h2, h3))
        if not INTERP:
            tl.debug_barrier()  # raw o of all chunks visible; last ring read done
        # gated RMSNorm epilogue on the bf16-rounded outputs (stock shared_out)
        w = tl.load(nw_ptr + cols).to(tl.float32)
        for t in range(0, T):
            tok = bos + t
            o = tl.load(out_row + t * HV * 128 + cols).to(tl.float32)
            s = _gdl_sumsq(o, INTERP, RED)
            rstd = _gdl_rsq(_gdl_fma(s, _gdl_c(s, 0.0078125), eps, INTERP), INTERP)
            y = _gdl_mul(_gdl_mul(o, rstd, INTERP), w, INTERP)
            gi = tl.load(gate_ptr + tok.to(tl.int64) * stride_gate + hv * 128
                         + cols).to(tl.float32)
            eg = _gdl_ex2(_gdl_mul(gi, _gdl_c(gi, -1.4426950216293334961), INTERP),
                          INTERP)
            sig = _gdl_rcp(_gdl_add(eg, _gdl_c(eg, 1.0), INTERP), INTERP)
            if SIGMOID:
                gate = sig
            else:
                gate = _gdl_mul(gi, sig, INTERP)
            y = _gdl_mul(gate, y, INTERP)
            if not INTERP:
                # every warp has loaded raw o of token t before any warp
                # overwrites it with y (the [128] reduce can be warp-local and
                # replicated across warps: a WAR race, seen at 8 warps)
                tl.debug_barrier()
            tl.store(out_row + t * HV * 128 + cols, _gdl_bf16(y, INTERP))
        if lazy:
            # append this step's exact inputs; replayed next step
            for t in range(0, T):
                tok = bos + t
                mrow = mixed_ptr + tok.to(tl.int64) * stride_mixed
                rec = ring_head + t * 256
                tl.store(rec + cols, tl.load(mrow + H * 128 + kh * 128 + cols))
                tl.store(rec + 128 + cols, tl.load(mrow + 2 * H * 128 + hv * 128 + cols))
                tl.store(ring_ab + t, tl.load(a_ptr + tok.to(tl.int64) * stride_a + hv))
                tl.store(ring_ab + TMAX + t,
                         tl.load(b_ptr + tok.to(tl.int64) * stride_b + hv))

    @triton.jit
    def _gdl_header_kernel(si_ptr, stride_si, cu_ptr, acc_ptr, seqlen_ptr, hdr_ptr,
                           W, BS, MODE: tl.constexpr, TMAX: tl.constexpr):
        req = tl.program_id(0)
        T = tl.load(cu_ptr + req + 1) - tl.load(cu_ptr + req)
        if T <= 0:
            return
        acc = tl.load(acc_ptr + req)
        seq_len = tl.load(seqlen_ptr + req)
        si_row = si_ptr + req.to(tl.int64) * stride_si
        valid, src, n_rep, lazy, col0, ring_slot = _gdl_row_plan(
            si_row, W, T, acc, hdr_ptr, seq_len, BS, MODE, TMAX)
        if valid:
            if lazy:
                tl.store(hdr_ptr + col0, T)
            else:
                for t in range(0, W):
                    s = tl.load(si_row + t)
                    if s > 0:
                        tl.store(hdr_ptr + s, 0)

    @triton.jit
    def _gdl_materialize_kernel(
        idx_ptr, stride_idx, flag_ptr,
        alog_ptr, dtb_ptr,
        state_ptr, ring_ptr, stride_slot, stride_slot_ring,
        hdr_ptr, W,
        INTERP: tl.constexpr, TMAX: tl.constexpr, BV: tl.constexpr,
        RING_OFF: tl.constexpr, RED: tl.constexpr,
    ):
        """Replay a pending ring into the stock layout: the state after record
        t goes to column t (t < r), exactly what the stock kernel stored."""
        row = tl.program_id(0)
        hv = tl.program_id(1)
        irow = idx_ptr + row.to(tl.int64) * stride_idx
        col0 = tl.load(irow)
        flag = tl.load(flag_ptr + row)
        if (flag == 0) | (col0 <= 0):
            return
        r = tl.load(hdr_ptr + col0)
        if r <= 0:
            return
        ring_slot = tl.load(irow + 1)
        alog = _gdl_load1(alog_ptr + hv)
        dtb = _gdl_load1(dtb_ptr + hv)
        head_off = hv.to(tl.int64) * 128 * 128
        col0_base = state_ptr + col0.to(tl.int64) * stride_slot + head_off
        ring_head = (ring_ptr + ring_slot.to(tl.int64) * stride_slot_ring
                     + hv.to(tl.int64) * (2 * 128 * 128) + RING_OFF)
        ring_ab = ring_head + TMAX * 256
        cols = tl.arange(0, 128)
        rr = tl.arange(0, BV)
        for c in tl.static_range(128 // BV):
            rows = c * BV + rr
            tile_off = rows[:, None] * 128 + cols[None, :]
            h0, h1, h2, h3 = _gdl_lane4(tl.load(col0_base + tile_off))
            s0 = h0
            s1 = h1
            s2 = h2
            s3 = h3
            for t in range(0, r):
                h0, h1, h2, h3 = _gdl_replay_one(
                    h0, h1, h2, h3, ring_head + t * 256, ring_ab, t, rows,
                    alog, dtb, INTERP, TMAX, RED)
                if t == 1:  # column 1 holds the ring: store it after all reads
                    s0 = h0
                    s1 = h1
                    s2 = h2
                    s3 = h3
                else:
                    if not INTERP:
                        # t = 0 stores to col0, the tile this chunk loaded: all
                        # warps' loads are consumed before any warp stores
                        tl.debug_barrier()
                    dst = tl.load(irow + t)
                    if dst > 0:
                        tl.store(state_ptr + dst.to(tl.int64) * stride_slot + head_off
                                 + tile_off, _gdl_unlane4(h0, h1, h2, h3))
            if not INTERP:
                tl.debug_barrier()
            if r > 1:
                if ring_slot > 0:
                    tl.store(state_ptr + ring_slot.to(tl.int64) * stride_slot
                             + head_off + tile_off, _gdl_unlane4(s0, s1, s2, s3))

    @triton.jit
    def _gdl_zero_hdr_kernel(idx_ptr, stride_idx, hdr_ptr, W):
        row = tl.program_id(0)
        irow = idx_ptr + row.to(tl.int64) * stride_idx
        for t in range(0, W):
            s = tl.load(irow + t)
            if s > 0:
                tl.store(hdr_ptr + s, 0)


# ------------------------------------------------------------------ launchers
def _gdl_state_ok(state: torch.Tensor) -> bool:
    try:
        state.view(torch.bfloat16)
    except RuntimeError:
        return False
    return (
        state.dtype == torch.float32
        and state.dim() == 4
        and state.shape[2] == GDL_DIM
        and state.shape[3] == GDL_DIM
        and state.stride(3) == 1
        and state.stride(2) == GDL_DIM
        and state.stride(1) == GDL_DIM * GDL_DIM
    )


def gdl_decode_launch(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
                      num_accepted, seq_lens, state, hdr, output_gate, norm_weight,
                      out, num_k_heads, scale, eps, sigmoid, block_size=0,
                      mode=GDL_MODE_AUTO):
    """One K3 verify step (decode kernel + header kernel). Shapes as the stock
    ops.fused_gdn_decode_post_conv_mtp; seq_lens [N] (num_computed + T) is only
    read when block_size > 0; hdr int32 [num_slots]."""
    n = state_indices.shape[0]
    W = state_indices.shape[1]
    HV = state.shape[1]
    G = HV // num_k_heads
    interp = _gdl_interpret()
    ring = state.view(torch.bfloat16)
    if n == 0:
        return
    _gdl_decode_kernel[(n, HV)](
        mixed_qkv, mixed_qkv.stride(0),
        a, a.stride(0), b, b.stride(0),
        A_log, dt_bias,
        state_indices, state_indices.stride(0),
        cu_seqlens, num_accepted, seq_lens,
        state, ring, state.stride(0), ring.stride(0),
        hdr,
        output_gate, output_gate.stride(0),
        norm_weight, out,
        num_k_heads, HV, W, int(block_size), float(scale), float(eps),
        G=G, SIGMOID=bool(sigmoid), MODE=int(mode), INTERP=interp,
        TMAX=GDL_TMAX, BV=GDL_BV, RING_OFF=GDL_RING_OFF, RED=_gdl_red(),
        num_warps=8,
    )
    _gdl_header_kernel[(n,)](
        state_indices, state_indices.stride(0), cu_seqlens, num_accepted, seq_lens,
        hdr, W, int(block_size), MODE=int(mode), TMAX=GDL_TMAX, num_warps=1,
    )


def gdl_materialize_launch(idx, flags, A_log, dt_bias, state, hdr):
    """idx int32 [n, W] (state columns per row), flags int32 [n]: 1 = replay a
    pending ring into the stock layout, 0 = only drop the headers (prefilling
    row: any ring there belongs to a dead request). Always zeroes the headers
    of all W columns afterwards."""
    n = idx.shape[0]
    if n == 0:
        return
    HV = state.shape[1]
    ring = state.view(torch.bfloat16)
    _gdl_materialize_kernel[(n, HV)](
        idx, idx.stride(0), flags, A_log, dt_bias,
        state, ring, state.stride(0), ring.stride(0), hdr, idx.shape[1],
        INTERP=_gdl_interpret(), TMAX=GDL_TMAX, BV=GDL_BV, RING_OFF=GDL_RING_OFF,
        RED=_gdl_red(), num_warps=4,
    )
    _gdl_zero_hdr_kernel[(n,)](idx, idx.stride(0), hdr, idx.shape[1], num_warps=1)


# ------------------------------------------------------------------ self-test
def _gdl_problem(device, gen, n=3, H=2, G=3, W=4, dt_dtype=torch.float32,
                 nw_dtype=torch.float32, slots_extra=2):
    """Random 'captured-like' verify inputs: post-conv qkv with SiLU-like
    magnitudes, a/b gate logits, a real-scale A_log / dt_bias."""
    HV = H * G
    D = GDL_DIM
    T_tokens = n * W

    def rn(*shape, s=1.0):
        # explicit cpu/fp32: model init runs under a cuda device context and a
        # bf16 default dtype
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cpu") * s

    p = {
        "H": H, "HV": HV, "W": W, "n": n,
        "mixed": (rn(T_tokens, 2 * H * D + HV * D, s=0.6)).to(torch.bfloat16).to(device),
        "a": rn(T_tokens, HV, s=2.0).to(torch.bfloat16).to(device),
        "b": rn(T_tokens, HV, s=2.0).to(torch.bfloat16).to(device),
        "A_log": (rn(HV, s=0.8) + 0.5).to(device),
        "dt_bias": rn(HV, s=1.0).to(dt_dtype).to(device),
        "gate": rn(T_tokens, HV, D, s=1.5).to(torch.bfloat16).to(device),
        "norm_w": (1.0 + rn(D, s=0.1)).to(nw_dtype).to(device),
        "num_slots": 1 + n * W + slots_extra,
    }
    return p


def gdl_self_test(device, G=3, sigmoid=True, dt_dtype=torch.float32,
                  nw_dtype=torch.float32, stock_fn=None, steps=4, W=4):
    """K3 lazy vs the stock kernel over `steps` verify steps with random
    acceptance, including T < W rows and a padded (T = 0) row. Returns (ok,
    detail). Requires bitwise equal outputs every step and bitwise equal
    committed states at the end (after a materialize). stock_fn defaults to
    torch.ops._C.fused_gdn_decode_post_conv_mtp via vllm._custom_ops."""
    if stock_fn is None:
        from vllm import _custom_ops as _ops

        def stock_fn(**kw):
            _ops.fused_gdn_decode_post_conv_mtp(**kw)
    gen = torch.Generator(device="cpu").manual_seed(0x4B33)

    def rn(*shape, s=1.0):
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cpu") * s

    H = 2
    n = 3
    p = _gdl_problem(device, gen, n=n, H=H, G=G, W=W, dt_dtype=dt_dtype,
                     nw_dtype=nw_dtype)
    HV, D = p["HV"], GDL_DIM
    num_slots = p["num_slots"]
    st0 = rn(num_slots, HV, D, D, s=0.05).to(device)
    s_stock = st0.clone()
    s_lazy = st0.clone()
    hdr = torch.zeros(num_slots, dtype=torch.int32, device=device)
    si = torch.arange(1, 1 + n * W, dtype=torch.int32, device="cpu").view(n, W).to(device)
    acc = torch.ones(n, dtype=torch.int32, device=device)
    eps = 1e-6
    scale = D ** -0.5
    try:
        for step in range(steps):
            # row lengths: full, short, and one padded row on odd steps
            lens = [W, max(1, W - 1 - (step % 2)), 0 if step % 2 else W]
            cum = [0]
            for x in lens:
                cum.append(cum[-1] + x)
            cu = torch.tensor(cum, dtype=torch.int32, device=device)
            ntok = cum[-1]
            mixed = rn(ntok, p["mixed"].shape[1], s=0.6).to(torch.bfloat16).to(device)
            a = rn(ntok, HV, s=2.0).to(torch.bfloat16).to(device)
            b = rn(ntok, HV, s=2.0).to(torch.bfloat16).to(device)
            gate = rn(ntok, HV, D, s=1.5).to(torch.bfloat16).to(device)
            seq = torch.full((n,), 100 + step * W, dtype=torch.int32, device=device)
            o_s = torch.zeros(ntok, HV, D, dtype=torch.bfloat16, device=device)
            o_l = torch.zeros_like(o_s)
            kw = dict(mixed_qkv=mixed, a=a, b=b, A_log=p["A_log"], dt_bias=p["dt_bias"],
                      state_indices=si, cu_seqlens=cu, num_accepted_tokens=acc,
                      state=s_stock, output_gate=gate, norm_weight=p["norm_w"], out=o_s,
                      scale=scale, norm_eps=eps,
                      output_gate_activation="sigmoid" if sigmoid else "silu")
            stock_fn(**kw)
            # step 2 runs eager rows (replay pending records, stock layout)
            gdl_decode_launch(mixed, a, b, p["A_log"], p["dt_bias"], si, cu, acc, seq,
                              s_lazy, hdr, gate, p["norm_w"], o_l, H, scale, eps,
                              sigmoid, block_size=0,
                              mode=GDL_MODE_EAGER if step == 2 else GDL_MODE_AUTO)
            if not torch.equal(o_s.view(torch.int16), o_l.view(torch.int16)):
                bad = int((o_s.view(torch.int16) != o_l.view(torch.int16)).sum())
                return False, f"step {step}: {bad} output elements differ"
            new_acc = []
            for r_, T in enumerate(lens):
                new_acc.append(int(torch.randint(1, max(T, 1) + 1, (1,), generator=gen,
                                                 device="cpu"))
                               if T > 0 else int(acc[r_]))
            acc = torch.tensor(new_acc, dtype=torch.int32, device=device)
        # committed state: materialize, then compare column a-1 of every row
        flags = torch.ones(n, dtype=torch.int32, device=device)
        gdl_materialize_launch(si, flags, p["A_log"], p["dt_bias"], s_lazy, hdr)
        for r_ in range(n):
            col = int(si[r_, int(acc[r_]) - 1])
            if not torch.equal(s_stock[col].view(torch.int32), s_lazy[col].view(torch.int32)):
                bad = int((s_stock[col] != s_lazy[col]).sum())
                return False, f"committed state row {r_}: {bad} elements differ"
        if torch.device(device).type == "cuda":
            # graph replay of one lazy step equals eager, twice
            gs = s_lazy.clone()
            gh = hdr.clone()
            cu = torch.tensor([0, W, 2 * W, 3 * W], dtype=torch.int32, device=device)
            ntok = 3 * W
            mixed = rn(ntok, p["mixed"].shape[1], s=0.6).to(torch.bfloat16).to(device)
            args = (mixed, mixed[:, :HV], mixed[:, HV:2 * HV], p["A_log"], p["dt_bias"],
                    si, cu, acc, torch.full((n,), 64, dtype=torch.int32, device=device))
            gate = rn(ntok, HV, D).to(torch.bfloat16).to(device)
            o_e = torch.zeros(ntok, HV, D, dtype=torch.bfloat16, device=device)
            ref_s, ref_h = gs.clone(), gh.clone()
            gdl_decode_launch(*args, ref_s, ref_h, gate, p["norm_w"], o_e, H, scale, eps,
                              sigmoid)
            o_g = torch.zeros_like(o_e)
            ws, wh = gs.clone(), gh.clone()
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                gdl_decode_launch(*args, ws, wh, gate, p["norm_w"], o_g, H, scale, eps,
                                  sigmoid)
            torch.cuda.current_stream().wait_stream(s_)
            g = torch.cuda.CUDAGraph()
            cs, ch = gs.clone(), gh.clone()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                gdl_decode_launch(*args, cs, ch, gate, p["norm_w"], o_g, H, scale, eps,
                                  sigmoid)
            for _ in range(2):
                cs.copy_(gs)
                ch.copy_(gh)
                g.replay()
                torch.cuda.synchronize()
                if not (torch.equal(o_g, o_e) and torch.equal(cs, ref_s)
                        and torch.equal(ch, ref_h)):
                    return False, "graph replay != eager"
            del g
    except Exception as e:  # noqa: BLE001 - fail closed
        return False, f"{type(e).__name__}: {e}"[:300]
    return True, f"bitwise over {steps} steps, W={W}, G={G}"


# ------------------------------------------------------------ layer dispatch
def _gdl_layer_key(layer):
    return (
        layer.num_v_heads // layer.num_k_heads,
        layer.norm.activation,
        layer.dt_bias.dtype,
        layer.norm.weight.dtype,
    )


def gdl_layer_init(layer) -> None:
    """Run the self-test once per variant at layer construction (before the
    FULL cudagraph capture), so captured decode graphs contain K3."""
    st = _GDL_STATE
    if not gdl_enabled() or st["failed"]:
        return
    if getattr(layer, "gdn_decode_kernel", None) != "cuda" or not torch.cuda.is_available():
        return
    if layer.head_k_dim != GDL_DIM or layer.head_v_dim != GDL_DIM:
        return
    key = _gdl_layer_key(layer)
    if key in st["tested"]:
        return
    G, act, dt_dtype, nw_dtype = key
    if st["env"] == "force":
        ok, detail = True, "self-test skipped (force)"
    else:
        ok, detail = gdl_self_test(torch.device("cuda", torch.cuda.current_device()),
                                   G=G, sigmoid=(act == "sigmoid"), dt_dtype=dt_dtype,
                                   nw_dtype=nw_dtype)
    st["detail"] = detail
    if not ok:
        st["failed"] = True
        _gdl_logger.warning("K3 gdn_lazy self-test FAILED %s: %s; stock GDN decode", key, detail)
        return
    st["tested"].add(key)
    _gdl_logger.info("K3 gdn_lazy self-test passed %s: %s", key, detail)


def _gdl_hdr(layer, md):
    hdrs = getattr(md, "lazy_hdr", None)
    if not hdrs:
        return None
    hdr = hdrs.get(layer.prefix)
    if hdr is None or hdr.numel() < layer.kv_cache[1].shape[0]:
        return None
    return hdr


def gdl_decode(layer, mixed_qkv, a, b, output_gate, out, md) -> bool:
    """Replace the stock fused MTP decode for a pure spec batch. Returns True
    if K3 ran (lazy or eager rows), False to run the stock op."""
    st = _GDL_STATE
    if not gdl_enabled():
        return False
    used = getattr(layer, "_gdl_used", False)
    state = layer.kv_cache[1]
    hdr = _gdl_hdr(layer, md)
    if hdr is None or not _gdl_state_ok(state):
        if used:
            raise RuntimeError("K3: pending GDN rings but no header table; refusing stock read")
        return False
    active = (not st["failed"]) and _gdl_layer_key(layer) in st["tested"]
    if not st.get("logged"):
        st["logged"] = True
        _gdl_logger.info("K3 gdn_lazy dispatch: %s (key %s, tested %s)",
                         "ACTIVE" if active else "inactive", _gdl_layer_key(layer),
                         sorted(map(str, st["tested"])))
    if not active and not used:
        return False
    n = md.num_spec_decodes
    si = md.spec_state_indices_tensor[:n]
    W = si.shape[1]
    if not (2 <= W <= GDL_TMAX):
        if used:
            raise RuntimeError("K3: unsupported width with pending rings")
        return False
    seq = getattr(md, "lazy_seq_lens", None)
    bs = int(getattr(md, "lazy_block_size", -1))
    mode = GDL_MODE_AUTO if active else GDL_MODE_EAGER
    if seq is None or bs < 0:
        # no boundary information (or mamba_cache_mode 'all'): never start a
        # ring, but still replay pending ones into the stock layout
        seq = md.num_accepted_tokens
        bs = 0
        mode = GDL_MODE_EAGER
    if mode == GDL_MODE_AUTO:
        layer._gdl_used = True
        st["used"] = True
    gdl_decode_launch(
        mixed_qkv, a, b, layer.A_log, layer.dt_bias, si,
        md.spec_query_start_loc[: n + 1], md.num_accepted_tokens[:n], seq[:n],
        state, hdr, output_gate, layer.norm.weight, out,
        layer.num_k_heads // layer.tp_size, layer.head_k_dim ** -0.5,
        layer.layer_norm_epsilon, layer.norm.activation == "sigmoid",
        block_size=bs, mode=mode,
    )
    return True


def gdl_fixup(layer, md) -> None:
    """Before any stock reader of the GDN state in this step: materialize
    pending rings of spec rows and of non-prefilling non-spec rows into the
    stock layout; drop headers of prefilling rows."""
    if not getattr(layer, "_gdl_used", False):
        return
    hdr = _gdl_hdr(layer, md)
    state = layer.kv_cache[1]
    if hdr is None:
        raise RuntimeError("K3: pending GDN rings but no header table")
    idx_parts, flag_parts = [], []
    if md.spec_state_indices_tensor is not None and md.num_spec_decodes > 0:
        si = md.spec_state_indices_tensor[: md.num_spec_decodes]
        idx_parts.append(si)
        flag_parts.append(torch.ones(si.shape[0], dtype=torch.int32, device=si.device))
    ns = getattr(md, "lazy_ns_state_indices", None)
    nsp = getattr(md, "lazy_ns_prefilling", None)
    if ns is not None and nsp is not None and ns.shape[0] > 0:
        W = idx_parts[0].shape[1] if idx_parts else ns.shape[1]
        idx_parts.append(ns[:, :W])
        flag_parts.append((~nsp).to(torch.int32))
    if not idx_parts:
        return
    idx = torch.cat([p.to(torch.int32) for p in idx_parts]).contiguous()
    flags = torch.cat(flag_parts).contiguous()
    gdl_materialize_launch(idx, flags, layer.A_log, layer.dt_bias, state, hdr)
