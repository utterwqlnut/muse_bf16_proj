# okc: problem=muse_bf16_proj
"""The dense projections: Triton kernels for the router (FP32 up to 2 tokens, fused BF16
hi + lo above); for the BF16 layers the Triton split-K kernel or cuBLAS (BF16_CONFIGS), and at
one token a lossless 12-bit copy of W read by a CUDA GEMV (nf12_gemv, CUDA cores), compiled in
prepare() with NVRTC (cuda-python) or CuPy. The shapes the GEMV serves (NF12_CUDA_DISPATCH) are
from sweep_all.py; prepare() checks each against an FP32 product before serving it, and if the
CUDA toolchain is missing the file runs the Triton / cuBLAS paths everywhere."""

import ctypes
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # Hopper TMA tensor descriptors (Triton 3.4+); without them the router uses plain loads.
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; up to 8 tokens, only rows 0-1023 (this GPU's slice)
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [M, K]
    config: Config


# The router runs in FP32 up to this many tokens, and as BF16 hi + lo GEMMs above it. On an
# H200 (sweep_all.py) FP32 is faster at m 1 and 2; hi + lo from m 4, by 1.5x at 8.
ROUTER_FP32_MAX_TOKENS = 2

# FP32 router kernel tiles: 1 output per program (256 programs for 132 SMs), and all of K in
# one step, so each program's 32 KB of W is requested at once and waited on once. BLOCK_M is m
# rounded up to a power of two, at least 2. From sweep_all.py (1.09x the 2-output, 16-warp tiles).
ROUTER_BLOCK_N = 1
ROUTER_BLOCK_K = 8192
ROUTER_NUM_WARPS = 8
ROUTER_NUM_STAGES = 1

# hi + lo kernel configs, swept on an H200: (max m, (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K,
# num_warps, num_stages)). Each m takes the first row whose max m >= m; the last row covers
# everything above. Runners-up were within 1-2% at each m.
ROUTER_HILO_CONFIGS = (
    (8, (16, 64, 64, 32, 4, 4)),      # 1 x 4 x 32 = 128 programs
    (16, (16, 64, 64, 32, 4, 6)),     # 1 x 4 x 32 = 128
    (32, (32, 64, 128, 32, 4, 4)),    # 1 x 4 x 32 = 128
    (64, (64, 32, 64, 32, 4, 4)),     # 1 x 8 x 32 = 256
    (128, (64, 64, 128, 16, 4, 4)),   # 2 x 4 x 16 = 128
    (256, (128, 32, 64, 8, 8, 5)),    # 2 x 8 x 8 = 128
)

# The TMA kernel runs hi + lo when Triton has tensor descriptors. Its configs, by the same rows.
# Its tl.dot takes the BLOCK_N weight rows as its first dimension, which Hopper's wgmma needs
# to be at least 64; the BLOCK_M tokens need only be at least 16.
ROUTER_USE_TMA = TensorDescriptor is not None
# Swept on an H200 (sweep_all.py): level with the plain-loads kernel at m 8 to 64, 3% faster at
# 128 and 15% at 256. An optional 7th element "ticket" replaces the zeroed-output + atomics
# combine with the partials + ticket combine (no memset node); swept, it was 2.5-6 us slower at
# every m, so every row keeps the atomics.
ROUTER_HILO_TMA_CONFIGS = (
    (8, (16, 64, 256, 32, 4, 3)),     # 1 x 4 x 32 = 128 programs (sweep_all.py: 1.055x)
    (16, (16, 64, 256, 32, 4, 2)),    # 1 x 4 x 32 = 128 (sweep_all.py)
    (32, (16, 64, 256, 16, 4, 3)),    # 2 x 4 x 16 = 128 (sweep_all.py)
    (64, (32, 64, 128, 16, 4, 4)),    # 2 x 4 x 16 = 128
    (128, (64, 64, 128, 16, 4, 4)),   # 2 x 4 x 16 = 128
    (256, (128, 64, 128, 16, 4, 3)),  # 2 x 4 x 16 = 128
)

# The BF16 layers run on _bf16_gemm_kernel where it beats cuBLAS, else on cuBLAS (F.linear),
# as SGLang does. By layer: (m, (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) or
# None for cuBLAS), each row for exactly that m, the m it was swept at; every other m takes
# cuBLAS, so a shape no sweep timed runs as SGLang runs it. Filled from sweep_all.py.
BF16_USE_TRITON = TensorDescriptor is not None
# Swept on an H200 (sweep_all.py): judge-style speedup over cuBLAS in each comment; kept where
# at least 2% faster, None (cuBLAS) elsewhere.
BF16_CONFIGS = {
    "o": (
        (1, (16, 64, 256, 1, 4, 4)),      # 1.11x
        (8, (16, 64, 256, 1, 4, 4)),      # 1.09x
        (16, (16, 64, 256, 1, 4, 4)),     # 1.10x
        (32, (32, 64, 128, 1, 4, 5)),     # 1.11x
        (128, (128, 64, 128, 1, 4, 4)),   # 1.02x
        (256, None),                      # 0.84x
    ),
    "pooler": (
        (1, (16, 64, 256, 1, 4, 4)),      # 1.12x
        (8, (16, 64, 256, 1, 4, 5)),      # 1.18x (sweep_all.py)
        (16, (16, 64, 256, 1, 4, 5)),     # 1.15x (sweep_all.py)
        (32, (16, 64, 256, 1, 4, 5)),     # 1.23x (sweep_all.py)
        (128, (64, 64, 128, 1, 4, 6)),    # 1.20x
        (256, (128, 64, 128, 1, 4, 4)),   # 1.10x
    ),
    "qkv_gate": (
        (1, (16, 64, 128, 4, 4, 4)),      # 1.07x
        (8, (16, 64, 128, 4, 4, 4)),      # 1.12x
        (16, (16, 64, 128, 5, 8, 4)),     # 1.08x
        (32, (16, 64, 128, 3, 4, 5)),     # 1.09x the cuBLAS this row had (sweep_all.py, twice ahead)
        (128, None),                      # 0.71x
        (256, None),                      # 0.79x
    ),
    "unpooler": (
        (1, (16, 64, 256, 6, 4, 5)),      # 1.21x, this GPU's 1024-row slice
        (8, (16, 64, 128, 6, 4, 6)),      # 1.17x, slice (sweep_all.py)
        (16, (16, 64, 256, 1, 4, 4)),     # 1.05x, all 8192 rows from here
        (32, (32, 64, 128, 1, 4, 6)),     # 1.03x
        (128, None),                      # 0.98x
        (256, None),                      # 0.89x
    ),
}
# The unpooler computes only this GPU's first 1024 rows of W at up to 8 tokens (TASK.md).
UNPOOL_SLICE, UNPOOL_SLICE_MAX_TOKENS = 1024, 8
SPLIT_COUNTERS = 1 << 16  # split-K tile counters, enough for any m these layers see

# The shapes the CUDA 12-bit GEMV serves, with its parameters (nf12_cuda_gemv's), from
# sweep_all.py (every backend per shape, timed as the judge does: graph replays, cold L2). It wins
# only at one token: there the GEMV is bound by its 12-bit reads, 25% fewer bytes than BF16's.
# From 8 tokens its m FMAs a weight on CUDA cores outrun the memory and the tensor cores win
# (Triton / cuBLAS, as BF16_CONFIGS has them). The speedups are over cuBLAS, judge-style.
# MINB = 2 (two blocks an SM, registers capped to fit) won on all four layers in the same run.
NF12_CUDA_DISPATCH = {
    ("qkv_gate", 1): dict(wpr=2, warps=4, rpw=1, unroll=2, ldhint=0, minb=2, xldg=0),   # 1.49x
    ("o", 1):        dict(wpr=1, warps=16, rpw=2, unroll=8, ldhint=0, minb=2, xldg=1),  # 1.18x
    ("pooler", 1):   dict(wpr=2, warps=16, rpw=2, unroll=4, ldhint=2, minb=2, xldg=0),  # 1.54x
    ("unpooler", 1): dict(wpr=8, warps=16, rpw=2, unroll=2, ldhint=2, minb=2, xldg=1),  # 1.77x, this GPU's 1024-row slice
}
# A layer is skipped if more than this fraction of its weights fall outside the 16-exponent window.
NF12_MAX_ESCAPE_FRACTION = 0.01


@triton.jit
def _router_fp32_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[:, n-block] = x @ W[n-block].T, BF16 x read as FP32, FP32 W, FP32 accumulation.

    Triton pipelines only loads that feed tl.dot, so this loop prefetches by hand: it starts
    the next W tile's load before summing the current one. Rows of x go one at a time, so
    registers hold [BLOCK_N, BLOCK_K] products, not [BLOCK_M, BLOCK_N, BLOCK_K]."""
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    w = tl.load(w_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < K), other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + offs_k
        # The next tile, in flight while this one is summed; fully masked (no read) past K.
        w_next = tl.load(w_ptrs + (k0 + BLOCK_K) * stride_wk,
                         mask=mask_n[:, None] & (ks[None, :] + BLOCK_K < K), other=0.0)
        for i in tl.static_range(BLOCK_M):
            x_i = tl.load(x_ptr + i * stride_xm + ks * stride_xk,
                          mask=(ks < K) & (i < M), other=0.0).to(tl.float32)
            # Plain FP32 multiply and sum: tl.dot on FP32 inputs would round to TF32.
            part = tl.sum(w * x_i[None, :], axis=1)
            acc += tl.where(offs_m[:, None] == i, part[None, :], 0.0)
        w = w_next

    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc,
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _router_hilo_kernel(
    x_ptr, hi_ptr, lo_ptr, out_ptr,
    M, N, K, K_PER_SPLIT,
    stride_xm, stride_xk,
    stride_hn, stride_hk,
    stride_ln, stride_lk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out += x @ (hi + lo).T over one K split: BF16 x, hi and lo on the tensor cores, FP32
    accumulation, atomically added into a zeroed FP32 out."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    k_start = pid_k * K_PER_SPLIT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        ks = k_start + k0 + offs_k
        mask_k = ks < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + ks[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        # hi and lo are [N, K]; load each tile as [BLOCK_K, BLOCK_N] so tl.dot needs no transpose.
        hi = tl.load(hi_ptr + offs_n[None, :] * stride_hn + ks[:, None] * stride_hk,
                     mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        lo = tl.load(lo_ptr + offs_n[None, :] * stride_ln + ks[:, None] * stride_lk,
                     mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(x, hi, acc)
        acc = tl.dot(x, lo, acc)

    tl.atomic_add(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc,
                  mask=mask_m[:, None] & mask_n[None, :], sem="relaxed")


@triton.jit
def _router_hilo_tma_kernel(
    x_desc, hi_desc, lo_desc, out_ptr, part_ptr, count_ptr,
    M, N, K_PER_SPLIT,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, TICKET: tl.constexpr,
):
    """_router_hilo_kernel for Hopper: TMA tile loads, and the product computed transposed.

    Each tile arrives by one TMA copy, which fills whatever lies past the tensor's edge (rows
    past M, columns past K) with zeros, so no load needs a mask. tl.dot computes out.T =
    W @ x.T, taking the BLOCK_N weight rows as its first dimension, which Hopper's wgmma needs
    to be at least 64; the BLOCK_M tokens need only be at least 16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    k_start = pid_k * K_PER_SPLIT

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)  # a tile of out.T
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        k = k_start + k0
        x_t = tl.trans(x_desc.load([m0, k]))             # [BLOCK_K, BLOCK_M]
        acc = tl.dot(hi_desc.load([n0, k]), x_t, acc)    # [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_M]
        acc = tl.dot(lo_desc.load([n0, k]), x_t, acc)

    # Stored transposed back: acc[j, i] is out[m0 + i, n0 + j].
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    offs = offs_n[:, None] * stride_on + offs_m[None, :] * stride_om
    if not TICKET:
        tl.atomic_add(out_ptr + offs, acc, mask=mask, sem="relaxed")
    else:
        # As _bf16_gemm_kernel's combine: partials + a ticket per tile, no zeroing, no atomics on out.
        splits = tl.num_programs(2)
        tl.store(part_ptr + pid_k * M * N + offs, acc, mask=mask)
        tl.debug_barrier()
        tile = pid_m * tl.num_programs(1) + pid_n
        ticket = tl.atomic_add(count_ptr + tile, 1, sem="acq_rel")
        if ticket == splits - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in range(splits):
                total += tl.load(part_ptr + s * M * N + offs, mask=mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + offs, total, mask=mask)
            tl.atomic_xchg(count_ptr + tile, 0)


@triton.jit
def _bf16_gemm_kernel(
    x_desc, w_desc, out_ptr, part_ptr, count_ptr,
    M, N, K_PER_SPLIT,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """out = x @ W.T in BF16 for few tokens, one launch: TMA tile loads and the product
    transposed for wgmma, as in _router_hilo_tma_kernel, with FP32 accumulation.

    With SPLIT_K > 1, each split stores its FP32 partial tile and takes a ticket from the tile's
    counter; the split that takes the last ticket sums the partials, writes the BF16 tile, and
    sets the counter back to 0 for the next call. So there is no zeroing or reduction kernel."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    k_start = pid_k * K_PER_SPLIT

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)  # a tile of out.T
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        k = k_start + k0
        acc = tl.dot(w_desc.load([n0, k]), tl.trans(x_desc.load([m0, k])), acc)

    # acc[j, i] is out[m0 + i, n0 + j]; partials are [SPLIT_K, M, N], each laid out like out.
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    offs = offs_n[:, None] * stride_on + offs_m[None, :] * stride_om
    if SPLIT_K == 1:
        tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(part_ptr + pid_k * M * N + offs, acc, mask=mask)
        tl.debug_barrier()  # every thread's partial stored before the ticket is taken
        tile = pid_m * tl.num_programs(1) + pid_n
        ticket = tl.atomic_add(count_ptr + tile, 1, sem="acq_rel")
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in range(SPLIT_K):
                # .cg reads from L2: other SMs wrote these, and L1 isn't coherent with them.
                total += tl.load(part_ptr + s * M * N + offs, mask=mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + offs, total.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(count_ptr + tile, 0)


# ----------------------------------------------------------------------------------------------
# The CUDA 12-bit kernels.
#
# Format (per layer, built in prepare, untimed). A BF16 weight is  s e7 e6 e5 e4 e3 e2 e1 | e0 m6..m0.
# Call E7 = the high byte without its sign (7 bits). Pick C0 so that [C0, C0+7] is the 8-value
# window of E7 holding the most weights (a 16-exponent window of the full exponent). A weight
# inside the window is stored as
#   lo   : its low byte, verbatim                              (8 bits)
#   code : s << 3 | (E7 - C0)                                  (4 bits, two to a byte)
# and rebuilt exactly as  high byte = (code & 8) << 4 | (C0 + (code & 7)),  bits = high << 8 | lo.
# A weight outside the window (exact zeros, outliers: ~0.01% here) is stored as lo = 0, code = 0,
# which rebuilds to the placeholder value bf16(C0 << 8); its true value minus that placeholder
# is kept in a per-row CSR list (esc_ptr, esc_k, esc_delta) and added in the kernel's epilogue.
#
# Layout: planes, for nf12_gemv (M <= 16 tokens, CUDA cores): lo [N, K] and nib [N, K/2], row-major.
#    One warp per row (WPR warps when N is small); a lane streams 16 weights a step: a 16-byte
#    load of lo, an 8-byte load of nib, M x two 16-byte loads of x (L1/L2 resident). Decode is
#    byte-parallel (prmt); FP32 FMAs; shuffle reduction. No shared memory, no barriers.
# ----------------------------------------------------------------------------------------------
NF12_CUDA_COMMON = r"""
typedef unsigned int  u32;
typedef unsigned short u16;
typedef unsigned char  u8;

__device__ __forceinline__ float bf_lo(u32 w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(u32 w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float bf1(u16 w) { return __uint_as_float(((u32)w) << 16); }

// Weight loads by LDHINT: 0 plain, 1 read-only path (__ldg), 2 streaming (__ldcs: evict first,
// the weights are read once per call).
#ifndef LDHINT
#define LDHINT 0
#endif
__device__ __forceinline__ uint4 ldw4(const void* p) {
#if LDHINT == 1
    return __ldg(reinterpret_cast<const uint4*>(p));
#elif LDHINT == 2
    return __ldcs(reinterpret_cast<const uint4*>(p));
#else
    return *reinterpret_cast<const uint4*>(p);
#endif
}
__device__ __forceinline__ uint2 ldw2(const void* p) {
#if LDHINT == 1
    return __ldg(reinterpret_cast<const uint2*>(p));
#elif LDHINT == 2
    return __ldcs(reinterpret_cast<const uint2*>(p));
#else
    return *reinterpret_cast<const uint2*>(p);
#endif
}

__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even
    u32 u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}

// 8 weights: L0, L1 their low bytes (4 each), n32 their 8 codes (two a byte, low nibble first).
// Out: four u32, each two BF16 (weight 2i in the low half, 2i+1 in the high half).
__device__ __forceinline__ void decode8(u32 L0, u32 L1, u32 n32, u32 c0x4,
                                        u32& w01, u32& w23, u32& w45, u32& w67) {
    u32 ce = n32 & 0x0F0F0F0Fu;            // codes 0 2 4 6
    u32 co = (n32 >> 4) & 0x0F0F0F0Fu;     // codes 1 3 5 7
    u32 cA = __byte_perm(ce, co, 0x5140);  // codes 0 1 2 3
    u32 cB = __byte_perm(ce, co, 0x7362);  // codes 4 5 6 7
    u32 hA = ((cA & 0x08080808u) << 4) | ((cA & 0x07070707u) + c0x4);  // high bytes 0..3
    u32 hB = ((cB & 0x08080808u) << 4) | ((cB & 0x07070707u) + c0x4);  // high bytes 4..7
    w01 = __byte_perm(L0, hA, 0x5140);     // lo0 hi0 lo1 hi1
    w23 = __byte_perm(L0, hA, 0x7362);
    w45 = __byte_perm(L1, hB, 0x5140);
    w67 = __byte_perm(L1, hB, 0x7362);
}
"""

NF12_GEMV_SRC = NF12_CUDA_COMMON + r"""
// M tokens (<= 16). RPW rows a warp, WPR warps a row (each a 1 / WPR slice of K, summed in
// shared memory), WARPS warps a block: WARPS / WPR * RPW rows a block. The K loop unrolled
// UNROLL chunks (512 weights a chunk a warp: 16 a lane). Each chunk's weights are widened to
// FP32 once (not once a token), and each token's x once and shared by the warp's RPW rows, so
// a weight costs ~1 FMA a token plus 1/RPW of x's widening: what keeps M = 8 near the memory.
#ifndef WARPS
#define WARPS 4
#endif
#ifndef UNROLL
#define UNROLL 4
#endif
#ifndef RPW
#define RPW 1
#endif
#ifndef MINB
#define MINB 1
#endif
#ifndef XLDG
#define XLDG 0
#endif
__device__ __forceinline__ uint4 ldx4(const u16* p) {
#if XLDG
    return __ldg(reinterpret_cast<const uint4*>(p));
#else
    return *reinterpret_cast<const uint4*>(p);
#endif
}
extern "C" __global__ void __launch_bounds__(32 * WARPS, MINB)
nf12_gemv(const u8* __restrict__ lo, const u8* __restrict__ nib, const u16* __restrict__ x,
          u16* __restrict__ out, const int* __restrict__ esc_ptr, const int* __restrict__ esc_k,
          const float* __restrict__ esc_delta, int N, int K, u32 c0x4)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * (WARPS / WPR) + warp / WPR) * RPW;  // this warp's first row
    const int wk   = warp % WPR;
    const int KW   = K / WPR;                      // this warp's K range
    const int kbase = wk * KW;
    const u8* lo_w  = lo  + (size_t)row0 * K + kbase;
    const u8* nib_w = nib + (size_t)row0 * (K >> 1) + (kbase >> 1);

    float acc[RPW][M];
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int t = 0; t < M; t++) acc[r][t] = 0.f;

    // The escape bounds, requested now so their misses overlap the weight stream (used after it).
    int esc_lo[RPW], esc_hi[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) { esc_lo[r] = esc_ptr[row0 + r]; esc_hi[r] = esc_ptr[row0 + r + 1]; }

    const int nchunks = KW >> 9;
    constexpr int kUnroll = UNROLL;
    #pragma unroll kUnroll
    for (int i = 0; i < nchunks; i++) {
        const int kk = (i << 9) + (lane << 4);
        // This chunk's 16 weights of each of the warp's rows, decoded and widened once.
        float wf[RPW][16];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            const uint4 L = ldw4(lo_w + (size_t)r * K + kk);
            const uint2 Nn = ldw2(nib_w + (size_t)r * (K >> 1) + (kk >> 1));
            u32 w[8];
            decode8(L.x, L.y, Nn.x, c0x4, w[0], w[1], w[2], w[3]);
            decode8(L.z, L.w, Nn.y, c0x4, w[4], w[5], w[6], w[7]);
            #pragma unroll
            for (int q = 0; q < 8; q++) { wf[r][2 * q] = bf_lo(w[q]); wf[r][2 * q + 1] = bf_hi(w[q]); }
        }
        #pragma unroll
        for (int t = 0; t < M; t++) {
            const u16* xr = x + (size_t)t * K + kbase + kk;
            const uint4 X0 = ldx4(xr);
            const uint4 X1 = ldx4(xr + 8);
            const u32 xs[8] = {X0.x, X0.y, X0.z, X0.w, X1.x, X1.y, X1.z, X1.w};
            float xf[16];
            #pragma unroll
            for (int q = 0; q < 8; q++) { xf[2 * q] = bf_lo(xs[q]); xf[2 * q + 1] = bf_hi(xs[q]); }
            #pragma unroll
            for (int r = 0; r < RPW; r++) {
                float a = acc[r][t];
                #pragma unroll
                for (int j = 0; j < 16; j++) a = fmaf(wf[r][j], xf[j], a);
                acc[r][t] = a;
            }
        }
    }

    // The escapes' correction: each row's warp 0, lanes over its entries.
    if (wk == 0) {
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            for (int e = esc_lo[r] + lane; e < esc_hi[r]; e += 32) {
                const int k = esc_k[e];
                const float d = esc_delta[e];
                #pragma unroll
                for (int t = 0; t < M; t++)
                    acc[r][t] = fmaf(d, bf1(x[(size_t)t * K + k]), acc[r][t]);
            }
        }
    }

    // Reduce over the warp's lanes.
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int t = 0; t < M; t++) {
            float v = acc[r][t];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            acc[r][t] = v;
        }

#if WPR == 1
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < RPW; r++)
            #pragma unroll
            for (int t = 0; t < M; t++) out[(size_t)t * N + row0 + r] = f2bf(acc[r][t]);
    }
#else
    __shared__ float red[WARPS][RPW][M];
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < RPW; r++)
            #pragma unroll
            for (int t = 0; t < M; t++) red[warp][r][t] = acc[r][t];
    }
    __syncthreads();
    if (wk == 0 && lane == 0) {
        const int w0 = warp - (warp % WPR);
        #pragma unroll
        for (int r = 0; r < RPW; r++)
            #pragma unroll
            for (int t = 0; t < M; t++) {
                float sum = 0.f;
                for (int j = 0; j < WPR; j++) sum += red[w0 + j][r][t];
                out[(size_t)t * N + row0 + r] = f2bf(sum);
            }
    }
#endif
}
"""

class _CudaKernel:
    """One compiled CUDA kernel: NVRTC through cuda-python, or CuPy, whichever imports.
    arg_types: ctypes types of the kernel's arguments, in order (pointers as c_void_p).
    launch() runs it on PyTorch's current stream."""

    def __init__(self, name: str, src: str, defines: dict, arg_types):
        self.name, self.arg_types = name, tuple(arg_types)
        src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + src
        self.backend = None
        errors = []
        for setup in (self._setup_cuda_python, self._setup_cupy):
            try:
                setup(src)
                break
            except Exception as e:  # noqa: BLE001 - any failure means "try the next backend"
                errors.append(f"{setup.__name__}: {type(e).__name__}: {str(e)[:400]}")
        if self.backend is None:
            raise RuntimeError("no CUDA backend: " + " | ".join(errors))

    @staticmethod
    def _arch():
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}" + ("a" if major >= 9 else "")

    # --- cuda-python (NVRTC + driver) -------------------------------------------------------
    def _setup_cuda_python(self, src: str):
        try:
            from cuda.bindings import driver as cu, nvrtc
        except ImportError:
            from cuda import cuda as cu, nvrtc  # cuda-python < 12.6

        def check(res):
            err, *rest = res
            if int(err) != 0:
                raise RuntimeError(f"cuda-python error {err}")
            return rest[0] if len(rest) == 1 else rest

        prog = check(nvrtc.nvrtcCreateProgram(src.encode(), f"{self.name}.cu".encode(), 0, [], []))
        opts = [f"--gpu-architecture={self._arch()}".encode(), b"-default-device", b"--std=c++17"]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(err) != 0:
            size = check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("nvrtc: " + log.decode(errors="replace")[:1500])
        try:
            size = check(nvrtc.nvrtcGetCUBINSize(prog))
            image = b" " * size
            check(nvrtc.nvrtcGetCUBIN(prog, image))
        except Exception:  # noqa: BLE001 - fall back to PTX
            size = check(nvrtc.nvrtcGetPTXSize(prog))
            image = b" " * size
            check(nvrtc.nvrtcGetPTX(prog, image))
        check(cu.cuInit(0))
        torch.cuda.current_stream().synchronize()  # the primary context exists
        self._module = check(cu.cuModuleLoadData(image))
        self._func = check(cu.cuModuleGetFunction(self._module, self.name.encode()))
        self._cu = cu
        self._smem_set = 48 * 1024
        self.backend = "cuda-python"

    def _launch_cuda_python(self, grid, block, args, smem):
        cu = self._cu
        if smem > self._smem_set:  # dynamic shared memory over 48 KB needs opting into
            err, = cu.cuFuncSetAttribute(
                self._func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            if int(err) != 0:
                raise RuntimeError(f"cuFuncSetAttribute: {err}")
            self._smem_set = smem
        params = (tuple(int(a) for a in args), self.arg_types)
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        err, = cu.cuLaunchKernel(self._func, *grid, block, 1, 1, smem, stream, params, 0)
        if int(err) != 0:
            raise RuntimeError(f"cuLaunchKernel: {err}")

    # --- CuPy -----------------------------------------------------------------------------
    def _setup_cupy(self, src: str):
        import cupy as cp
        self._cp = cp
        mod = cp.RawModule(code=src, options=("--std=c++17", f"--gpu-architecture={self._arch()}"))
        self._kernel = mod.get_function(self.name)
        self.backend = "cupy"

    def _launch_cupy(self, grid, block, args, smem):
        cp = self._cp
        import numpy as np
        conv = []
        for a, ty in zip(args, self.arg_types):
            if ty is ctypes.c_void_p:
                conv.append(cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(int(a), 0, None), 0))
            elif ty is ctypes.c_uint:
                conv.append(np.uint32(a))
            else:
                conv.append(np.int32(a))
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        if smem > 48 * 1024:
            self._kernel.max_dynamic_shared_size_bytes = smem
        with stream:
            self._kernel(grid, (block, 1, 1), tuple(conv), stream=stream, shared_mem=smem)

    def launch(self, grid, block: int, args, smem: int = 0):
        """grid: blocks, an int or (x, y, z); smem: dynamic shared memory bytes."""
        grid = (grid, 1, 1) if isinstance(grid, int) else tuple(grid) + (1,) * (3 - len(grid))
        if self.backend == "cuda-python":
            self._launch_cuda_python(grid, block, args, smem)
        else:
            self._launch_cupy(grid, block, args, smem)


_GEMV_ARGS = (ctypes.c_void_p,) * 7 + (ctypes.c_int, ctypes.c_int, ctypes.c_uint)


def nf12_pack_cuda(w: torch.Tensor):
    """BF16 W [N, K] -> its 12-bit planes (lo [N, K], nib [N, K/2]), the escape CSR, and C0;
    None if too many weights fall outside the best 16-exponent window. Checked lossless, bit
    for bit."""
    n, k = w.shape
    if k % 512 or n % 16:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e7 = (bits >> 8) & 0x7F
    counts = torch.bincount(e7.flatten(), minlength=128).to(torch.int64)
    windows = counts.unfold(0, 8, 1).sum(1)            # windows [c, c + 8) for c in 0..120
    c0 = int(windows.argmax())
    inside = (e7 >= c0) & (e7 < c0 + 8)
    n_esc = int((~inside).sum())
    if n_esc > NF12_MAX_ESCAPE_FRACTION * bits.numel():
        return None
    code = torch.where(inside, ((bits >> 15) << 3) | (e7 - c0), torch.zeros_like(bits))
    lo = torch.where(inside, bits & 0xFF, torch.zeros_like(bits))
    lo8 = lo.to(torch.uint8).contiguous()
    nib = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).contiguous()
    # The decode, in torch: inside weights rebuild bit for bit; escapes rebuild to the placeholder.
    c = torch.stack([nib.to(torch.int32) & 0xF, nib.to(torch.int32) >> 4], dim=-1).reshape(n, k)
    hb = ((c & 8) << 4) | (c0 + (c & 7))
    rebuilt = (hb << 8) | lo8.to(torch.int32)
    placeholder_bits = c0 << 8
    expect = torch.where(inside, bits, torch.full_like(bits, placeholder_bits))
    assert torch.equal(rebuilt, expect), "nf12_pack_cuda: decode mismatch"
    placeholder = torch.tensor([placeholder_bits], dtype=torch.int16).view(torch.bfloat16).float().item()
    rows, ks = (~inside).nonzero(as_tuple=True)        # row-major order: a CSR by row
    per_row = torch.bincount(rows, minlength=n)
    esc_ptr = torch.zeros(n + 1, dtype=torch.int32, device=w.device)
    esc_ptr[1:] = torch.cumsum(per_row, 0).to(torch.int32)
    esc_delta = (w[rows, ks].float() - placeholder).contiguous()
    packed = {"lo": lo8, "nib": nib, "c0x4": c0 * 0x01010101, "esc_ptr": esc_ptr,
              "esc_k": ks.to(torch.int32).contiguous(), "esc_delta": esc_delta, "escapes": n_esc, "c0": c0}
    return packed


def _nf12_gemv_wprs(n: int, k: int, warps: int = 4, rpw: int = 1):
    """Warps a row the GEMV can run with `warps` a block and `rpw` rows a warp: K / WPR whole
    512-weight chunks, whole rows a block, N whole blocks."""
    return [wpr for wpr in (1, 2, 4, 8) if wpr <= warps and warps % wpr == 0 and (k // wpr) % 512 == 0
            and n % (warps // wpr * rpw) == 0]


# The CUDA GEMV's tuning knobs' defaults: WARPS, UNROLL, LDHINT, RPW (with WPR, chosen per shape).
NF12_GEMV_DEFAULTS = dict(warps=4, unroll=4, ldhint=0, rpw=1, minb=1, xldg=0)
NF12_GEMV_MAX_TOKENS = 16


def _nf12_cuda_kernel(state, kind: str, m: int, **params):
    """The compiled kernel for (kind, m, params), compiled on first use: in prepare() or a sweep,
    never mid-graph (kernel_fn only runs configs prepare() compiled). None if it won't compile."""
    assert kind == "gemv" and m <= NF12_GEMV_MAX_TOKENS, (kind, m)
    params = {**NF12_GEMV_DEFAULTS, **params}
    key = (kind, m) + tuple(sorted(params.items()))
    kernels = state["nf12_kernels"]
    if key not in kernels:
        src, defines, args = NF12_GEMV_SRC, {"M": m, "WPR": params["wpr"], "RPW": params["rpw"]}, _GEMV_ARGS
        defines.update(WARPS=params["warps"], UNROLL=params["unroll"], LDHINT=params["ldhint"],
                       MINB=params["minb"], XLDG=params["xldg"])
        try:
            kernels[key] = _CudaKernel(f"nf12_{kind}", src, defines, args)
        except Exception as e:  # noqa: BLE001
            print(f"nf12 cuda: compile {key} failed: {str(e)[:600]}")
            kernels[key] = None
    return kernels[key]


def nf12_cuda_gemv(x, layer, n, state, wpr, warps=4, unroll=4, ldhint=0, rpw=1, minb=1, xldg=0):
    """x [m, K] BF16 @ W[:n].T -> [m, n] BF16 on nf12_gemv (M = m <= 16, WPR = wpr warps a row,
    WARPS = warps a block, RPW = rpw rows a warp, UNROLL, LDHINT)."""
    m, k = x.shape
    p = state["nf12"][layer]
    kernel = _nf12_cuda_kernel(state, "gemv", m, wpr=wpr, warps=warps, unroll=unroll, ldhint=ldhint,
                               rpw=rpw, minb=minb, xldg=xldg)
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    kernel.launch(n // (warps // wpr * rpw), 32 * warps, (
        p["lo"].data_ptr(), p["nib"].data_ptr(), x.data_ptr(), out.data_ptr(),
        p["esc_ptr"].data_ptr(), p["esc_k"].data_ptr(), p["esc_delta"].data_ptr(), n, k, p["c0x4"]))
    return out


def _nf12_cuda_setup(state, device):
    """Packs the layers NF12_CUDA_DISPATCH uses into 12 bits, compiles each entry's GEMV, checks
    it against an FP32 product on random input, and serves the ones that pass:
    state["dispatch"][(layer, m, n)] -> callable(x). A shape without an entry (or whose entry
    fails to pack, compile or check) runs the existing Triton / cuBLAS path."""
    state["nf12"], state["nf12_kernels"], state["dispatch"] = {}, {}, {}
    for layer in sorted({layer for layer, _ in NF12_CUDA_DISPATCH}):
        try:
            packed = nf12_pack_cuda(state[layer])
        except Exception as e:  # noqa: BLE001
            print(f"nf12 cuda: {layer}: pack failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        if packed is None:
            print(f"nf12 cuda: {layer}: doesn't fit the window, skipped")
            continue
        state["nf12"][layer] = packed
        print(f"nf12 cuda: {layer}: c0 {packed['c0']}, {packed['escapes']} escapes")

    gen = torch.Generator(device=device).manual_seed(0)
    for (layer, m), params in sorted(NF12_CUDA_DISPATCH.items()):
        if layer not in state["nf12"]:
            continue
        k = state[layer].shape[1]
        n = UNPOOL_SLICE if layer == "unpooler" and m <= UNPOOL_SLICE_MAX_TOKENS else state[layer].shape[0]
        if params["wpr"] not in _nf12_gemv_wprs(n, k, params.get("warps", 4), params.get("rpw", 1)):
            print(f"nf12 cuda: {layer} m={m} {params}: can't run at n={n}, k={k}")
            continue
        if _nf12_cuda_kernel(state, "gemv", m, **params) is None:
            continue
        fn = lambda x, layer=layer, n=n, params=params: nf12_cuda_gemv(x, layer, n, state, **params)
        x = torch.randn((m, k), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
        ref = x.float() @ state[layer][:n].float().T
        tol = 2e-2 * ref.abs().max().item() + 1e-3
        try:
            got = fn(x).float()
            torch.cuda.synchronize()
            err = (got - ref).abs().max().item()
        except Exception as e:  # noqa: BLE001
            print(f"nf12 cuda: {layer} m={m}: failed: {type(e).__name__}: {str(e)[:200]}")
            torch.cuda.synchronize()
            continue
        if not (torch.isfinite(got).all().item() and err <= tol):
            print(f"nf12 cuda: {layer} m={m}: WRONG, max abs err {err:.3e} (tol {tol:.3e}); not served")
            continue
        state["dispatch"][(layer, m, n)] = fn


def router_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [m, k] BF16 (m <= ROUTER_FP32_MAX_TOKENS) @ w [n, k] FP32 .T -> [m, n] FP32, one launch."""
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(n, ROUTER_BLOCK_N),)
    _router_fp32_kernel[grid](
        x, w, out,
        m, n, k,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=max(2, triton.next_power_of_2(m)), BLOCK_N=ROUTER_BLOCK_N, BLOCK_K=ROUTER_BLOCK_K,
        num_warps=ROUTER_NUM_WARPS, num_stages=ROUTER_NUM_STAGES,
    )
    return out


def _router_hilo_config(m: int, configs=ROUTER_HILO_CONFIGS):
    """(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) for m, from a config table."""
    for max_m, config in configs:
        if m <= max_m:
            return config
    return configs[-1][1]


def router_hilo(x: torch.Tensor, state) -> torch.Tensor:
    """x [m, k] BF16 @ (hi + lo) [n, k] BF16 .T -> [m, n] FP32: the TMA kernel when Triton has
    tensor descriptors, else plain loads."""
    if ROUTER_USE_TMA:
        return router_hilo_tma(x, state)
    return router_hilo_ptr(x, state["router_hi"], state["router_lo"])


def router_hilo_tma(x: torch.Tensor, state) -> torch.Tensor:
    """router_hilo by _router_hilo_tma_kernel: a zeroing and one split-K launch."""
    m, k = x.shape
    hi, lo = state["router_hi"], state["router_lo"]
    n = hi.shape[0]
    block_m, block_n, block_k, split_k, num_warps, num_stages, combine = (
        *_router_hilo_config(m, ROUTER_HILO_TMA_CONFIGS), "atomic")[:7]
    descs = state["router_descs"]  # hi and lo descriptors by tile shape, each made once
    if (block_n, block_k) not in descs:
        descs[(block_n, block_k)] = (TensorDescriptor.from_tensor(hi, [block_n, block_k]),
                                     TensorDescriptor.from_tensor(lo, [block_n, block_k]))
    hi_desc, lo_desc = descs[(block_n, block_k)]
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k])
    # Each split's K range is a whole number of BLOCK_K steps, so splits never overlap.
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    if combine == "ticket":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        part = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
        assert grid[0] * grid[1] <= SPLIT_COUNTERS
    else:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        part = out
    _router_hilo_tma_kernel[grid](
        x_desc, hi_desc, lo_desc, out, part, state["split_counts"],
        m, n, k_per_split,
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, TICKET=(combine == "ticket"),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def router_hilo_ptr(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    """router_hilo by _router_hilo_kernel, with plain loads: a zeroing and one split-K launch."""
    m, k = x.shape
    n = hi.shape[0]
    block_m, block_n, block_k, split_k, num_warps, num_stages = _router_hilo_config(m)
    # Each split's K range is a whole number of BLOCK_K steps, so splits never overlap.
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    _router_hilo_kernel[grid](
        x, hi, lo, out,
        m, n, k, k_per_split,
        x.stride(0), x.stride(1),
        hi.stride(0), hi.stride(1),
        lo.stride(0), lo.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _bf16_config(layer: str, m: int):
    """The Triton config for this layer at exactly m, or None for cuBLAS."""
    if not BF16_USE_TRITON:
        return None
    return dict(BF16_CONFIGS.get(layer, ())).get(m)


def bf16_gemm(x: torch.Tensor, w: torch.Tensor, layer: str, state, config) -> torch.Tensor:
    """x [m, k] BF16 @ w [n, k] BF16 .T -> [m, n] BF16 by _bf16_gemm_kernel, one launch."""
    m, k = x.shape
    n = w.shape[0]
    block_m, block_n, block_k, split_k, num_warps, num_stages = config
    descs = state["bf16_descs"]  # W descriptors by layer, width and tile shape, each made once
    key = (layer, n, block_n, block_k)
    if key not in descs:
        descs[key] = TensorDescriptor.from_tensor(w, [block_n, block_k])
    x_desc, w_desc = TensorDescriptor.from_tensor(x, [block_m, block_k]), descs[key]
    # Each split's K range is a whole number of BLOCK_K steps, so splits never overlap.
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device) if split_k > 1 else out
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    assert grid[0] * grid[1] <= SPLIT_COUNTERS, "more output tiles than split-K counters"
    _bf16_gemm_kernel[grid](
        x_desc, w_desc, out, part, state["split_counts"],
        m, n, k_per_split,
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _warm_ms(rows):
    """The m each Triton row runs at: each row serves exactly its own m."""
    return sorted(m for m, config in rows if config is not None)


def prepare(weights: Weights):
    # In the run's log: the versions to pin the sweeps' Modal images to.
    print(f"torch {torch.__version__}, triton {triton.__version__}, cuda {torch.version.cuda}, "
          f"TMA descriptors {TensorDescriptor is not None}")
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()  # [256, 8192], the same layout as the FP32 router
    state["router_lo"] = lo.contiguous()
    state["router_descs"] = {}
    state["split_counts"] = torch.zeros(SPLIT_COUNTERS, dtype=torch.int32, device=router.device)
    # Compile the router kernels now, so no call compiles mid-graph: every m of the FP32 path,
    # and each hi + lo config at m both a multiple of 16 and not (Triton specializes on it).
    zeros = lambda m: torch.zeros((m, router.shape[1]), dtype=torch.bfloat16, device=router.device)
    for m in range(1, ROUTER_FP32_MAX_TOKENS + 1):
        router_fp32(zeros(m), router)
    for m in (3, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256):
        router_hilo(zeros(m), state)
    # The BF16 layers' Triton configs: split-K tile counters start at 0 and each call leaves them so.
    state["bf16_descs"] = {}
    for layer, rows in BF16_CONFIGS.items():
        for m in _warm_ms(rows):
            n = UNPOOL_SLICE if layer == "unpooler" and m <= UNPOOL_SLICE_MAX_TOKENS else state[layer].shape[0]
            x = torch.zeros((m, state[layer].shape[1]), dtype=torch.bfloat16, device=router.device)
            bf16_gemm(x, state[layer][:n], layer, state, _bf16_config(layer, m))
    # The CUDA 12-bit kernels, where they compile, check out and win; else the paths above serve.
    try:
        _nf12_cuda_setup(state, router.device)
    except Exception as e:  # noqa: BLE001
        print(f"nf12 cuda: setup failed, disabled: {type(e).__name__}: {str(e)[:300]}")
        state["dispatch"] = {}
    print(f"nf12 cuda: serving {sorted(state['dispatch'])}")
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        n = data["config"]["n"]
        m = x.shape[0]
        fast = state["dispatch"].get((layer, m, n))
        if fast is not None:
            return fast(x)
        # The unpooler's slice is its first n rows (all of them above 8 tokens).
        w = state[layer][:n]
        config = _bf16_config(layer, m)
        if config is not None:
            return bf16_gemm(x, w, layer, state, config)
        # cuBLAS, as SGLang's bf16_gemm_dispatch falls through to on H200.
        return F.linear(x, w)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return router_fp32(x, state["router"])
    return router_hilo(x, state)
