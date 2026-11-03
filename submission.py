# okc: problem=muse_bf16_proj
"""The dense projections. What serves each shape (kernel_fn):
- BF16 layers at one token: a lossless 12-bit copy of W read by a CUDA GEMV (nf12_gemv).
- BF16 layers where a hand-written Hopper kernel wins (TMA_DISPATCH, by m range): wa_gemm, TMA +
  wgmma with the K split inside a thread-block cluster.
- Elsewhere the Triton split-K kernel (BF16_CONFIGS) or cuBLAS, as SGLang runs it.
- The router: Triton, FP32 up to 2 tokens, BF16 hi + lo above.
The CUDA kernels compile in prepare() with NVRTC (cuda-python); each is checked against an FP32
product before serving, and any that fails to compile or check leaves its shapes on Triton /
cuBLAS."""

import ctypes
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor  # Hopper TMA descriptors (Triton 3.4+)


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

# hi + lo kernel configs (TMA), swept on an H200 (sweep_all.py): (max m, (BLOCK_M, BLOCK_N,
# BLOCK_K, SPLIT_K, num_warps, num_stages)). Each m takes the first row whose max m >= m; the
# last row covers everything above. tl.dot takes the BLOCK_N weight rows as its first dimension,
# which Hopper's wgmma needs to be at least 64; the BLOCK_M tokens need only be at least 16. The
# splits add into a zeroed output with atomics (a partials + ticket combine was 2.5-6 us slower;
# hand-written TMA + wgmma kernels, sweep_router.py in the archive, were 0.82-0.91x of these).
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
def _router_hilo_tma_kernel(
    x_desc, hi_desc, lo_desc, out_ptr,
    M, N, K_PER_SPLIT,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out += x @ (hi + lo).T over one K split: BF16 x, hi and lo on the tensor cores, FP32
    accumulation, atomically added into a zeroed FP32 out. TMA tile loads, the product computed
    transposed.

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
    tl.atomic_add(out_ptr + offs, acc, mask=mask, sem="relaxed")


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
        """grid: blocks, an int or (x, y, z); smem: dynamic shared memory bytes. Cluster kernels
        carry their cluster shape in the compiled code (__cluster_dims__): a plain launch."""
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


# Hand-written Hopper BF16 GEMMs (wa_gemm) for the shapes where cuBLAS / Triton leave the GPU
# under-used (qkv_gate: N = 2304 rows of W make few tiles for 132 SMs). A warpgroup takes 64 rows
# of W as wgmma's A and the BT tokens as its N (m64nBTk16, BT any multiple of 8, so few tokens
# waste nothing). A cluster of CS CTAs splits K and sums its partial tiles through distributed
# shared memory: one launch, no global partials or counters. x comes by TMA (128-byte swizzle,
# kept in L2: every tile reads it; without the hint the W stream evicts it, 25% slower); W by
# plain bulk copies from a copy packed in prepare() (per tile [K / 64][rows][64], the swizzle
# applied), so a CTA's W is one contiguous run. A producer warp fills a STAGES-deep ring guarded
# by mbarriers. From dev_qkv.py / sweep_cuda.py (Modal H200, graph replays, cold L2): loads bound
# it (loads-only == full kernel); clusters must all fit at once (H200: 39 of 3 CTAs, 30 of 4,
# 15 of 8), and the DSMEM reduce's cost grows with the tile and cluster.
HOPPER_HELPERS_SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
struct __align__(64) TmaDesc { u64 d[16]; };


__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 bar, u32 count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_2d(u32 dst, const TmaDesc* desc, int c0, int c1, u32 bar, u64 policy) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
                 :: "r"(dst), "l"((u64)desc), "r"(c0), "r"(c1), "r"(bar), "l"(policy) : "memory");
}
__device__ __forceinline__ void bulk_load(u32 dst, const void* src, u32 bytes, u32 bar, u64 policy) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 :: "r"(dst), "l"((u64)src), "r"(bytes), "r"(bar), "l"(policy) : "memory");
}
__device__ __forceinline__ u64 policy_evict_first() { u64 p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p)); return p; }
__device__ __forceinline__ u64 policy_evict_last() { u64 p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p)); return p; }
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ u32 cluster_rank() { u32 r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// A peer's shared memory (DSMEM). Not volatile, so a sum's CS loads issue back to back; the
// memory clobber keeps them after the cluster barrier.
__device__ __forceinline__ float4 ld_peer_f4(u32 local_addr, u32 rank) {
    u32 remote; float4 v;
    asm("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank));
    asm("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(remote) : "memory");
    return v;
}
__device__ __forceinline__ u32 pack_bf16(float a, float b) {
    u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(r) : "f"(a), "f"(b)); return r;
}
// K-major, 128-byte swizzle: rows of 128 bytes, 8-row groups 1024 bytes apart (SBO); LBO unused.
__device__ __forceinline__ u64 desc_sw128(u32 addr) {
    return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}

"""
WA_GEMM_SRC = HOPPER_HELPERS_SRC + r"""
#define BN (64 * WG)
#define A_BYTES (BN * 128)                      // W stage: BN rows x 64 k
#define B_BYTES (BT * 128)                      // x stage: BT tokens x 64 k
#define THREADS (128 * (WG + 1))
#define RS (BN + 4)                             // row stride (floats) of the parked tile [BT][RS]

//@WGMMAS@

extern "C" __global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(THREADS, 1)
wa_gemm(const __grid_constant__ TmaDesc tmX, const u16* __restrict__ wp, u16* __restrict__ out, int M, int N, int K)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
    constexpr u32 RING = STAGES * (A_BYTES + B_BYTES), REDB = BT * RS * 4;
    const u32 sA = smem_u32(smem);                  // [STAGES][BN][128 B]
    const u32 sB = sA + STAGES * A_BYTES;           // [STAGES][BT][128 B]
    const u32 full = sA + (RING > REDB ? RING : REDB), empty = full + 8 * STAGES;
    float* red = reinterpret_cast<float*>(smem);    // [BT][RS] after the main loop
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2;
    const u32 rank = CS > 1 ? cluster_rank() : 0;
    const int tile = blockIdx.x / CS, n0 = tile * BN;
    const int nkb = K / 64;
    const int kb0 = (int)(rank * nkb) / CS, kb1 = (int)((rank + 1) * nkb) / CS;
    const int nk = kb1 - kb0;
    const unsigned char* wsrc = reinterpret_cast<const unsigned char*>(wp) + ((size_t)tile * nkb + kb0) * A_BYTES;

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, 4 * WG); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    float d[BT / 2];
    #pragma unroll
    for (int i = 0; i < BT / 2; i++) d[i] = 0.f;

    if (wg == WG) {
        if (threadIdx.x == 128 * WG) {   // producer: W read once (evict first), x by every tile (keep)
            const u64 pw = policy_evict_first(), px = policy_evict_last();
            for (int i = 0; i < nk; i++) {
                const int s = i % STAGES;
                mbar_wait(empty + 8 * s, ((i / STAGES) & 1) ^ 1);
                mbar_expect_tx(full + 8 * s, A_BYTES + B_BYTES);
                bulk_load(sA + s * A_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full + 8 * s, pw);
                tma_load_2d(sB + s * B_BYTES, &tmX, (kb0 + i) * 64, 0, full + 8 * s, px);
            }
        }
    } else {
        for (int i = 0; i < nk; i++) {
            const int s = i % STAGES;
            mbar_wait(full + 8 * s, (i / STAGES) & 1);
            const u32 a = sA + s * A_BYTES + wg * (64 * 128);
            const u32 b = sB + s * B_BYTES;
            wgmma_fence();
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) wgmma_n<BT>(d, desc_sw128(a + 32 * kk), desc_sw128(b + 32 * kk));
            wgmma_commit();
            wgmma_wait<1>();   // the previous step's wgmmas are done: release its stage
            if (i > 0 && lane == 0) mbar_arrive(empty + 8 * ((i - 1) % STAGES));
        }
        wgmma_wait<0>();
    }
    __syncthreads();

    if (wg < WG) {
        // d: warp (warp & 3) holds W rows 16 (warp & 3) + lane / 4 (+8); d[4j..4j+3]: tokens 8j + 2 (lane % 4), +1.
        const int r = wg * 64 + (warp & 3) * 16 + (lane >> 2), t = 2 * (lane & 3);
        #pragma unroll
        for (int j = 0; j < BT / 8; j++) {
            const int c = 8 * j + t;
            red[c * RS + r] = d[4 * j];           red[(c + 1) * RS + r] = d[4 * j + 1];
            red[c * RS + r + 8] = d[4 * j + 2];   red[(c + 1) * RS + r + 8] = d[4 * j + 3];
        }
    }
    if (CS > 1) cluster_sync(); else __syncthreads();
    constexpr int Q = BN / 4;                    // float4s a token row
    constexpr int ITEMS = BT * Q;
    constexpr int IT = ((ITEMS + CS - 1) / CS + THREADS - 1) / THREADS;
    const int e0 = (int)(rank * ITEMS) / CS, e1 = (int)((rank + 1) * ITEMS) / CS;
    float4 v[IT][CS];
    #pragma unroll
    for (int it = 0; it < IT; it++) {
        const int e = e0 + threadIdx.x + it * THREADS;
        const u32 addr = smem_u32(red + (e / Q) * RS + 4 * (e % Q));
        #pragma unroll
        for (int p = 0; p < CS; p++) v[it][p] = e < e1 ? ld_peer_f4(addr, p) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    #pragma unroll
    for (int it = 0; it < IT; it++) {
        const int e = e0 + threadIdx.x + it * THREADS;
        const int c = e / Q, q = e % Q;
        if (e >= e1 || c >= M) continue;
        float4 s = v[it][0];
        #pragma unroll
        for (int p = 1; p < CS; p++) { s.x += v[it][p].x; s.y += v[it][p].y; s.z += v[it][p].z; s.w += v[it][p].w; }
        uint2 o; o.x = pack_bf16(s.x, s.y); o.y = pack_bf16(s.z, s.w);
        *reinterpret_cast<uint2*>(out + (size_t)c * N + n0 + 4 * q) = o;
    }
    if (CS > 1) cluster_sync();   // no CTA leaves while a peer may still read its shared memory
}
"""

# The shapes wa_gemm serves, from sweep_cuda.py (a grid over tile, cluster and ring depth, every
# config checked; Modal H200, judge-style timing): (layer, first m, last m) -> parameters, for
# every m in the range. Each m runs a kernel with BT = m rounded up to 8 tokens (TMA zero-fills
# the rest; a box mostly past x's edge is slow, so no wider). Speedups over the path that served
# the range before (Triton / cuBLAS), pseudo-judge.
#   wg: warpgroups a CTA (64 * wg rows of W), cs: CTAs a cluster (K splits), stages: ring depth
#   (capped to fit shared memory at the range's BT).
TMA_DISPATCH = {
    ("qkv_gate", 2, 8):    dict(wg=2, cs=4, stages=11),   # 1.14 -> 1.19-1.21x
    ("qkv_gate", 9, 16):   dict(wg=1, cs=3, stages=8),    # 1.11 -> 1.17-1.20x
    ("qkv_gate", 17, 32):  dict(wg=2, cs=4, stages=9),    # 1.10 -> 1.14-1.16x
    ("qkv_gate", 33, 64):  dict(wg=1, cs=3, stages=12),   # 1.00 -> 1.00-1.10x
    ("qkv_gate", 65, 128): dict(wg=1, cs=3, stages=8),    # 1.00 -> 1.15x at 128
    ("unpooler", 2, 8):    dict(wg=2, cs=8, stages=6),    # 1.13 -> 1.26-1.32x, the 1024-row slice
}
TMA_SMEM = 220 * 1024

def _wgmma_n_src(ns) -> str:
    """wgmma_n<N>(d, desc_a, desc_b): one m64nNk16 BF16 -> FP32 wgmma into d[N / 2], per N used."""
    out = ["template <int N> __device__ __forceinline__ void wgmma_n(float* d, u64 da, u64 db);"]
    for n in sorted(set(ns)):
        nacc = n // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        out.append(
            f"template <> __device__ __forceinline__ void wgmma_n<{n}>(float* d, u64 da, u64 db) {{\n"
            f'    asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nacc + 2}, 0;\\n"\n'
            f'                 "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, '
            f'%{nacc}, %{nacc + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'                 : {outs} : "l"(da), "l"(db), "r"(1));\n}}')
    return "\n".join(out)


def pack_w_tiles(w: torch.Tensor, bn: int) -> torch.Tensor:
    """W [n, K] -> per tile of bn rows: [K / 64][bn][64], each block in TMA's 128-byte swizzle
    (16-byte chunk c of row r stored at chunk c ^ (r % 8)), the bytes a 2D TMA box would land.
    Lossless: the same BF16 values, reordered."""
    n, k = w.shape
    r = torch.arange(bn, device=w.device)[:, None]
    src = (torch.arange(8, device=w.device)[None, :] ^ (r % 8))[None, None, :, :, None]
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)       # [tile][kb][r][chunk][8]
    return torch.gather(t, 3, src.expand(t.shape)).contiguous()


class TmaGemm:
    """wa_gemm compiled for (wg, cs, stages, bt); __call__(x, wp) runs it on W packed by
    pack_w_tiles(w, 64 * wg). x's TMA descriptor is encoded on the host each call (it holds x's
    address, no values)."""

    def __init__(self, n, wg, cs, stages, bt):
        try:
            from cuda.bindings import driver as cu
        except ImportError:
            from cuda import cuda as cu  # cuda-python < 12.6
        bn = 64 * wg
        self.cu, self.n, self.bt, self.threads = cu, n, bt, 128 * (wg + 1)
        self.ctas = (n // bn) * cs
        self.kernel = _CudaKernel("wa_gemm", WA_GEMM_SRC.replace("//@WGMMAS@", _wgmma_n_src([bt])),
                                  dict(WG=wg, CS=cs, STAGES=stages, BT=bt), (ctypes.c_void_p,) * 3 + (ctypes.c_int,) * 3)
        if self.kernel.backend != "cuda-python":
            raise RuntimeError("wa_gemm needs cuda-python (TMA descriptor arguments)")
        self.smem = max(stages * (bn + bt) * 128, bt * (bn + 4) * 4) + 16 * stages + 1024
        err, = cu.cuFuncSetAttribute(self.kernel._func,
                                     cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem)
        if int(err) != 0:
            raise RuntimeError(f"cuFuncSetAttribute: {err}")

    def _x_desc(self, x: torch.Tensor):
        """A TMA descriptor over x [m, K]: boxes of bt tokens x 64 k, 128-byte swizzle; tokens past
        m read as zeros."""
        cu = self.cu
        m, k = x.shape
        err, desc = cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(),
            [cu.cuuint64_t(k), cu.cuuint64_t(m)], [cu.cuuint64_t(k * 2)],
            [cu.cuuint32_t(64), cu.cuuint32_t(self.bt)], [cu.cuuint32_t(1), cu.cuuint32_t(1)],
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if int(err) != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled: {err}")
        return desc

    def __call__(self, x: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        assert m <= self.bt and x.is_contiguous()
        out = torch.empty((m, self.n), dtype=torch.bfloat16, device=x.device)
        cu = self.cu
        params = ((self._x_desc(x), wp.data_ptr(), out.data_ptr(), m, self.n, k),
                  (None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int))
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        err, = cu.cuLaunchKernel(self.kernel._func, self.ctas, 1, 1, self.threads, 1, 1, self.smem, stream, params, 0)
        if int(err) != 0:
            raise RuntimeError(f"cuLaunchKernel: {err}")
        return out


def _tma_setup(state, device):
    """For each TMA_DISPATCH range: packs W once, compiles a kernel per 8-token step of the range,
    checks each against an FP32 product at both ends of its step, and serves
    state["dispatch"][(layer, m, n)] with it. A step that fails leaves its m on Triton / cuBLAS."""
    gen = torch.Generator(device=device).manual_seed(1)
    for (layer, m_lo, m_hi), p in sorted(TMA_DISPATCH.items()):
        n = UNPOOL_SLICE if layer == "unpooler" and m_hi <= UNPOOL_SLICE_MAX_TOKENS else state[layer].shape[0]
        w = state[layer][:n]
        k = w.shape[1]
        wp = pack_w_tiles(w, 64 * p["wg"])
        for bt in range((m_lo + 7) // 8 * 8, m_hi + 1, 8):
            lo_m = max(m_lo, bt - 7)
            stages = min(p["stages"], TMA_SMEM // ((64 * p["wg"] + bt) * 128))
            try:
                g = TmaGemm(n, p["wg"], p["cs"], stages, bt)
                err = 0.0
                for m in (lo_m, bt):
                    x = torch.randn((m, k), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
                    ref = x.float() @ w.float().T
                    got = g(x, wp).float()
                    torch.cuda.synchronize()
                    if not torch.isfinite(got).all().item():
                        raise RuntimeError(f"non-finite output at m={m}")
                    err = max(err, ((got - ref).norm() / ref.norm()).item())
                if err >= 4e-3:
                    raise RuntimeError(f"WRONG, rel err {err:.2e}")
            except Exception as e:  # noqa: BLE001
                print(f"tma gemm: {layer} m={lo_m}..{bt} {p}: not served: {type(e).__name__}: {str(e)[:300]}")
                torch.cuda.synchronize()
                continue
            for m in range(lo_m, bt + 1):
                state["dispatch"][(layer, m, n)] = lambda x, g=g, wp=wp: g(x, wp)
        served = sorted(m for (l, m, _) in state["dispatch"] if l == layer and m_lo <= m <= m_hi)
        print(f"tma gemm: {layer} m={m_lo}..{m_hi} {p}: serving {len(served)} of {m_hi - m_lo + 1} m")


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


def _router_hilo_config(m: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) for m."""
    for max_m, config in ROUTER_HILO_TMA_CONFIGS:
        if m <= max_m:
            return config
    return ROUTER_HILO_TMA_CONFIGS[-1][1]


def router_hilo(x: torch.Tensor, state) -> torch.Tensor:
    """x [m, k] BF16 @ (hi + lo) [n, k] BF16 .T -> [m, n] FP32: a zeroing and one split-K launch."""
    m, k = x.shape
    hi, lo = state["router_hi"], state["router_lo"]
    n = hi.shape[0]
    block_m, block_n, block_k, split_k, num_warps, num_stages = _router_hilo_config(m)
    descs = state["router_descs"]  # hi and lo descriptors by tile shape, each made once
    if (block_n, block_k) not in descs:
        descs[(block_n, block_k)] = (TensorDescriptor.from_tensor(hi, [block_n, block_k]),
                                     TensorDescriptor.from_tensor(lo, [block_n, block_k]))
    hi_desc, lo_desc = descs[(block_n, block_k)]
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k])
    # Each split's K range is a whole number of BLOCK_K steps, so splits never overlap.
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    _router_hilo_tma_kernel[grid](
        x_desc, hi_desc, lo_desc, out,
        m, n, k_per_split,
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _bf16_config(layer: str, m: int):
    """The Triton config for this layer at exactly m, or None for cuBLAS."""
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
    try:
        _tma_setup(state, router.device)
    except Exception as e:  # noqa: BLE001
        print(f"tma gemm: setup failed, disabled: {type(e).__name__}: {str(e)[:300]}")
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
