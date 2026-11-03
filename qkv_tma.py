"""Hand-written Hopper BF16 GEMM for the short-wide shapes (qkv_gate at m=128 first).

out[m, n] = x[m, K] @ W[n, K].T. Swapped onto wgmma: a consumer warpgroup takes 64 rows of W as
wgmma's M and all BT tokens as its N (m64nBTk16), both operands K-major in shared memory, landed
by TMA with the 128-byte swizzle wgmma reads directly. One producer warp issues the TMA loads into
a STAGES-deep ring guarded by mbarriers. A cluster of CS CTAs splits K; after the main loop each
CTA parks its FP32 tile in its own shared memory and, after one cluster barrier, sums a 1/CS slice
of the tile across the cluster (ld.shared::cluster), converts to BF16 and stores it. One launch.
"""

import ctypes

import torch

SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
struct __align__(64) TmaDesc { u64 d[16]; };

#ifndef NOMMA
#define NOMMA 0
#endif
#define BK 64
#ifndef NOX
#define NOX 0
#endif
#ifndef NOW
#define NOW 0
#endif
#ifndef PACKW
#define PACKW 0
#endif
#ifndef G
#define G 1
#endif
#ifndef EMPTY
#define EMPTY 0
#endif
#define BN (64 * WG)
#ifndef KSUB
#define KSUB 1                 // 64-wide k-blocks a stage (x then comes by a 3D tensor map, W packed)
#endif
#define A_SUB (BN * BK * 2)
#define B_SUB (BT * BK * 2)
#define A_BYTES (A_SUB * KSUB)
#define B_BYTES (B_SUB * KSUB)
#ifndef WST
#define WST STAGES             // W ring depth
#endif
#ifndef XST
#define XST STAGES             // x ring depth
#endif
#define THREADS (128 * (WG + 1))
#define NACC (BT / 2)
#define RS (BN + 4)            // row stride (floats) of the parked FP32 tile [BT][BN]: no bank conflicts

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
    asm volatile(
        "{\n.reg .pred P;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n" :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_2d(u32 dst, const TmaDesc* desc, int c0, int c1, u32 bar, u64 policy) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
        :: "r"(dst), "l"((u64)desc), "r"(c0), "r"(c1), "r"(bar), "l"(policy) : "memory");
}
// A contiguous run of global memory into shared memory, completing on an mbarrier (no tensor map).
__device__ __forceinline__ void bulk_load(u32 dst, const void* src, u32 bytes, u32 bar, u64 policy) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
        :: "r"(dst), "l"((u64)src), "r"(bytes), "r"(bar), "l"(policy) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TmaDesc* desc, int c0, int c1, int c2, u32 bar, u64 policy) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;\n"
        :: "r"(dst), "l"((u64)desc), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(policy) : "memory");
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
__device__ __forceinline__ u32 peer_addr(u32 local_addr, u32 rank) {
    u32 remote; asm("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank)); return remote;
}
// Not volatile: the loads of a sum's CS peers issue back to back, not one round trip each. Ordered
// after the cluster barrier by its memory clobber.
__device__ __forceinline__ float4 ld_peer_f4(u32 local_addr, u32 rank) {
    float4 v;
    asm("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(peer_addr(local_addr, rank)) : "memory");
    return v;
}
__device__ __forceinline__ float4 ld_local_f4(u32 addr) {
    float4 v;
    asm("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr) : "memory");
    return v;
}
__device__ __forceinline__ u32 pack_bf16(float a, float b) {
    u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(r) : "f"(a), "f"(b)); return r;
}
// K-major, 128-byte swizzle: rows of 128 bytes, 8-row groups 1024 bytes apart (SBO); LBO unused.
__device__ __forceinline__ u64 desc_sw128(u32 addr) {
    return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}

//@WGMMA@

extern "C" __global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(THREADS, 1)
qkv_tma_gemm(const __grid_constant__ TmaDesc tmW, const __grid_constant__ TmaDesc tmX,
             const u16* __restrict__ wp, u16* __restrict__ out, float4* __restrict__ ws, u32* __restrict__ cnt,
             int M, int N, int K)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const u32 raw = smem_u32(smem_raw);
    unsigned char* smem = smem_raw + ((1024 - (raw & 1023)) & 1023);
    const u32 sA = smem_u32(smem);                       // [WST][BN rows][128 B]
    const u32 sB = sA + WST * A_BYTES;                   // [XST][BT rows][128 B]
    const u32 bars = sB + XST * B_BYTES;
    // mbarriers: W full[WST], W empty[WST], x full[XST], x empty[XST]
    const u32 wfull = bars, wempty = bars + 8 * WST, xfull = bars + 16 * WST, xempty = xfull + 8 * XST;
    float* red = reinterpret_cast<float*>(smem);         // [BT][RS] after the main loop

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2;
    const u32 rank = CS > 1 ? cluster_rank() : 0;
    // CTAs: [tile][G groups][CS cluster ranks]; K splits into G * CS parts, part g * CS + rank.
    const int tile = blockIdx.x / (CS * G), g = (blockIdx.x / CS) % G;
    const int n0 = tile * BN;
    const int nkb = K / (BK * KSUB);   // stages along all of K
    const int part = g * CS + rank;
    const int kb0 = (part * nkb) / (G * CS), kb1 = ((part + 1) * nkb) / (G * CS);
    const int nk = EMPTY ? 0 : kb1 - kb0;
    // PACKW: W repacked [N / BN][K / BK][BN rows][BK] with the 128-byte swizzle applied, so a CTA's
    // whole K range is one contiguous run and each stage one bulk copy.
    const unsigned char* wtile = reinterpret_cast<const unsigned char*>(wp) + ((size_t)tile * nkb + kb0) * A_BYTES;

    if (threadIdx.x == 0) {
        for (int s = 0; s < WST; s++) { mbar_init(wfull + 8 * s, 1); mbar_init(wempty + 8 * s, 4 * WG); }
        for (int s = 0; s < XST; s++) { mbar_init(xfull + 8 * s, 1); mbar_init(xempty + 8 * s, 4 * WG); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    float d[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; i++) d[i] = 0.f;

    if (wg == WG) {
        // Producers, one thread each, on separate rings so W can run further ahead than x: W (read
        // once: evict first) and x (read by every n-tile: keep).
        if (threadIdx.x == 128 * WG && !NOW) {
            const u64 pw = policy_evict_first();
            for (int i = 0; i < nk; i++) {
                const int s = i % WST;
                mbar_wait(wempty + 8 * s, ((i / WST) & 1) ^ 1);
                mbar_expect_tx(wfull + 8 * s, A_BYTES);
                if (PACKW) bulk_load(sA + s * A_BYTES, wtile + (size_t)i * A_BYTES, A_BYTES, wfull + 8 * s, pw);
                else tma_load_2d(sA + s * A_BYTES, &tmW, (kb0 + i) * BK, n0, wfull + 8 * s, pw);
            }
        }
        if (threadIdx.x == 128 * WG + 32 && !NOX) {
            const u64 px = policy_evict_last();
            for (int i = 0; i < nk; i++) {
                const int s = i % XST;
                mbar_wait(xempty + 8 * s, ((i / XST) & 1) ^ 1);
                mbar_expect_tx(xfull + 8 * s, B_BYTES);
                if (KSUB == 1) tma_load_2d(sB + s * B_BYTES, &tmX, (kb0 + i) * BK, 0, xfull + 8 * s, px);
                else tma_load_3d(sB + s * B_BYTES, &tmX, 0, 0, (kb0 + i) * KSUB, xfull + 8 * s, px);
            }
        }
    } else {
        for (int i = 0; i < nk; i++) {
            if (!NOW) mbar_wait(wfull + 8 * (i % WST), (i / WST) & 1);
            if (!NOX) mbar_wait(xfull + 8 * (i % XST), (i / XST) & 1);
            const u32 a = sA + (i % WST) * A_BYTES + wg * (64 * 128);
            const u32 b = sB + (i % XST) * B_BYTES;
            wgmma_fence();
            #pragma unroll
            for (int j = 0; j < KSUB; j++)
                #pragma unroll
                for (int kk = 0; kk < BK / 16; kk++)
                    wgmma_tile(d, desc_sw128(a + j * A_SUB + 32 * kk), desc_sw128(b + j * B_SUB + 32 * kk));
            wgmma_commit();
            wgmma_wait<1>();   // the previous step's wgmmas are done: release its stages
            if (i > 0 && lane == 0) {
                mbar_arrive(wempty + 8 * ((i - 1) % WST));
                mbar_arrive(xempty + 8 * ((i - 1) % XST));
            }
        }
        wgmma_wait<0>();
    }
    __syncthreads();   // every stage consumed: the ring is free for the parked tile

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
    // This CTA's 1/CS slice of the cluster's sum, from every CTA's parked tile (fixed order).
    float4 acc[IT];
    #pragma unroll
    for (int it = 0; it < IT; it++) {
        const int e = e0 + threadIdx.x + it * THREADS;
        acc[it] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (e < e1) {
            const u32 addr = smem_u32(red + (e / Q) * RS + 4 * (e % Q));
            #pragma unroll
            for (int p = 0; p < CS; p++) {
                const float4 v = ld_peer_f4(addr, p);
                acc[it].x += v.x; acc[it].y += v.y; acc[it].z += v.z; acc[it].w += v.w;
            }
            if (G > 1) __stcg(ws + (size_t)(tile * G + g) * ITEMS + e, acc[it]);
        }
    }
    bool last = true;
    if (G > 1) {
        // The G groups of a tile meet in global memory: each parks its slice, and the last to
        // arrive (a counter per tile and rank) adds the others' in group order, so the sum is the
        // same whoever is last; it resets the counter for the next call.
        __shared__ u32 s_last;
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            const u32 t = atomicAdd(cnt + tile * CS + rank, 1u);
            s_last = (t == G - 1);
            if (t == G - 1) { cnt[tile * CS + rank] = 0; __threadfence(); }
        }
        __syncthreads();
        last = s_last;
    }
    if (last) {
        #pragma unroll
        for (int it = 0; it < IT; it++) {
            const int e = e0 + threadIdx.x + it * THREADS;
            const int c = e / Q, q = e % Q;
            if (e >= e1 || c >= M) continue;
            float4 s = acc[it];
            if (G > 1) {
                s = make_float4(0.f, 0.f, 0.f, 0.f);
                #pragma unroll
                for (int gg = 0; gg < G; gg++) {
                    const float4 v = gg == g ? acc[it] : __ldcg(ws + (size_t)(tile * G + gg) * ITEMS + e);
                    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
                }
            }
            uint2 o; o.x = pack_bf16(s.x, s.y); o.y = pack_bf16(s.z, s.w);
            *reinterpret_cast<uint2*>(out + (size_t)c * N + n0 + 4 * q) = o;
        }
    }
    if (CS > 1) cluster_sync();   // no CTA leaves while a peer may still read its shared memory
}
"""


def _wgmma_asm(bt: int) -> str:
    nacc = bt // 2
    regs = ", ".join(f"%{i}" for i in range(nacc))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
    return ("__device__ __forceinline__ void wgmma_tile(float* d, u64 da, u64 db) {\n"
            "    if (NOMMA) { d[0] += (float)(da ^ db); return; }\n"
            f'    asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nacc + 2}, 0;\\n"\n'
            f'                 "wgmma.mma_async.sync.aligned.m64n{bt}k16.f32.bf16.bf16 {{{regs}}}, '
            f'%{nacc}, %{nacc + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'                 : {outs} : "l"(da), "l"(db), "r"(1));\n}}')


def _driver():
    try:
        from cuda.bindings import driver as cu, nvrtc
    except ImportError:
        from cuda import cuda as cu, nvrtc
    return cu, nvrtc


def _check(res):
    err, *rest = res
    if int(err) != 0:
        raise RuntimeError(f"cuda error {err}")
    return rest[0] if len(rest) == 1 else (rest or None)


class TmaGemm:
    """The compiled kernel for one (WG, CS, STAGES, BT); __call__(x, w, out) launches it."""

    def __init__(self, wg=2, cs=7, stages=6, bt=128, nomma=0, g=1, src=SRC, name="qkv_tma_gemm", extra=None):
        self.g = g
        extra = dict(extra or {}, G=g)
        self.ksub = (extra or {}).get("KSUB", 1)
        self.packw = bool((extra or {}).get("PACKW", 0)) or self.ksub > 1
        if self.ksub > 1:
            extra = dict(extra, PACKW=1)
        self.packed = {}
        cu, nvrtc = _driver()
        self.cu, self.wg, self.cs, self.stages, self.bt, self.name = cu, wg, cs, stages, bt, name
        defines = dict(WG=wg, CS=cs, STAGES=stages, BT=bt, NOMMA=nomma, **extra)
        code = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + src.replace("//@WGMMA@", _wgmma_asm(bt))
        prog = _check(nvrtc.nvrtcCreateProgram(code.encode(), b"k.cu", 0, [], []))
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-default-device", b"--std=c++17"]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(err) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("nvrtc: " + log.decode(errors="replace")[:3000])
        size = _check(nvrtc.nvrtcGetCUBINSize(prog))
        image = b" " * size
        _check(nvrtc.nvrtcGetCUBIN(prog, image))
        _check(cu.cuInit(0))
        torch.cuda.current_stream().synchronize()
        self.module = _check(cu.cuModuleLoadData(image))
        self.func = _check(cu.cuModuleGetFunction(self.module, name.encode()))
        self.threads = 128 * (wg + 1)
        bn = 64 * wg
        wst = (extra or {}).get("WST", stages)
        xst = (extra or {}).get("XST", stages)
        pipe = (wst * bn + xst * bt) * 64 * 2 * self.ksub
        self.smem = max(pipe, bt * (bn + 4) * 4) + 16 * (wst + xst) + 1024
        A = cu.CUfunction_attribute
        _check(cu.cuFuncSetAttribute(self.func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))
        if cs > 8:
            _check(cu.cuFuncSetAttribute(self.func, A.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1))
        self.descs = {}
        self.regs = _check(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func))
        self.spill = _check(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.func))

    def max_clusters(self, n_tiles):
        cu = self.cu
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = n_tiles * self.cs * self.g, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = self.threads, 1, 1
        cfg.sharedMemBytes = self.smem
        attr = cu.CUlaunchAttribute()
        attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = self.cs, 1, 1
        cfg.attrs, cfg.numAttrs = [attr], 1
        return _check(cu.cuOccupancyMaxActiveClusters(self.func, cfg))

    def _desc(self, t: torch.Tensor, box_rows: int, ksub: int = 1):
        key = (t.data_ptr(), tuple(t.shape), box_rows, ksub)
        d = self.descs.get(key)
        if d is None and ksub > 1:
            # [rows, K] as [K / 64][rows][64]: a box of ksub k-blocks x box_rows rows x 64 lands as
            # ksub consecutive [box_rows][64] swizzled tiles.
            cu = self.cu
            rows, k = t.shape
            d = _check(cu.cuTensorMapEncodeTiled(
                cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(),
                [cu.cuuint64_t(64), cu.cuuint64_t(rows), cu.cuuint64_t(k // 64)], [cu.cuuint64_t(k * 2), cu.cuuint64_t(128)],
                [cu.cuuint32_t(64), cu.cuuint32_t(box_rows), cu.cuuint32_t(ksub)], [cu.cuuint32_t(1)] * 3,
                cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
                cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
            self.descs[key] = d
        if d is None:
            cu = self.cu
            rows, k = t.shape
            d = _check(cu.cuTensorMapEncodeTiled(
                cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(),
                [cu.cuuint64_t(k), cu.cuuint64_t(rows)], [cu.cuuint64_t(k * 2)],
                [cu.cuuint32_t(64), cu.cuuint32_t(box_rows)], [cu.cuuint32_t(1), cu.cuuint32_t(1)],
                cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
                getattr(cu.CUtensorMapL2promotion, "CU_TENSOR_MAP_L2_PROMOTION_" + getattr(self, "l2promo", "L2_256B")),
                cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
            self.descs[key] = d
        return d

    def pack(self, w: torch.Tensor) -> torch.Tensor:
        """W [n, K] -> [n / BN][K / 64][BN][64], each [BN][64] tile in TMA's 128-byte swizzle (16-byte
        chunk c of row r at chunk c ^ (r % 8)): the bytes a 2D TMA box would land, stored in order."""
        key = (w.data_ptr(), tuple(w.shape))
        p = self.packed.get(key)
        if p is None:
            n, k = w.shape
            bn = 64 * self.wg
            t = w.view(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)   # [nt][kb][r][chunk][8]
            r = torch.arange(bn, device=w.device)[:, None]
            c = torch.arange(8, device=w.device)[None, :]
            src = (c ^ (r % 8))                                              # chunk stored at c holds src chunk c ^ (r % 8)
            t = torch.gather(t, 3, src[None, None, :, :, None].expand(t.shape))
            p = t.contiguous()
            self.packed[key] = p
        return p

    def __call__(self, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor = None):
        m, k = x.shape
        n = w.shape[0]
        assert m <= self.bt and n % (64 * self.wg) == 0 and k % 64 == 0
        if out is None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        cu = self.cu
        dw, dx = self._desc(w, 64 * self.wg), self._desc(x, self.bt, self.ksub)
        tiles = n // (64 * self.wg)
        grid = tiles * self.cs * self.g
        wp = self.pack(w) if self.packw else w
        if getattr(self, "_ws_n", None) != n:
            self.ws = torch.empty(tiles * self.g * self.bt * 64 * self.wg, dtype=torch.float32, device=x.device)
            self.cnt = torch.zeros(tiles * self.cs, dtype=torch.int32, device=x.device)
            self._ws_n = n
        params = ((dw, dx, wp.data_ptr(), out.data_ptr(), self.ws.data_ptr(), self.cnt.data_ptr(), m, n, k),
                  (None, None) + (ctypes.c_void_p,) * 4 + (ctypes.c_int,) * 3)
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(cu.cuLaunchKernel(self.func, grid, 1, 1, self.threads, 1, 1, self.smem, stream, params, 0))
        return out


# ------------------------------------------------------------------------------------------------
# Orientation 2: x is wgmma's A (two consumer warpgroups, 64 tokens each) and a tile of W its B
# (N = the tile's rows, any multiple of 8). The tiles can then be cut to fill the GPU: NT1 tiles of
# BN1 rows then the rest of BN2 rows, a cluster of CS CTAs each, each CTA a 1/CS of K. W is packed
# per tile ([tile rows / 8 ...] at byte n0 * K * 2: [K / 64][BN rows][64], 128-byte swizzle), so a
# CTA's W is one contiguous run. Partials are summed across the cluster through DSMEM as above.
HELPERS = SRC[: SRC.index("//@WGMMA@")]

SRC_XA = HELPERS + r"""
#define XBYTES (BT * 128)                       // x stage: BT tokens x 64 k
#define BNMAX (BN1 > BN2 ? BN1 : BN2)
#define WSLOT (((BNMAX * 128) + 1023) / 1024 * 1024)
#ifndef KW
#define KW 1               // intra-block split-K: KW groups of two consumer warpgroups take alternate stages
#endif
#define NWG (BT / 64)              // token warpgroups (64 tokens each)
#define XTHREADS (128 * (NWG * KW + 1))
#define XRS (BNMAX + 4)
#ifndef WPOL
#define WPOL 1
#endif
#ifndef XPOL
#define XPOL 1
#endif
__device__ __forceinline__ u64 policy_normal() { u64 p; asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(p)); return p; }
#ifndef PUSH
#define PUSH 0
#endif
#define SROWS ((BT + CS - 1) / CS)                     // most token rows a cluster rank owns
#define RED_BYTES ((BT * XRS * 4 + 1023) / 1024 * 1024)
#define SLOT_BYTES (SROWS * XRS * 4)
#ifndef LOCALONLY
#define LOCALONLY 0
#endif

//@WGMMAS@

template <int BNT>
__device__ __forceinline__ void xa_run(const TmaDesc* tmX, const unsigned char* __restrict__ wp, u16* __restrict__ out,
                                       int M, int N, int K, int n0, u32 rank, unsigned char* smem)
{
    const u32 sX = smem_u32(smem);
    const u32 sW = sX + STAGES * XBYTES;
    // Barriers past both the ring and (PUSH) the parked tile + receive slots that reuse it.
    constexpr u32 RING = STAGES * (XBYTES + WSLOT), PUSHB = RED_BYTES + (CS - 1) * SLOT_BYTES;
    constexpr u32 KWB = KW * BT * XRS * 4, R2 = RING > PUSHB ? RING : PUSHB;
    const u32 full = sX + (R2 > KWB ? R2 : KWB), empty = full + 8 * STAGES;
    float* red = reinterpret_cast<float*>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2;
    constexpr u32 WB = BNT * 128;               // W stage bytes
    const int nkb = K / 64;
    const int kb0 = (int)(rank * nkb) / CS, kb1 = (int)((rank + 1) * nkb) / CS;
    const int nk = EMPTY ? 0 : kb1 - kb0;
    const unsigned char* wsrc = wp + (size_t)n0 * K * 2 + (size_t)kb0 * WB;

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, 4 * NWG); }
        mbar_init(full + 16 * STAGES, 1);   // PUSH: the peers' slices have landed
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    float d[BNT / 2];
    #pragma unroll
    for (int i = 0; i < BNT / 2; i++) d[i] = 0.f;

    if (wg == NWG * KW) {
        if (threadIdx.x == 128 * NWG * KW) {
            const u64 pw = WPOL == 1 ? policy_evict_first() : policy_normal();
            const u64 px = XPOL == 1 ? policy_evict_last() : policy_normal();
            for (int i = 0; i < nk; i++) {
                const int s = i % STAGES;
                mbar_wait(empty + 8 * s, ((i / STAGES) & 1) ^ 1);
                mbar_expect_tx(full + 8 * s, XBYTES + WB);
                bulk_load(sW + s * WSLOT, wsrc + (size_t)i * WB, WB, full + 8 * s, pw);
                tma_load_2d(sX + s * XBYTES, tmX, (kb0 + i) * 64, 0, full + 8 * s, px);
            }
        }
    } else {
        const int kg = wg / NWG, wt = wg % NWG;   // K group, token quarter
        for (int i = kg; i < nk; i += KW) {
            const int s = i % STAGES;
            mbar_wait(full + 8 * s, (i / STAGES) & 1);
            const u32 a = sX + s * XBYTES + wt * (64 * 128);
            const u32 b = sW + s * WSLOT;
            wgmma_fence();
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) wgmma_n<BNT>(d, desc_sw128(a + 32 * kk), desc_sw128(b + 32 * kk));
            wgmma_commit();
            wgmma_wait<1>();
            if (i >= KW && lane == 0) mbar_arrive(empty + 8 * ((i - KW) % STAGES));
        }
        wgmma_wait<0>();
    }
    __syncthreads();
    if (KW > 1) {
        // K groups 1.. park their partials; group 0 (warpgroups 0, 1) adds them in group order.
        const int kg = wg / NWG, r = (wg % NWG) * 64 + (warp & 3) * 16 + (lane >> 2), c = 2 * (lane & 3);
        if (kg >= 1 && kg < KW) {
            float* g = red + (size_t)kg * BT * XRS;
            #pragma unroll
            for (int j = 0; j < BNT / 8; j++) {
                *reinterpret_cast<float2*>(g + r * XRS + 8 * j + c) = make_float2(d[4 * j], d[4 * j + 1]);
                *reinterpret_cast<float2*>(g + (r + 8) * XRS + 8 * j + c) = make_float2(d[4 * j + 2], d[4 * j + 3]);
            }
        }
        __syncthreads();
        if (kg == 0) {
            for (int q = 1; q < KW; q++) {
                const float* g = red + (size_t)q * BT * XRS;
                #pragma unroll
                for (int j = 0; j < BNT / 8; j++) {
                    const float2 u = *reinterpret_cast<const float2*>(g + r * XRS + 8 * j + c);
                    const float2 v = *reinterpret_cast<const float2*>(g + (r + 8) * XRS + 8 * j + c);
                    d[4 * j] += u.x; d[4 * j + 1] += u.y; d[4 * j + 2] += v.x; d[4 * j + 3] += v.y;
                }
            }
        }
    }

    if (wg < NWG) {
        // d: tokens 64 wg + 16 (warp & 3) + lane / 4 (+8); d[4j..4j+3]: W rows 8j + 2 (lane % 4), +1.
        const int r = wg * 64 + (warp & 3) * 16 + (lane >> 2), c = 2 * (lane & 3);
        #pragma unroll
        for (int j = 0; j < BNT / 8; j++) {
            *reinterpret_cast<float2*>(red + r * XRS + 8 * j + c) = make_float2(d[4 * j], d[4 * j + 1]);
            *reinterpret_cast<float2*>(red + (r + 8) * XRS + 8 * j + c) = make_float2(d[4 * j + 2], d[4 * j + 3]);
        }
    }
    if (PUSH && CS > 1) {
        // Each rank owns token rows [r0, r1) of the tile. After one cluster barrier (every tile
        // parked, every ring idle), each CTA bulk-copies the rows it holds of every peer's slice
        // into that peer's receive slot; the owner adds them to its own rows from local memory.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        cluster_sync();
        const u32 rbar = full + 16 * STAGES;
        const u32 recv = smem_u32(smem) + RED_BYTES;
        const int r0 = (int)(rank * BT) / CS, r1 = (int)((rank + 1) * BT) / CS;
        if (threadIdx.x == 0) {
            mbar_expect_tx(rbar, (u32)((CS - 1) * (r1 - r0) * XRS * 4));
        }
        if (threadIdx.x == 32) {
            for (int p = 1; p < CS; p++) {
                const u32 o = (rank + p) % CS;                       // the owner
                const int o0 = (int)(o * BT) / CS, o1 = (int)((o + 1) * BT) / CS;
                const u32 src = smem_u32(red + o0 * XRS);
                const u32 dst = peer_addr(recv + (u32)(CS - p - 1) * SLOT_BYTES, o);   // slot of sender rank at owner o
                asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                             :: "r"(dst), "r"(src), "r"((u32)((o1 - o0) * XRS * 4)), "r"(peer_addr(rbar, o)) : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        }
        mbar_wait(rbar, 0);
        constexpr int Q = BNT / 4;
        const int items = (r1 - r0) * Q;
        for (int e = threadIdx.x; e < items; e += XTHREADS) {
            const int rr = e / Q, q = e % Q, t = r0 + rr;
            float4 s = *reinterpret_cast<const float4*>(red + t * XRS + 4 * q);
            #pragma unroll
            for (int j = 0; j < CS - 1; j++) {
                const float4 v = *reinterpret_cast<const float4*>(smem + RED_BYTES + j * SLOT_BYTES + (rr * XRS + 4 * q) * 4);
                s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
            }
            if (t < M) {
                uint2 o; o.x = pack_bf16(s.x, s.y); o.y = pack_bf16(s.z, s.w);
                *reinterpret_cast<uint2*>(out + (size_t)t * N + n0 + 4 * q) = o;
            }
        }
        if (threadIdx.x == 32) asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        return;
    }
#ifdef NORED
    if (NORED == 3) { if (CS > 1) { cluster_sync(); cluster_sync(); } return; }
    if (NORED == 2) { if (CS > 1) { cluster_sync(); cluster_sync(); } }
    return;
#endif
    if (CS > 1) cluster_sync(); else __syncthreads();
    constexpr int Q = BNT / 4;
    constexpr int ITEMS = BT * Q;
    const int e0 = (int)(rank * ITEMS) / CS, e1 = (int)((rank + 1) * ITEMS) / CS;
    constexpr int XIT = ((ITEMS + CS - 1) / CS + XTHREADS - 1) / XTHREADS;
    float4 v[XIT][CS];
    #pragma unroll
    for (int it = 0; it < XIT; it++) {   // every peer load in flight before any sum
        const int e = e0 + threadIdx.x + it * XTHREADS;
        const u32 addr = smem_u32(red + (e / Q) * XRS + 4 * (e % Q));
        #pragma unroll
        for (int p = 0; p < CS; p++) v[it][p] = e < e1 ? (LOCALONLY ? ld_local_f4(addr) : ld_peer_f4(addr, p)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    #pragma unroll
    for (int it = 0; it < XIT; it++) {
        const int e = e0 + threadIdx.x + it * XTHREADS;
        const int t = e / Q, q = e % Q;
        if (e >= e1 || t >= M) continue;
        float4 s = v[it][0];
        #pragma unroll
        for (int p = 1; p < CS; p++) { s.x += v[it][p].x; s.y += v[it][p].y; s.z += v[it][p].z; s.w += v[it][p].w; }
        uint2 o; o.x = pack_bf16(s.x, s.y); o.y = pack_bf16(s.z, s.w);
        *reinterpret_cast<uint2*>(out + (size_t)t * N + n0 + 4 * q) = o;
    }
    if (CS > 1) cluster_sync();
}

extern "C" __global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(XTHREADS, 1)
xa_gemm(const __grid_constant__ TmaDesc tmX, const u16* __restrict__ wp, u16* __restrict__ out, int M, int N, int K)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
    const u32 rank = CS > 1 ? cluster_rank() : 0;
    const int cl = blockIdx.x / CS;
    const unsigned char* w = reinterpret_cast<const unsigned char*>(wp);
    if (cl < NT1) { xa_run<BN1>(&tmX, w, out, M, N, K, cl * BN1, rank, smem); return; }
#if BN2 > 0
    xa_run<BN2>(&tmX, w, out, M, N, K, NT1 * BN1 + (cl - NT1) * BN2, rank, smem);
#endif
}
"""


def _wgmma_n_asm(ns) -> str:
    out = ["template <int N> __device__ __forceinline__ void wgmma_n(float* d, u64 da, u64 db);"]
    for n in sorted(set(ns)):
        nacc = n // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        out.append(
            f"template <> __device__ __forceinline__ void wgmma_n<{n}>(float* d, u64 da, u64 db) {{\n"
            "    if (NOMMA) { d[0] += (float)(da ^ db); return; }\n"
            f'    asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nacc + 2}, 0;\\n"\n'
            f'                 "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, '
            f'%{nacc}, %{nacc + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'                 : {outs} : "l"(da), "l"(db), "r"(1));\n}}')
    return "\n".join(out)


class XaGemm(TmaGemm):
    """x as A, W tiles as B: tiles = [(rows, count), ...] covering n (at most two sizes)."""

    def __init__(self, tiles=((96, 24),), cs=4, stages=8, nomma=0, extra=None, l2promo="L2_256B", bt=128):
        self.l2promo = l2promo
        cu, nvrtc = _driver()
        self.cu, self.cs, self.stages, self.bt, self.name = cu, cs, stages, bt, "xa_gemm"
        self.tiles = [tuple(t) for t in tiles]
        (bn1, nt1), (bn2, nt2) = self.tiles[0], (self.tiles[1] if len(self.tiles) > 1 else (0, 0))
        self.n_total = bn1 * nt1 + bn2 * nt2
        self.ctas = (nt1 + nt2) * cs
        self.wg, self.g = 1, 1  # for the bench's printouts
        defines = dict(dict(CS=cs, BN1=bn1, NT1=nt1, BN2=bn2, STAGES=stages, BT=bt, NOMMA=nomma, EMPTY=0), **(extra or {}))
        code = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + SRC_XA.replace(
            "//@WGMMAS@", _wgmma_n_asm([b for b in (bn1, bn2) if b]))
        self._build(code)
        bnmax = max(bn1, bn2)
        wslot = (bnmax * 128 + 1023) // 1024 * 1024
        kw = (extra or {}).get("KW", 1)
        self.threads = 128 * ((bt // 64) * kw + 1)
        red = (bt * (bnmax + 4) * 4 + 1023) // 1024 * 1024
        push = max(red + (cs - 1) * ((bt + cs - 1) // cs) * (bnmax + 4) * 4, kw * bt * (bnmax + 4) * 4)
        self.smem = max(stages * (bt * 128 + wslot), push) + 16 * stages + 16 + 1024
        A = cu.CUfunction_attribute
        _check(cu.cuFuncSetAttribute(self.func, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))
        self.descs, self.packed = {}, {}
        self.regs = _check(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func))
        self.spill = _check(cu.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.func))

    def _build(self, code):
        cu, nvrtc = _driver()
        prog = _check(nvrtc.nvrtcCreateProgram(code.encode(), b"k.cu", 0, [], []))
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-default-device", b"--std=c++17"]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(err) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("nvrtc: " + log.decode(errors="replace")[:3000])
        size = _check(nvrtc.nvrtcGetCUBINSize(prog))
        image = b" " * size
        _check(nvrtc.nvrtcGetCUBIN(prog, image))
        _check(cu.cuInit(0))
        torch.cuda.current_stream().synchronize()
        self.module = _check(cu.cuModuleLoadData(image))
        self.func = _check(cu.cuModuleGetFunction(self.module, self.name.encode()))

    def max_clusters(self, _n_tiles=None):
        nt = sum(c for _, c in self.tiles)
        cs, self.cs_saved = self.cs, None
        cu = self.cu
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = nt * cs, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = self.threads, 1, 1
        cfg.sharedMemBytes = self.smem
        attr = cu.CUlaunchAttribute()
        attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = cs, 1, 1
        cfg.attrs, cfg.numAttrs = [attr], 1
        return _check(cu.cuOccupancyMaxActiveClusters(self.func, cfg))

    def pack(self, w: torch.Tensor) -> torch.Tensor:
        key = (w.data_ptr(), tuple(w.shape))
        p = self.packed.get(key)
        if p is None:
            n, k = w.shape
            parts, r0 = [], 0
            for bn, count in self.tiles:
                for _ in range(count):
                    t = w[r0:r0 + bn].view(bn, k // 64, 8, 8).permute(1, 0, 2, 3)   # [kb][r][chunk][8]
                    r = torch.arange(bn, device=w.device)[:, None]
                    c = torch.arange(8, device=w.device)[None, :]
                    t = torch.gather(t, 2, (c ^ (r % 8))[None, :, :, None].expand(t.shape))
                    parts.append(t.reshape(-1))
                    r0 += bn
            assert r0 == n
            p = torch.cat(parts).contiguous()
            self.packed[key] = p
        return p

    def __call__(self, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor = None):
        m, k = x.shape
        n = w.shape[0]
        assert m <= self.bt and n == self.n_total and k % 64 == 0
        if out is None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        cu = self.cu
        dx = self._desc(x, self.bt)
        wp = self.pack(w)
        params = ((dx, wp.data_ptr(), out.data_ptr(), m, n, k), (None,) + (ctypes.c_void_p,) * 2 + (ctypes.c_int,) * 3)
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(cu.cuLaunchKernel(self.func, self.ctas, 1, 1, self.threads, 1, 1, self.smem, stream, params, 0))
        return out

