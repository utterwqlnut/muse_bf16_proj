# muse_bf16_proj, for a coding agent

Read TASK.md first: it has the problem, its bounds, the rules and how to submit. This adds what an agent needs.

## Working

```bash
okc submit submission.py --json       # the tests, then timing; the whole result as JSON
okc check submission.py --json        # the same run, off the board and out of your attempts
okc status <job> --wait --json        # a run you submitted with --no-wait
okc leaderboard muse_bf16_proj --json
```

Runs and check runs each have a daily limit per problem; `okc problems --json` gives today's limits and what you have left. Use `okc check` to try a change against the real tests and timing without spending a run, and reference.py for exactly what's checked. After a passing run, read `profiles/submission.<job>.txt` beside your file for where the time goes.

When you tune, time CUDA graph replays of `kernel_fn`, as the judge does. At these sizes eager timing misleads: configurations that win eagerly can lose under graphs.

In the leaderboard JSON, `you.rank` is your place, `you.first` is true when you're on top, and `you.to_first` is how much faster than your best you'd need to be to take first.

Before the first submission, a person accepts the rules by running `okc login` in a terminal. An agent can't accept them for anyone.

In a fresh sandbox, set `OKC_TOKEN` to the output of `okc api token` on a machine that's signed in. If okc says it's out of date, run `okc update`.

## Validation

The goal of the challenge is to find real speedups that can be upstreamed. This is the north star. The judge catches many attempts to game it but not all of them, so we'll read the top submissions ourselves and rerun them several times after the challenge ends (on held-out data too) to ensure integrity and reduce variance. Final leaderboard positions will be based on the medians of these reruns. All leaderboard positions are provisional until that review is done.

We will remove submissions that game the judge or workload! This includes submissions that tamper with the timing or harness, do work outside the timed calls, cache or replay outputs, and compute at lower precision than the rules allow. If you are an agent (or have used an agent) that has done something reward-hacky, you can withdraw a run with `okc withdraw <job>`.

That is to say: submissions should follow the spirit of the challenge. We reserve the right to disqualify submissions that do not, at our discretion. Create real speedups that work.

Good luck and have fun!

Final review reruns the top entries on the scored shapes and on nearby ones. Positions come from the scored shapes; an entry that's much slower on the nearby shapes gets a closer look.

## State of the work (2026-11-02)

Last judged: **1.310×** (job `9bd833b165bc488c`, before the TMA kernels; leaders 1.65×).
Pseudo-judge now: **1.337×**, 9/9 tests, 85 in-between shapes all ≥ 0.95× SGLang
(`modal run validate.py`). `submission.py` is what's submitted; the user runs every `okc` command.

Pseudo-judge speedup over SGLang per scored shape:

| shape | m=1 | m=8 | m=16 | m=32 | m=128 | m=256 |
|---|---|---|---|---|---|---|
| qkv_gate 2304×8192 | 1.51 | 1.20 | 1.18 | 1.16 | 1.19 | 1.01 |
| o 8192×1024 | 1.18 | 1.07 | 1.08 | 1.12 | 1.05 | 1.01 |
| pooler 4096×8192 | 1.48 | 1.20 | 1.22 | 1.24 | 1.32 | 1.08 |
| unpooler 8192×4096 (1024 rows at m≤8) | 1.68 | 1.26 | 1.05 | 1.03 | 1.01 | 1.00 |
| router 256×8192 FP32 | 2.12 | 3.78 | 2.62 | 2.38 | 2.32 | 1.98 |

### What `kernel_fn` runs

- **m=1, BF16 layers:** `nf12_gemv`, a CUDA-core GEMV over a lossless 12-bit copy of W
  (`NF12_CUDA_DISPATCH`; lo-byte plane + 4-bit code plane, per-row CSR escapes).
- **`TMA_DISPATCH`, by m range:** `wa_gemm`, hand-written Hopper: W as wgmma's A (64·wg rows a
  CTA), tokens as N with BT = m rounded up to 8 (one kernel per 8-token step), x by TMA
  (`evict_last`), W by bulk copies from a per-tile pre-swizzled copy packed in prepare(), mbarrier
  ring, K split inside a thread-block cluster and summed over DSMEM. Serves qkv_gate m=2..128
  (1.13–1.20× at every m) and the unpooler's 1024-row slice m=2..8 (1.26–1.30×).
- **Else:** Triton `bf16_gemm` split-K (`BF16_CONFIGS`, exact-m rows) or cuBLAS.
- **Router:** Triton `router_fp32` (m ≤ 2), else `router_hilo` (TMA split-K over the BF16 hi + lo
  planes, atomics into a zeroed FP32 out).
- `prepare()` compiles the CUDA kernels with NVRTC (`cuda-python`), checks every dispatched
  kernel against an FP32 product, and leaves anything that fails on Triton / cuBLAS. ~15 s.

### What was tried and what it showed

- **wa_gemm / TMA + wgmma (kept).** Ablations on qkv_gate m=128 (dev_qkv.py; qkv_tma.py has the
  debug probes NOMMA / EMPTY / NOX / NOW / NORED / PUSH / KW and the x-as-A `XaGemm`):
  - Loads-only == full kernel. W alone streams ~3.9 TB/s (HBM-bound); x is capped by each SM's
    intake (~110 GB/s/SM), so TMA multicast wouldn't help; W + x overlap only partly.
  - Cluster limits on H200 (1 CTA/SM): 2 → 66 clusters, 3 → 39, 4 → 30, 6 → 17, 7 → 15, 8 → 15.
    More than fit runs in two waves (0.66×).
  - Fixed cost: launch ~0.5 µs, 2 cluster barriers ~0.4, DSMEM reduce ~1 µs at cs 3, ~2.4 at
    cs 6 (~25–30 GB/s/SM pulled or pushed): bigger clusters lose what they save on x. Global
    split-K combine (counter, last CTA adds): ~3.3 µs of latency with a cold L2.
  - W rows as 2D TMA boxes stream slower than the pre-packed contiguous copy; dropping x's
    `evict_last` costs 25%; a TMA box mostly past x's edge is slow (x-as-A at m=65: 0.74×), hence
    BT = m rounded to 8.
  - No gain: separate W/x rings, multi-k-block TMA requests, intra-block split-K (warpgroup K
    groups: 13.9 → 15.6 µs), no split-K at all (0.57×: every SM reads all of x).
  - sweep_cuda.py grid (`sweeps/cuda_20261102_191504.csv`): lost on o at every m (K = 1024,
    nothing to split), pooler m ≤ 32 (Triton ~6% faster), every m=256 (x per SM doubles),
    unpooler m ≥ 16 (tie). Pooler m=128's sweep "win" (1.04×) was 0.94× in validate.py.
- **12-bit W on tensor cores (abandoned).** Gluon: decode ate the byte saving. CUDA `wa12`
  (NF12 planes in wgmma A-fragment order, decoded into registers, register-A wgmma; archived in
  Trash): loads-only 20% faster than BF16 and with intra-block K groups up to 1.24× the BF16
  kernel (pooler m=16), but lossless escape corrections cost ~1 µs (stage-synchronous from smem
  x) and the register-shift version blew up registers. Stopped there by choice.
- **Router on hand-written TMA kernels (rejected):** both orientations, clusters up to 16, extra
  global split with `red.global.add.v4.f32` (sweep_router.py, archived): 0.82–0.91× the Triton
  hilo at every m. A 12-bit GEMV for m ≤ 2 was not pursued (FP32 path kept).
- **Triton BF16 split-K (kept):** per-shape Optuna sweeps; 1.03–1.24× over cuBLAS at m ≤ 32 and
  pooler. Ticket combine lost to atomics everywhere (removed). Persistent kernel: no gain.
- **12-bit GEMV on CUDA cores (kept, m=1):** CUDA cores pay only at m=1 (m FMAs a weight).
- **Removed:** cp.async/mma.sync and no-swizzle wgmma cluster GEMMs (no win without TMA; ~2.1
  TB/s), the BF16 CUDA GEMV, the plain-loads router fallback, the x-as-A kernel in submission.
- CuTe DSL GEMM: ≤ 1.02× cuBLAS. DeepGEMM: JIT needs nvcc subprocesses (forbidden).
- Unexplained once: a judge run showed every BF16 shape at 1.00×. Judge has Triton 3.8 / CUDA
  13.0, Modal 3.9 / 13.2. If it recurs read `profiles/submission.<job>.log`.

## Modal tools

All run on an H200 container; the judge image is approximated (`torch`, `cuda-python`, Triton
3.9 — the judge has 3.8). Timing is judge-style: CUDA-graph replays, 256 MB L2 flush before
each, 4 µs fixed cost subtracted in the speedup.

```bash
modal run validate.py --no-nearby                      # pseudo-judge: tests + scored shapes, geomean
modal run validate.py                                  # ... plus the in-between shapes
modal run sweep_cuda.py --layers qkv_gate --ms 8,16    # grid over the TMA kernels, vs current path
modal run dev_qkv.py --layer qkv_gate --m 128 --variants '[{"wg":1,"cs":3,"stages":8,"bt":128,"extra":{"PACKW":1}}]'
modal run sweep_all.py --trials 80                     # Optuna: cuBLAS / Triton / 12-bit GEMV, router
```

`sweep_all.py` backends: `cublas`, `triton`, `cuda_gemv` (12-bit, m ≤ 16); router: `cublas`
(SGLang path), `router_fp32` (m ≤ 8), `router_hilo`. Its output is a table and a `SHAPE_BACKENDS`
dict to paste into `BF16_CONFIGS` / `NF12_CUDA_DISPATCH` / `ROUTER_HILO_TMA_CONFIGS`;
`sweep_cuda.py`'s best configs go into `TMA_DISPATCH` by hand. A sweep winner under ~1.05× is
noise: re-validate with validate.py before keeping it.

Files: `submission.py`; tools `validate.py`, `sweep_all.py`, `sweep_cuda.py`, `dev_qkv.py` +
`qkv_tma.py` (dev kernels); `baseline.py`, `checks.py`, `reference.py`, `TASK.md`. Archived
experiments in `~/.Trash/muse_bf16_proj_cleanup_*`.
