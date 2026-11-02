"""One sweep over every backend for every BF16 shape, on Modal H200s: per shape, the backend
first (cuBLAS, the Triton split-K kernel, the CUDA 12-bit GEMV),
then that backend's own parameters, searched together by Optuna (TPE). Times like the judge:
CUDA graph replays, each after reading a buffer larger than L2. Not part of the submission.

    conda activate muse_bf16
    modal run sweep_all.py                                  # 4 layers x 6 scored m, 60 trials each
    modal run sweep_all.py --layers pooler --ms 1,16        # a subset
    modal run sweep_all.py --trials 100                     # a longer search per shape

Backends and their parameters (submission.py's names). The router has its own: SGLang's path
(the reference, "cublas" in the CSV: FP32 F.linear up to 8 tokens, two BF16 mm above),
router_fp32 (m <= 8: BLOCK_N, BLOCK_K, num_warps, num_stages) and router_hilo (the hi + lo TMA
kernel: BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages). The BF16 layers':
  cublas      F.linear, nothing to tune
  triton      bf16_gemm: BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
  cuda_gemv   nf12_cuda_gemv (m <= 16): wpr (warps a row), warps (a block), rpw (rows a warp),
              unroll, ldhint
Each search starts from cuBLAS, the current Triton row (BF16_CONFIGS) and the CUDA defaults;
combinations that can't run (shared memory, empty splits, rows not whole blocks...) are pruned
untimed. Every candidate is checked (reference.check_implementation) before it's timed and its
graph's output again after; the best three are re-timed with 3x the replays.

Out: sweeps/all_<time>.csv (every trial), a table per shape (best backend and parameters,
judge-style speedups over cuBLAS and over the current choice, 4 us of fixed cost taken out),
and SHAPE_BACKENDS, the per-shape choices to put into submission.py.
"""

import csv
import json
import time
from pathlib import Path

import modal

PROJ = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "optuna", "cuda-python")
    .env({"TRITON_DEFAULT_FP_FUSION": "1"})
    .add_local_file(PROJ / "submission.py", "/root/proj/submission.py")
    .add_local_file(PROJ / "reference.py", "/root/proj/reference.py")
    .add_local_file(PROJ / "checks.py", "/root/proj/checks.py")
)
app = modal.App("muse-sweep-all", image=image)

L2_FLUSH_BYTES = 256 << 20
JUDGE_FIXED_US = 4.0
SMEM_LIMIT = 227 * 1024
LAYERS = "qkv_gate,o,pooler,unpooler,router"
MS = "1,8,16,32,128,256"
FIELDS = ["layer", "m", "n", "k", "trial", "backend", "params", "us", "us_retimed", "cublas_us", "current_us",
          "x_vs_cublas", "x_vs_current", "status"]


@app.function(gpu="H200", timeout=2 * 3600)
def sweep_shape(layer: str, m: int, trials: int, reps: int, seed: int):
    import statistics
    import sys
    import warnings

    import optuna

    sys.path.insert(0, "/root/proj")
    import torch
    import torch.nn.functional as F
    import triton

    import reference as R
    import submission as S

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    tag = f"[{layer} m={m}]"
    weights = R.generate_weights(R.WEIGHTS_SEED)
    device = "cuda"
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    state = {layer: weights[layer].contiguous(), "bf16_descs": {}, "nf12": {}, "nf12_kernels": {},
             "split_counts": torch.zeros(S.SPLIT_COUNTERS, dtype=torch.int32, device=device)}
    packed = None
    try:
        if layer != "router":  # the router's weights are FP32: no 12-bit copy
            packed = S.nf12_pack_cuda(state[layer])
    except Exception as e:  # noqa: BLE001
        print(f"{tag} 12-bit pack failed: {type(e).__name__}: {e}", flush=True)
    if packed is not None:
        state["nf12"][layer] = packed

    data = R.generate_input(m, layer, seed=31 + m)
    x, n = data["x"], data["config"]["n"]
    k = x.shape[1]
    w = state[layer][:n]

    def graph_us(fn, n_reps):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n_reps):
            flush.sum()  # read, as the judge does: every replay starts with a cold L2
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) * 1e3)
        return statistics.median(times), out

    judge = lambda base, us: (base - JUDGE_FIXED_US) / max(us - JUDGE_FIXED_US, 1e-3)

    if layer != "router":
        # --- the backends: what each can run here, and how to run a config -------------------------
        backends = ["cublas", "triton"]
        if packed is not None and m <= S.NF12_GEMV_MAX_TOKENS:
            backends.append("cuda_gemv")
        triton_space = {
            "BLOCK_M": [b for b in (16, 32, 64, 128, 256) if b <= max(16, triton.next_power_of_2(m))],
            "BLOCK_N": [64, 128, 256], "BLOCK_K": [64, 128, 256],
            "SPLIT_K": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16], "num_warps": [4, 8], "num_stages": [2, 3, 4, 5, 6]}
        gemv_space = {"wpr": [1, 2, 4, 8], "warps": [2, 4, 8, 16], "unroll": [1, 2, 4, 8], "ldhint": [0, 1, 2],
                      "rpw": [1, 2, 4, 8], "minb": [1, 2, 3], "xldg": [0, 1]}
        spaces = {"cublas": {}, "triton": triton_space, "cuda_gemv": gemv_space}

        def why_not(backend, p):
            """Why this config can't run, or None."""
            if backend == "triton":
                bm, bn, bk, sk, nw, ns = (p[c] for c in triton_space)
                if ns * (bn * bk + bm * bk) * 2 > SMEM_LIMIT:
                    return "shared memory"
                if triton.cdiv(k, sk) < bk or (sk - 1) * triton.cdiv(triton.cdiv(k, sk), bk) * bk >= k:
                    return "an empty split"
                if bn * bm / (32 * nw) > 128:
                    return "accumulator registers"
                if triton.cdiv(m, bm) * triton.cdiv(n, bn) > S.SPLIT_COUNTERS:
                    return "split-K counters"
            elif backend == "cuda_gemv":
                if p["wpr"] not in S._nf12_gemv_wprs(n, k, p["warps"], p["rpw"]):
                    return "rows not whole blocks, or K not whole chunks"
                if p["rpw"] * (m + 16) > 160:
                    return "registers: RPW rows x (M accumulators + 16 widened weights)"
                if p["minb"] * p["warps"] * 32 > 2048:
                    return "MINB blocks of WARPS warps won't fit an SM (2048 threads)"
            return None

        def runner(backend, p):
            if backend == "cublas":
                return lambda: F.linear(x, w)
            if backend == "triton":
                cfg = tuple(p[c] for c in triton_space)
                return lambda: S.bf16_gemm(x, w, layer, state, cfg)
            kern = S._nf12_cuda_kernel(state, "gemv", m, **p)  # cuda_gemv
            if kern is None:
                raise RuntimeError("didn't compile")
            return lambda: S.nf12_cuda_gemv(x, layer, n, state, **p)

        # --- the references: cuBLAS, and what the submission runs without its CUDA dispatch ------
        cublas_us, _ = graph_us(lambda: F.linear(x, w), reps)
        cur_cfg = S._bf16_config(layer, m)
        current_us = graph_us(runner("triton", dict(zip(triton_space, cur_cfg))), reps)[0] if cur_cfg else cublas_us
        print(f"{tag} n={n} k={k}: cuBLAS {cublas_us:.2f} us, current ({cur_cfg or 'cuBLAS'}) {current_us:.2f} us; "
              f"backends {backends}; {trials} trials", flush=True)
        base = dict(layer=layer, m=m, n=n, k=k, cublas_us=round(cublas_us, 3), current_us=round(current_us, 3))

    else:
        # The router: FP32 W [256, 8192]. SGLang's path is the reference ("cublas" in the CSV):
        # F.linear in FP32 up to 8 tokens, two BF16 mm (hi + lo, FP32 out) above.
        W = state[layer]
        hi = W.to(torch.bfloat16)
        lo = (W - hi.float()).to(torch.bfloat16)
        state.update(router=W, router_hi=hi.contiguous(), router_lo=lo.contiguous(), router_descs={})
        hi_t, lo_t = hi.t().contiguous(), lo.t().contiguous()

        def sglang():
            if m <= 8:
                return F.linear(x.float(), W)
            acc = torch.mm(x, hi_t, out_dtype=torch.float32)
            acc += torch.mm(x, lo_t, out_dtype=torch.float32)
            return acc

        backends = ["cublas", "router_hilo"] + (["router_fp32"] if m <= 8 else [])
        fp32_space = {"BLOCK_N": [1, 2, 4, 8], "BLOCK_K": [512, 1024, 2048, 4096, 8192],
                      "num_warps": [2, 4, 8, 16], "num_stages": [1, 2, 3]}
        hilo_space = {"BLOCK_M": [b for b in (16, 32, 64, 128, 256) if b <= max(16, triton.next_power_of_2(m))],
                      "BLOCK_N": [64, 128, 256], "BLOCK_K": [64, 128, 256],
                      "SPLIT_K": [1, 2, 4, 8, 16, 32, 64], "num_warps": [4, 8], "num_stages": [2, 3, 4, 5, 6],
                      "combine": ["atomic", "ticket"]}
        spaces = {"cublas": {}, "router_fp32": fp32_space, "router_hilo": hilo_space}
        tile_keys = list(hilo_space)[:6]

        def hilo_params(cfg):
            """A ROUTER_HILO_TMA_CONFIGS row (6 or 7 fields) as router_hilo parameters."""
            return {**dict(zip(tile_keys, cfg[:6])), "combine": cfg[6] if len(cfg) > 6 else "atomic"}

        def why_not(backend, p):
            if backend == "router_fp32":
                if p["BLOCK_N"] * p["BLOCK_K"] / (32 * p["num_warps"]) > 64:
                    return "registers: a W tile and the next one's"
            elif backend == "router_hilo":
                bm, bn, bk, sk, nw, ns = (p[c] for c in tile_keys)
                if ns * (2 * bn * bk + bm * bk) * 2 > SMEM_LIMIT:
                    return "shared memory"
                if triton.cdiv(k, sk) < bk or (sk - 1) * triton.cdiv(triton.cdiv(k, sk), bk) * bk >= k:
                    return "an empty split"
                if bn * bm / (32 * nw) > 128:
                    return "accumulator registers"
            return None

        def runner(backend, p):
            if backend == "cublas":
                return sglang
            if backend == "router_fp32":
                def run():
                    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
                    S._router_fp32_kernel[(triton.cdiv(n, p["BLOCK_N"]),)](
                        x, W, out, m, n, k, x.stride(0), x.stride(1), W.stride(0), W.stride(1),
                        out.stride(0), out.stride(1),
                        BLOCK_M=max(2, triton.next_power_of_2(m)), BLOCK_N=p["BLOCK_N"], BLOCK_K=p["BLOCK_K"],
                        num_warps=p["num_warps"], num_stages=p["num_stages"])
                    return out
                return run
            bm, bn, bk, sk, nw, ns = (p[c] for c in tile_keys)  # router_hilo, as router_hilo_tma
            ticket = p["combine"] == "ticket"
            descs = state["router_descs"]
            if (bn, bk) not in descs:
                descs[(bn, bk)] = (S.TensorDescriptor.from_tensor(state["router_hi"], [bn, bk]),
                                   S.TensorDescriptor.from_tensor(state["router_lo"], [bn, bk]))
            hi_desc, lo_desc = descs[(bn, bk)]
            k_per_split = triton.cdiv(triton.cdiv(k, sk), bk) * bk

            def run():
                x_desc = S.TensorDescriptor.from_tensor(x, [bm, bk])
                if ticket:  # partials + a ticket: no zeroing
                    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
                    part = torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
                else:       # a zeroed output + atomics
                    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
                    part = out
                S._router_hilo_tma_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn), sk)](
                    x_desc, hi_desc, lo_desc, out, part, state["split_counts"], m, n, k_per_split,
                    out.stride(0), out.stride(1),
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, TICKET=ticket, num_warps=nw, num_stages=ns)
                return out
            return run

        # The references: SGLang's path, and the submission's own router choice.
        cublas_us, _ = graph_us(sglang, reps)
        if m <= S.ROUTER_FP32_MAX_TOKENS:
            cur = ("router_fp32", dict(BLOCK_N=S.ROUTER_BLOCK_N, BLOCK_K=S.ROUTER_BLOCK_K,
                                       num_warps=S.ROUTER_NUM_WARPS, num_stages=S.ROUTER_NUM_STAGES))
        else:
            cur = ("router_hilo", hilo_params(S._router_hilo_config(m, S.ROUTER_HILO_TMA_CONFIGS)))
        current_us = graph_us(lambda: S.router_fp32(x, W) if m <= S.ROUTER_FP32_MAX_TOKENS else S.router_hilo(x, state),
                              reps)[0]
        print(f"{tag} n={n} k={k}: SGLang {cublas_us:.2f} us, current ({cur[0]} {cur[1]}) {current_us:.2f} us; "
              f"backends {backends}; {trials} trials", flush=True)
        base = dict(layer=layer, m=m, n=n, k=k, cublas_us=round(cublas_us, 3), current_us=round(current_us, 3))

    rows, seen, broken = [], {}, [False]

    def evaluate(backend, p, trial_no):
        key = (backend, json.dumps(p, sort_keys=True))
        if key in seen:
            return seen[key]
        row = dict(base, trial=trial_no, backend=backend, params=json.dumps(p, sort_keys=True))
        us = float("inf")
        try:
            fn = runner(backend, p)
            out = fn()
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, out, weights)
            if ok:
                us, out = graph_us(fn, reps)
                ok, msg = R.check_implementation(data, out, weights)  # the graph's own output
            if not ok:
                us = float("inf")
                row.update(status=f"WRONG {msg}"[:200])
                print(f"{tag} {backend} {p}: WRONG {msg}", flush=True)
            else:
                row.update(us=round(us, 3), x_vs_cublas=round(judge(cublas_us, us), 4),
                           x_vs_current=round(judge(current_us, us), 4), status="ok")
                print(f"{tag} {backend:<9} {p}: {us:7.2f} us  {judge(cublas_us, us):.3f}x cuBLAS  "
                      f"{judge(current_us, us):.3f}x current", flush=True)
        except Exception as e:  # noqa: BLE001
            text = " ".join(str(e).split())
            row.update(status=f"FAILED {type(e).__name__}: {text[-200:]}")
            print(f"{tag} {backend} {p}: FAILED {type(e).__name__}: ...{text[-300:]}", flush=True)
            broken[0] = "launch failure" in text or "illegal" in text
        seen[key] = us
        rows.append(row)
        return us

    def objective(trial):
        backend = trial.suggest_categorical("backend", backends)
        space = spaces[backend]
        p = {name: trial.suggest_categorical(f"{backend}.{name}", choices) for name, choices in space.items()}
        reason = why_not(backend, p)
        if reason:
            raise optuna.TrialPruned(reason)  # untimed; TPE learns to avoid it
        return evaluate(backend, p, len(seen))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True,
                                         n_startup_trials=max(8, trials // 4))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    starts = [{"backend": "cublas"}]
    if layer == "router":
        starts.append({"backend": cur[0], **{f"{cur[0]}.{c}": v for c, v in cur[1].items()}})
        other = "router_hilo" if cur[0] == "router_fp32" else ("router_fp32" if m <= 8 else None)
        if other == "router_hilo":
            starts.append({"backend": other, **{f"router_hilo.{c}": v for c, v in
                                                hilo_params(S._router_hilo_config(m, S.ROUTER_HILO_TMA_CONFIGS)).items()}})
        for st in [st for st in starts if st.get("backend") == "router_hilo"]:  # and with the other combine
            flip = "ticket" if st.get("router_hilo.combine", "atomic") == "atomic" else "atomic"
            starts.append({**st, "router_hilo.combine": flip})
        if other == "router_fp32":
            starts.append({"backend": other, "router_fp32.BLOCK_N": S.ROUTER_BLOCK_N,
                           "router_fp32.BLOCK_K": S.ROUTER_BLOCK_K, "router_fp32.num_warps": S.ROUTER_NUM_WARPS,
                           "router_fp32.num_stages": S.ROUTER_NUM_STAGES})
    elif cur_cfg:
        starts.append({"backend": "triton", **{f"triton.{c}": v for c, v in zip(triton_space, cur_cfg)}})
    if "cuda_gemv" in backends:
        for wpr in S._nf12_gemv_wprs(n, k, 4):
            starts.append({"backend": "cuda_gemv", "cuda_gemv.wpr": wpr,
                           **{f"cuda_gemv.{c}": v for c, v in S.NF12_GEMV_DEFAULTS.items()}})
        if m > 1:  # and with x shared by 4 rows a warp, what makes more tokens pay
            for wpr in S._nf12_gemv_wprs(n, k, 4, 4)[:2]:
                starts.append({"backend": "cuda_gemv", "cuda_gemv.wpr": wpr, "cuda_gemv.warps": 4,
                               "cuda_gemv.unroll": 2, "cuda_gemv.ldhint": 2, "cuda_gemv.rpw": 4,
                               "cuda_gemv.minb": 1, "cuda_gemv.xldg": 0})
    for start in starts:
        study.enqueue_trial(start)
    for _ in range(trials * 50):  # pruned suggestions don't count as trials
        if len(seen) >= trials or broken[0]:
            break
        study.optimize(objective, n_trials=1, catch=())
    if broken[0]:
        print(f"{tag} CUDA context broken: stopped after {len(seen)} trials", flush=True)
        return rows

    # The best three again with 3x the replays, so the pick isn't a lucky draw.
    ok_rows = sorted((r for r in rows if r.get("status") == "ok"), key=lambda r: r["us"])[:3]
    for r in ok_rows:
        try:
            us, _ = graph_us(runner(r["backend"], json.loads(r["params"])), 3 * reps)
            r.update(us_retimed=round(us, 3), x_vs_cublas=round(judge(cublas_us, us), 4),
                     x_vs_current=round(judge(current_us, us), 4))
        except Exception as e:  # noqa: BLE001
            print(f"{tag} re-time {r['backend']} {r['params']}: {type(e).__name__}: {e}", flush=True)
    if ok_rows:
        b = min(ok_rows, key=lambda r: r.get("us_retimed") or r["us"])
        print(f"{tag} BEST {b['backend']} {b['params']}: {b.get('us_retimed') or b['us']:.2f} us = "
              f"{b['x_vs_cublas']:.3f}x cuBLAS, {b['x_vs_current']:.3f}x current ({len(seen)} trials)", flush=True)
    return rows


@app.local_entrypoint()
def main(layers: str = LAYERS, ms: str = MS, trials: int = 60, reps: int = 60, seed: int = 0):
    shapes = [(l, int(m)) for l in layers.split(",") if l for m in ms.split(",") if m]
    out_dir = PROJ / "sweeps"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"all_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for shape_rows in sweep_shape.starmap(shapes, kwargs={"trials": trials, "reps": reps, "seed": seed},
                                              order_outputs=False, return_exceptions=True):
            if isinstance(shape_rows, Exception):
                print(f"a shape's container failed: {shape_rows}")
                continue
            for r in shape_rows:
                writer.writerow({c: r.get(c, "") for c in FIELDS})
            f.flush()  # each shape saved as it finishes
            rows += shape_rows
    print(f"\nwrote {path} ({len(rows)} rows)")

    print(f"\n{'shape':<16}{'ref':>8}{'current':>9}{'best':>8}{'vs ref':>8}{'vs cur':>8}  backend and parameters"
          f"   (ref: cuBLAS; the router's: SGLang's path)")
    choices = {}
    for layer, m in shapes:
        ok = [r for r in rows if r["layer"] == layer and r["m"] == m and r.get("status") == "ok"]
        if not ok:
            print(f"{layer + ' m=' + str(m):<16}  nothing passed")
            continue
        b = min(ok, key=lambda r: r.get("us_retimed") or r["us"])
        us = b.get("us_retimed") or b["us"]
        print(f"{layer + ' m=' + str(m):<16}{b['cublas_us']:>8.2f}{b['current_us']:>9.2f}{us:>8.2f}"
              f"{b['x_vs_cublas']:>8.3f}{b['x_vs_current']:>8.3f}  {b['backend']} {b['params']}")
        choices[(layer, m)] = (b["backend"], json.loads(b["params"]), b["x_vs_cublas"])
    print("\nSHAPE_BACKENDS = {  # (layer, m): (backend, parameters); from sweep_all.py")
    for (layer, m), (backend, p, x) in sorted(choices.items()):
        print(f'    ("{layer}", {m}): ("{backend}", {p}),  # {x:.2f}x cuBLAS')
    print("}")
