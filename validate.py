"""A pseudo-judge on a Modal H200: submission.py run the way okc runs it, on the scored shapes and
on in-between ones no sweep tuned, against SGLang's path, logged to CSV.

    conda activate muse_bf16
    modal run validate.py                          # tests, scored shapes, in-between shapes
    modal run validate.py --weights-seed 1234      # different weights, as held-out data would be
    modal run validate.py --nearby-ms 20,48,300    # pick the in-between m

What it does, after okc's rules (TASK.md):
- prepare(weights) once, timed against the 300 s limit.
- The 9 tests, each on several fresh inputs, checked with reference.check_implementation.
- Each shape captured as a CUDA graph and replayed: before every replay a fresh input is copied
  into the graph's input and a buffer larger than L2 is read; SGLang's replays and yours take
  turns on the same input, two at a time with the order swapped. The shape's speedup is the
  median of the paired ratios, with the graph replay's fixed cost (JUDGE_FIXED_US) taken off
  both sides, as the judge does. Every one of your timed outputs is checked, and the split-K
  counters must be back at 0 after each shape.
- The scored shapes give a load time each (the sum of its five shapes) and a speedup: the
  geometric mean over loads of SGLang's load time over yours.
- The in-between shapes (no sweep tuned at these m) are flagged where yours is under FLAG_BELOW
  of SGLang's speed.

SGLang isn't installed here; baseline_kernel is baseline.py's path as it runs on H200: its
bf16_gemm_dispatch falls through to F.linear (sglang unquant.py, _bf16_gemm_dispatch_impl), and
the router is F.linear in FP32 up to 8 tokens, two BF16 GEMMs on a hi/lo split above. Not part
of the submission: okc submits only the file you name.
"""

import csv
import math
import os
import time
from pathlib import Path

import modal

PROJ = Path(__file__).parent
# Pin to the judge's version once known (submission.py's prepare() prints it in a check run's log).
TORCH_VERSION = None

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"torch=={TORCH_VERSION}" if TORCH_VERSION else "torch", "numpy", "cuda-python")
    .env({"TRITON_DEFAULT_FP_FUSION": "1"})  # Triton's current default, set so it can't move
    .add_local_file(PROJ / "submission.py", "/root/proj/submission.py")
    .add_local_file(PROJ / "reference.py", "/root/proj/reference.py")
    .add_local_file(PROJ / "checks.py", "/root/proj/checks.py")
)
app = modal.App("muse-pseudo-judge", image=image)

LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
# The scored loads (TASK.md): name -> m. Each load runs all five layers at that m.
LOADS = {"decode 1": 1, "decode 16": 16, "decode 32": 32, "verify 1": 8, "verify 16": 128, "verify 32": 256}
# The 9 tests okc runs before timing (TASK.md): (m, layer, dist).
TESTS = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"),
         (130, "unpooler", "normal"), (5, "unpooler", "normal"), (8, "router", "normal"),
         (9, "router", "outliers"), (512, "qkv_gate", "outliers"), (1000, "unpooler", "outliers")]
# In-between m that no sweep tuned (router: 1,2,4,8,16,32,64,128,256; BF16: 1,8,16,32,128,256),
# including m just past a tile size, where a config tuned at the top of its row is weakest.
NEARBY_MS = "3,5,12,17,20,24,33,40,48,65,96,129,160,192,257,300,512"
COLUMNS = ["time", "gpu", "torch", "triton", "weights_seed", "phase", "load", "layer", "m", "n",
           "dist", "status", "detail", "checked", "sglang_us", "yours_us", "speedup_raw", "speedup"]
L2_FLUSH_BYTES = 256 << 20   # larger than the H200's 50-60 MB L2
JUDGE_FIXED_US = 4.0         # the judge takes about this (a replay's fixed cost) off both sides
PREPARE_LIMIT_S = 300
FLAG_BELOW = 0.95            # flag in-between shapes where yours is under this fraction of SGLang


@app.function(gpu="H200", timeout=2 * 3600)
def judge(weights_seed: int, reps: int, input_seeds: int, nearby_ms: list, run_nearby: bool):
    import statistics
    import sys

    sys.path.insert(0, "/root/proj")
    import torch
    import torch.nn.functional as F
    import triton

    import reference as R
    import submission as S

    env = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
           "weights_seed": weights_seed}

    def row(**fields):
        r = {c: "" for c in COLUMNS}
        r.update(env, time=time.strftime("%Y-%m-%d %H:%M:%S"), **fields)
        return r

    def baseline_prepare(weights):
        state = {name: w.contiguous() for name, w in weights.items()}
        hi = weights["router"].to(torch.bfloat16)
        lo = (weights["router"] - hi.float()).to(torch.bfloat16)
        state["router_hi_t"], state["router_lo_t"] = hi.t().contiguous(), lo.t().contiguous()
        return state

    def baseline_kernel(data, state):
        x, layer = data["x"], data["config"]["layer"]
        if layer != "router":
            return F.linear(x, state[layer][: data["config"]["n"]])
        if x.shape[0] <= 8:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    weights = R.generate_weights(weights_seed)
    start = time.time()
    state = S.prepare(weights)
    torch.cuda.synchronize()
    prep_s = time.time() - start
    yield row(phase="prepare", status="ok" if prep_s <= PREPARE_LIMIT_S else "fail",
              detail=f"prepare took {prep_s:.1f} s (limit {PREPARE_LIMIT_S} s)")
    base_state = baseline_prepare(weights)
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")

    def counters_clean():
        counts = state.get("split_counts")
        return counts is None or int(torch.count_nonzero(counts)) == 0

    def inputs(m, layer, dist, base_seed):
        return [R.generate_input(m, layer, seed=base_seed + i, dist=dist) for i in range(input_seeds)]

    def capture(fn, st, data):
        """A CUDA graph of fn(data, st), with data["x"] as its input buffer, and its output."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # first calls (and any compiles) off the capture stream
            for _ in range(2):
                fn(data, st)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(data, st)
        torch.cuda.synchronize()
        return graph, out

    def replay_us(graph, x_buf, x_new):
        x_buf.copy_(x_new)
        flush.sum()  # read, as the judge does: a written buffer would leave L2 dirty
        start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_ev.record()
        graph.replay()
        end_ev.record()
        torch.cuda.synchronize()
        return start_ev.elapsed_time(end_ev) * 1e3

    def time_shape(m, layer, dist, seed):
        """(sglang median us, yours median us, raw speedup, judge-style speedup, outputs checked,
        first wrong output's message or None)."""
        pool = inputs(m, layer, dist, seed)
        data = {"x": pool[0]["x"].clone(), "config": pool[0]["config"]}
        g_yours, out_yours = capture(S.kernel_fn, state, data)
        g_base, _ = capture(baseline_kernel, base_state, data)
        base_t, yours_t, ratios, ratios_adj, wrong, checked = [], [], [], [], None, 0
        for i in range(reps):
            fresh = pool[i % len(pool)]
            # Two at a time with the order swapped, so which side went first cancels.
            order = ("base", "yours") if i % 2 == 0 else ("yours", "base")
            t = {}
            for side in order:
                if side == "base":
                    t["base"] = replay_us(g_base, data["x"], fresh["x"])
                else:
                    t["yours"] = replay_us(g_yours, data["x"], fresh["x"])
                    ok, msg = R.check_implementation(fresh, out_yours, weights)  # every timed call
                    checked += 1
                    if not ok and wrong is None:
                        wrong = f"timed call {i}: {msg}"
            base_t.append(t["base"])
            yours_t.append(t["yours"])
            ratios.append(t["base"] / t["yours"])
            ratios_adj.append((t["base"] - JUDGE_FIXED_US) / max(t["yours"] - JUDGE_FIXED_US, 1e-3))
        del g_yours, g_base
        return (statistics.median(base_t), statistics.median(yours_t), statistics.median(ratios),
                statistics.median(ratios_adj), checked, wrong)

    def run_shape(phase, m, layer, dist="normal", load="", seed=0):
        try:
            # Plain calls first, on every input of the pool, like the judge's tests.
            for data in inputs(m, layer, dist, seed):
                ok, msg = R.check_implementation(data, S.kernel_fn(data, state), weights)
                if not ok:
                    return row(phase=phase, load=load, layer=layer, m=m, dist=dist, status="fail",
                               detail=f"plain call: {msg}")
            base_us, yours_us, raw, adj, checked, wrong = time_shape(m, layer, dist, seed)
        except Exception as e:
            return row(phase=phase, load=load, layer=layer, m=m, dist=dist, status="fail",
                       detail=f"{type(e).__name__}: {e}".replace("\n", " ")[:300])
        status, detail = "ok", ""
        if wrong:
            status, detail = "fail", wrong
        elif not counters_clean():
            status, detail = "fail", "split-K counters not all back at 0"
        n = R.out_features(layer, m)
        return row(phase=phase, load=load, layer=layer, m=m, n=n, dist=dist, status=status, detail=detail,
                   checked=checked, sglang_us=base_us, yours_us=yours_us, speedup_raw=raw, speedup=adj)

    for m, layer, dist in TESTS:
        yield run_shape("test", m, layer, dist, seed=7000 + m)
    for load, m in LOADS.items():
        for layer in LAYERS:
            yield run_shape("scored", m, layer, load=load, seed=8000 + m)
    if run_nearby:
        for m in nearby_ms:
            for layer in LAYERS:
                yield run_shape("nearby", m, layer, seed=9000 + m)


@app.local_entrypoint()
def main(weights_seed: int = 5151, reps: int = 60, input_seeds: int = 4, nearby_ms: str = NEARBY_MS,
         nearby: bool = True, out: str = ""):
    path = PROJ / (out or f"sweeps/validate_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(path.parent, exist_ok=True)
    rows = []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        ms = [int(v) for v in nearby_ms.split(",") if v]
        for r in judge.remote_gen(weights_seed, reps, input_seeds, ms, nearby):
            writer.writerow(r)
            f.flush()
            rows.append(r)
            if r["phase"] == "prepare":
                print(f"prepare: {r['status']} ({r['detail']})")
                continue
            tag = f"{r['phase']:<6} {r['layer']:>8} m={r['m']:<5} {r['dist']:<8}"
            if r["status"] != "ok":
                print(f"{tag} FAIL  {r['detail'][:120]}")
            else:
                print(f"{tag} SGLang {r['sglang_us']:7.2f} us  yours {r['yours_us']:7.2f} us  "
                      f"{r['speedup']:5.2f}x  ({r['checked']} timed outputs checked)")

    print(f"\nwrote {path}")
    fails = [r for r in rows if r["status"] != "ok"]
    print(f"\ntests: {sum(r['phase'] == 'test' and r['status'] == 'ok' for r in rows)}"
          f"/{sum(r['phase'] == 'test' for r in rows)} passed")
    scored = [r for r in rows if r["phase"] == "scored" and r["status"] == "ok"]
    if len(scored) == len(LOADS) * len(LAYERS):
        logs = []
        print("\nscored loads (judge-style: SGLang's load time over yours):")
        for load in LOADS:
            shapes = [r for r in scored if r["load"] == load]
            base = sum(r["sglang_us"] - JUDGE_FIXED_US for r in shapes)
            yours = sum((r["sglang_us"] - JUDGE_FIXED_US) / r["speedup"] for r in shapes)
            logs.append(math.log(base / yours))
            print(f"  {load:<10} {base / yours:.3f}x   " +
                  "  ".join(f"{r['layer']} {r['speedup']:.2f}x" for r in shapes))
        print(f"\n  speedup (geometric mean over loads): {math.exp(sum(logs) / len(logs)):.3f}x")
    else:
        print("\nscored loads: not all 30 shapes passed, so no speedup")
    nearby_rows = [r for r in rows if r["phase"] == "nearby" and r["status"] == "ok"]
    slow = [r for r in nearby_rows if r["speedup"] < FLAG_BELOW]
    print(f"\nin-between shapes: {len(nearby_rows)} ran, {len(slow)} under {FLAG_BELOW}x SGLang:")
    for r in sorted(slow, key=lambda r: r["speedup"]):
        print(f"  {r['layer']:>8} m={r['m']:<5} {r['speedup']:.2f}x")
    if fails:
        print(f"\n{len(fails)} FAILED:")
        for r in fails:
            print(f"  {r['phase']} {r['layer']} m={r['m']} {r['dist']}: {r['detail'][:160]}")
