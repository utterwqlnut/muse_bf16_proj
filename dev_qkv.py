"""Bench hand-written CUDA kernels (qkv_tma.py) on one shape on a Modal H200, judge-style: CUDA
graph replays, each after reading a buffer larger than L2, against cuBLAS (F.linear).

    modal run dev_qkv.py --variants '[{"wg":2,"cs":7,"stages":6}]'
    modal run dev_qkv.py --layer unpooler --m 128 --variants '...'
"""

import json
from pathlib import Path

import modal

PROJ = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "cuda-python")
    .add_local_file(PROJ / "qkv_tma.py", "/root/proj/qkv_tma.py")
    .add_local_file(PROJ / "reference.py", "/root/proj/reference.py")
    .add_local_file(PROJ / "checks.py", "/root/proj/checks.py")
)
app = modal.App("muse-dev-qkv", image=image)
JUDGE_FIXED_US = 4.0


@app.function(gpu="H200", timeout=1800)
def bench(layer: str, m: int, variants: list, reps: int):
    import statistics
    import sys
    import traceback

    sys.path.insert(0, "/root/proj")
    import torch
    import torch.nn.functional as F

    import qkv_tma as Q
    import reference as R

    weights = R.generate_weights(R.WEIGHTS_SEED)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    datas = [R.generate_input(m, layer, seed=31 + m + i) for i in range(3)]
    n = datas[0]["config"]["n"]
    w = weights[layer][:n].contiguous()
    x = datas[0]["x"].clone()
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__} SMs {torch.cuda.get_device_properties(0).multi_processor_count}")

    def graph_us(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            flush.sum()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        return statistics.median(ts), out

    tiny = torch.zeros(1, device="cuda")
    null, _ = graph_us(lambda: tiny.add_(1))
    print(f"null graph (one tiny kernel): {null:.2f} us", flush=True)
    base, _ = graph_us(lambda: F.linear(x, w))
    print(f"cuBLAS: {base:.2f} us (kernel ~{base - JUDGE_FIXED_US:.2f})", flush=True)
    for v in variants:
        v = dict(v)
        tag = json.dumps(v, separators=(",", ":"))
        try:
            kind = v.pop("kind", "tma")
            kern = Q.XaGemm(**v) if kind == "xa" else Q.TmaGemm(**v)
            try:
                mc = kern.max_clusters(n // (64 * kern.wg)) if kind != "xa" else kern.max_clusters()
            except Exception as e:  # noqa: BLE001
                mc = f"? ({e})"
            ok_all, msgs = True, []
            for d in datas:
                out = kern(d["x"], w)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, out, weights)
                ok_all &= ok
                msgs.append(msg)
            us, out = graph_us(lambda: kern(x, w))
            ok2, _ = R.check_implementation({"x": x, "config": datas[0]["config"]}, out, weights)
            grid = kern.ctas if kind == "xa" else (n // (64 * kern.wg)) * kern.cs * kern.g
            print(f"{tag}: {us:.2f} us  (-null {us - null:.2f} vs {base - null:.2f})  {(base - JUDGE_FIXED_US) / (us - JUDGE_FIXED_US):.3f}x cuBLAS  "
                  f"ok={ok_all and ok2} grid={grid} maxclusters={mc} regs={kern.regs} spill={kern.spill} smem={kern.smem}"
                  + ("" if ok_all else f"  {msgs[0]}"), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{tag}: FAILED {type(e).__name__}: {str(e)[:2500]}", flush=True)
            traceback.print_exc()


@app.local_entrypoint()
def main(layer: str = "qkv_gate", m: int = 128, variants: str = '[{"wg":2,"cs":7,"stages":6}]', reps: int = 200):
    bench.remote(layer, m, json.loads(variants), reps)
