"""Grid sweep of the hand-written TMA + wgmma kernels (qkv_tma.py) on every BF16 shape, against
cuBLAS and the current submission path (submission.kernel_fn: the swept Triton / 12-bit GEMV /
xa_gemm choices), timed judge-style on Modal H200s: CUDA-graph replays, each after reading a
buffer larger than L2. One container per (layer, m).

    modal run sweep_cuda.py                                  # all BF16 layers, m 8,16,32,128,256
    modal run sweep_cuda.py --layers pooler --ms 16,32

Orientations (qkv_tma.py):
  wa: W is wgmma's A (64 * wg rows a CTA), the tokens its N (bt = m): any m up to 256.
  xa: x is wgmma's A (bt / 64 warpgroups of 64 tokens), a W tile its B (any multiple of 8 rows;
      tiles may mix two sizes so the clusters fill the GPU): m >= 64.
Either way a cluster of cs CTAs splits K and sums through DSMEM. Configs needing more clusters
than fit at once (cuOccupancyMaxActiveClusters) are skipped: they'd run in two waves.
"""

import csv
import json
import time
from pathlib import Path

import modal

PROJ = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "cuda-python")
    .env({"TRITON_DEFAULT_FP_FUSION": "1"})
    .add_local_file(PROJ / "qkv_tma.py", "/root/proj/qkv_tma.py")
    .add_local_file(PROJ / "submission.py", "/root/proj/submission.py")
    .add_local_file(PROJ / "reference.py", "/root/proj/reference.py")
    .add_local_file(PROJ / "checks.py", "/root/proj/checks.py")
)
app = modal.App("muse-sweep-cuda", image=image)

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
SMEM = 227 * 1024 - 2048
# Clusters that fit at once on an H200 at one CTA an SM (measured); refined after compiling.
MAX_CLUSTERS = {1: 132, 2: 66, 3: 39, 4: 30, 6: 17, 8: 15}
JUDGE_FIXED_US = 4.0


def fill_tilings(n, clusters):
    """Tilings of n rows into exactly `clusters` tiles of two adjacent multiples of 8 (or one)."""
    out = []
    base = (n // clusters) // 8 * 8
    for b1 in (base, base + 8):
        b2 = b1 + 8
        # a tiles of b1, clusters - a of b2
        num = b2 * clusters - n
        if num % 8 == 0 and 0 <= num // 8 <= clusters and b1 >= 16:
            a = num // 8
            t = [(b, c) for b, c in ((b1, a), (b2, clusters - a)) if c]
            if all(b <= 256 for b, _ in t):
                out.append(t)
    return out


def candidates(m, n, k):
    nkb = k // 64
    cands = []
    # wa: W as A
    bt = max(8, (m + 7) // 8 * 8)
    if bt <= 256:
        for wg in (1, 2):
            bn = 64 * wg
            if n % bn:
                continue
            tiles = n // bn
            if bt * wg > 256 and bt > 128:   # registers: bt / 2 accumulators a thread
                continue
            for cs in (1, 2, 3, 4, 6, 8):
                if tiles > MAX_CLUSTERS[cs] or tiles * cs < 48 or nkb // cs < 4:
                    continue
                stage = (bn + bt) * 128
                smax = min(16, SMEM // stage)
                for st in sorted({smax, max(3, smax - 2), max(3, smax // 2)}):
                    if st < 3:
                        continue
                    for packw in (1, 0):
                        cands.append(dict(kind="wa", wg=wg, cs=cs, stages=st, bt=bt, extra=dict(PACKW=packw)))
    # xa: x as A
    if m >= 64:
        bt = (m + 63) // 64 * 64
        if bt <= 256:
            tilings = []
            for bn in (64, 96, 128):
                if n % bn == 0:
                    tilings.append([(bn, n // bn)])
            for cs in (1, 2, 3, 4, 6, 8):
                if nkb // cs < 4:
                    continue
                for t in tilings + fill_tilings(n, MAX_CLUSTERS[cs]):
                    clusters = sum(c for _, c in t)
                    bnmax = max(b for b, _ in t)
                    if clusters > MAX_CLUSTERS[cs] or clusters * cs < 48 or bnmax > 160:
                        continue
                    wslot = (bnmax * 128 + 1023) // 1024 * 1024
                    stage = bt * 128 + wslot
                    smax = min(12, SMEM // stage)
                    for st in sorted({smax, max(3, smax - 2)}):
                        if st < 3:
                            continue
                        for kw in ((1, 2) if bt == 128 and bnmax <= 96 else (1,)):
                            cands.append(dict(kind="xa", tiles=t, cs=cs, stages=st, bt=bt, extra=dict(KW=kw)))
    # dedupe
    seen, out = set(), []
    for c in cands:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


@app.function(gpu="H200", timeout=3600)
def sweep_shape(layer: str, m: int, reps: int, top: int):
    import statistics
    import sys

    sys.path.insert(0, "/root/proj")
    import torch
    import torch.nn.functional as F

    import qkv_tma as Q
    import reference as R
    import submission as S

    tag = f"[{layer} m={m}]"
    weights = R.generate_weights(R.WEIGHTS_SEED)
    state = S.prepare(weights)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    data = R.generate_input(m, layer, seed=31 + m)
    checks = [R.generate_input(m, layer, seed=77 + m + i) for i in range(2)]
    x, n, k = data["x"], data["config"]["n"], data["config"]["k"]
    w = weights[layer][:n].contiguous()

    def graph_us(fn, n_reps):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n_reps):
            flush.sum()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        return statistics.median(ts), out

    judge = lambda base, us: (base - JUDGE_FIXED_US) / max(us - JUDGE_FIXED_US, 1e-3)
    cublas_us, _ = graph_us(lambda: F.linear(x, w), reps)
    current_us, _ = graph_us(lambda: S.kernel_fn(data, state), reps)
    cands = candidates(m, n, k)
    print(f"{tag} n={n} k={k}: cuBLAS {cublas_us:.2f} us, current {current_us:.2f} us "
          f"({judge(cublas_us, current_us):.3f}x cuBLAS); {len(cands)} configs", flush=True)

    rows = []
    for c in cands:
        c = json.loads(json.dumps(c))
        kind = c.pop("kind")
        key = json.dumps(dict(kind=kind, **c), separators=(",", ":"))
        row = dict(layer=layer, m=m, n=n, k=k, config=key, us="", status="")
        try:
            if kind == "wa":
                kern = Q.TmaGemm(**c)
                clusters = (n // (64 * kern.wg))
            else:
                c["tiles"] = [tuple(t) for t in c["tiles"]]
                kern = Q.XaGemm(**c)
                clusters = sum(t[1] for t in c["tiles"])
            if kern.smem > 227 * 1024:
                row["status"] = "smem"
                rows.append(row)
                continue
            mc = kern.max_clusters(clusters) if kind == "wa" else kern.max_clusters()
            if clusters > mc:
                row["status"] = f"two waves ({clusters} > {mc})"
                rows.append(row)
                continue
            ok_all = True
            for d in checks:
                out = kern(d["x"], w)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, out, weights)
                ok_all &= ok
            if not ok_all:
                row["status"] = f"WRONG {msg[:80]}"
                print(f"{tag} {key}: WRONG {msg[:120]}", flush=True)
                rows.append(row)
                continue
            us, _ = graph_us(lambda: kern(x, w), reps)
            row.update(us=round(us, 3), status="ok", regs=kern.regs)
            print(f"{tag} {key}: {us:.2f} us  {judge(cublas_us, us):.3f}x cuBLAS  {judge(current_us, us):.3f}x current", flush=True)
        except Exception as e:  # noqa: BLE001
            text = f"{type(e).__name__}: {e}"
            row["status"] = "FAILED " + text[:200]
            print(f"{tag} {key}: FAILED {text[:300]}", flush=True)
            if "illegal" in text or "misaligned" in text or "unspecified launch" in text:
                rows.append(row)
                print(f"{tag} CUDA context broken; stopping", flush=True)
                break
        rows.append(row)

    # Re-time the best few, interleaved with the baselines, at 3x the replays.
    good = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: r["us"])[:top]
    if good:
        cublas_us, _ = graph_us(lambda: F.linear(x, w), 3 * reps)
        current_us, _ = graph_us(lambda: S.kernel_fn(data, state), 3 * reps)
        for r in good:
            c = json.loads(r["config"])
            kind = c.pop("kind")
            if kind == "xa":
                c["tiles"] = [tuple(t) for t in c["tiles"]]
            kern = Q.XaGemm(**c) if kind == "xa" else Q.TmaGemm(**c)
            us, _ = graph_us(lambda: kern(x, w), 3 * reps)
            r["us_retimed"] = round(us, 3)
    for r in rows:
        r["cublas_us"] = round(cublas_us, 3)
        r["current_us"] = round(current_us, 3)
        t = r.get("us_retimed") or r.get("us")
        if t:
            r["x_vs_cublas"] = round(judge(cublas_us, t), 4)
            r["x_vs_current"] = round(judge(current_us, t), 4)
    best = min(good, key=lambda r: r["us_retimed"]) if good else None
    if best:
        print(f"{tag} BEST {best['config']}: {best['us_retimed']:.2f} us, {best['x_vs_cublas']:.3f}x cuBLAS, "
              f"{best['x_vs_current']:.3f}x current (cuBLAS {cublas_us:.2f}, current {current_us:.2f})", flush=True)
    return rows


@app.local_entrypoint()
def main(layers: str = "qkv_gate,o,pooler,unpooler", ms: str = "8,16,32,128,256", reps: int = 100, top: int = 4):
    jobs = [(l, int(m)) for l in layers.split(",") for m in ms.split(",")]
    path = PROJ / f"sweeps/cuda_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    path.parent.mkdir(exist_ok=True)
    fields = ["layer", "m", "n", "k", "config", "status", "regs", "us", "us_retimed", "cublas_us", "current_us",
              "x_vs_cublas", "x_vs_current"]
    summary = []
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for rows in sweep_shape.starmap([(l, m, reps, top) for l, m in jobs], return_exceptions=True):
            if isinstance(rows, Exception):
                print(f"shape failed: {rows}")
                continue
            for r in rows:
                wr.writerow(r)
            f.flush()
            ok = [r for r in rows if r.get("us_retimed")]
            if rows:
                r0 = rows[0]
                b = min(ok, key=lambda r: r["us_retimed"]) if ok else None
                summary.append((r0["layer"], r0["m"], r0["cublas_us"], r0["current_us"],
                                b["us_retimed"] if b else None, b["x_vs_current"] if b else None, b["config"] if b else "-"))
    print(f"\nwrote {path}\n")
    print(f"{'layer':9} {'m':>4} {'cuBLAS':>8} {'current':>8} {'best':>8} {'vs cur':>7}  config")
    for l, m, cb, cur, b, x, cfg in sorted(summary):
        print(f"{l:9} {m:>4} {cb:8.2f} {cur:8.2f} {b if b else float('nan'):8.2f} {x if x else float('nan'):7.3f}  {cfg}")
