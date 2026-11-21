"""Native CUDA GWO with the three selection mechanisms against GWO and PSO
written with CuPy array operations (D = 100, 1000 iterations).

Output: library_baseline.csv
"""
import time

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from gpumeta import gpu
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import MASTER_SEED, banner, checkpoint, save_checkpoint, save_csv
from gpumeta.gpu import run_gwo_gpu


def _cupy_eval(name, X):
    if name == 'Sphere':
        return (X ** 2).sum(axis=1)
    if name == 'Rastrigin':
        return (X ** 2 + 20.0 * cp.sin(cp.pi * X) ** 2).sum(axis=1)
    raise ValueError(name)


def gwo_cupy(name, lb, ub, ps, iters, dim, seed):
    rng = cp.random.RandomState(seed)
    pos = lb + rng.rand(ps, dim) * (ub - lb)
    best = np.inf
    t0 = time.time()
    for it in range(iters):
        cp.clip(pos, lb, ub, out=pos)
        obj = _cupy_eval(name, pos)
        order = cp.argsort(obj)[:3]
        a_p, b_p, d_p = pos[order[0]], pos[order[1]], pos[order[2]]
        cur = float(obj[order[0]])
        if cur < best:
            best = cur
        a = 2.0 - 2.0 * it / iters

        def step(lead):
            r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
            A = 2 * a * r1 - a; C = 2 * r2
            return lead - A * cp.abs(C * lead - pos)
        pos = (step(a_p) + step(b_p) + step(d_p)) / 3.0
    cp.cuda.Stream.null.synchronize()
    return best, (time.time() - t0) * 1000


def pso_cupy(name, lb, ub, ps, iters, dim, seed):
    rng = cp.random.RandomState(seed)
    pos = lb + rng.rand(ps, dim) * (ub - lb)
    vel = cp.zeros((ps, dim))
    pbest = pos.copy()
    pbest_obj = _cupy_eval(name, pos)
    gi = int(cp.argmin(pbest_obj))
    gbest = pbest[gi].copy(); gbest_obj = float(pbest_obj[gi])
    w, c1, c2 = 0.729, 1.49445, 1.49445
    t0 = time.time()
    for it in range(iters):
        r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = cp.clip(pos + vel, lb, ub)
        obj = _cupy_eval(name, pos)
        mask = obj < pbest_obj
        pbest_obj = cp.where(mask, obj, pbest_obj)
        pbest = cp.where(mask[:, None], pos, pbest)
        gi = int(cp.argmin(pbest_obj))
        if float(pbest_obj[gi]) < gbest_obj:
            gbest_obj = float(pbest_obj[gi]); gbest = pbest[gi].copy()
    cp.cuda.Stream.null.synchronize()
    return gbest_obj, (time.time() - t0) * 1000


def run():
    tag = 'library_baseline'
    gpu.warmup()
    banner("CuPy implementations against the native CUDA GWO")
    if cp is None:
        print("  skipped: CuPy is not installed (pip install cupy-cuda12x)")
        return None
    ck = checkpoint.setdefault(tag, {})
    rows = []
    iters, dim = 1000, 100
    # warm-up: kernel compilation and memory-pool allocation of CuPy stay out of the timings
    for fname in ['Sphere', 'Rastrigin']:
        lbw, ubw = BENCHMARKS[fname]['bounds']
        gwo_cupy(fname, lbw, ubw, 8192, 5, dim, 1)
        pso_cupy(fname, lbw, ubw, 8192, 5, dim, 1)
    for fname in ['Sphere', 'Rastrigin']:
        lb, ub = BENCHMARKS[fname]['bounds']
        for ps in [1024, 8192]:
            key = f"{fname}_{ps}"
            if key in ck:
                res = ck[key]
            else:
                print(f"  {fname} | pop={ps}", flush=True)
                res = {}
                b, t = gwo_cupy(fname, lb, ub, ps, iters, dim, MASTER_SEED)
                res['CuPy_GWO'] = {'best': b, 'time_ms': t}
                b, t = pso_cupy(fname, lb, ub, ps, iters, dim, MASTER_SEED)
                res['CuPy_PSO'] = {'best': b, 'time_ms': t}
                fid = BENCHMARKS[fname]['id']
                for m in ['naive', 'tournament', 'ballot']:
                    res[f'CUDA_GWO_{m}'] = run_gwo_gpu(ps, iters, fid, (lb, ub), dim=dim,
                                                       method=m, seed=MASTER_SEED)
                ck[key] = res
                save_checkpoint()
            for plat, r in res.items():
                rows.append({'function': fname, 'pop': ps, 'platform': plat,
                             'best': r['best'], 'time_ms': r['time_ms']})
    df = save_csv(rows, tag)
    print(df.pivot_table(index=['function', 'pop'], columns='platform', values='time_ms').to_string())
    return df


if __name__ == '__main__':
    run()
