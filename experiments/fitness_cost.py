"""End-to-end speedup over CPU Serial as a function of a synthetic evaluation
cost multiplier k (GWO, N = 1024, D = 50, 200 iterations).

Outputs: fitness_cost.csv, fitness_cost.png
"""
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gpumeta import cpu, gpu
from gpumeta.benchmarks import EXPENSIVE_FID
from gpumeta.config import COST_GRID, MASTER_SEED, RESULTS_DIR, banner, checkpoint, save_checkpoint, save_csv
from gpumeta.cpu import gwo_serial_expensive
from gpumeta.gpu import run_gwo_gpu, set_fitness_cost


def run():
    tag = 'fitness_cost'
    cpu.warmup()
    gpu.warmup()
    banner(f"Fitness-cost break-even: k = {COST_GRID} (GWO on the cost-scalable sphere)")
    ck = checkpoint.setdefault(tag, {})
    ps, dim, iters = 1024, 50, 200
    bnd = (-100, 100)
    rows = []
    for k in COST_GRID:
        key = f"k_{k}"
        if key in ck:
            res = ck[key]
        else:
            print(f"  k={k}", flush=True)
            set_fitness_cost(k)
            res = {}
            t0 = time.time()
            best_cpu = gwo_serial_expensive(np.float64(bnd[0]), np.float64(bnd[1]),
                                            ps, iters, dim, k, MASTER_SEED)
            res['CPU_Serial'] = {'best': float(best_cpu), 'time_ms': (time.time() - t0) * 1000}
            for m in ['naive', 'tournament', 'ballot']:
                res[f'GPU_{m}'] = run_gwo_gpu(ps, iters, EXPENSIVE_FID, bnd, dim=dim,
                                              method=m, seed=MASTER_SEED)
            set_fitness_cost(1)
            ck[key] = res
            save_checkpoint()
        tser = res['CPU_Serial']['time_ms']
        for plat, r in res.items():
            rows.append({'cost_k': k, 'platform': plat, 'best': r['best'],
                         'time_ms': r['time_ms'], 'speedup_vs_serial': tser / r['time_ms']})
    df = save_csv(rows, tag)
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        for plat in ['GPU_naive', 'GPU_tournament', 'GPU_ballot']:
            d = df[df.platform == plat]
            ax.plot(d['cost_k'], d['speedup_vs_serial'], marker='o', label=plat)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('fitness cost multiplier k'); ax.set_ylabel('speedup vs CPU serial')
        ax.set_title('break-even vs fitness cost'); ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(RESULTS_DIR, f"{tag}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped: {e}")
    return df


if __name__ == '__main__':
    run()
