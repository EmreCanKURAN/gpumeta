"""Shifted and rotated CEC2017-class functions evaluated entirely on the GPU
(official shift vectors and rotation matrices copied to device memory).

Reports the ratio of the CPU (opfunu) evaluation time per iteration to the GPU
time of one complete GWO iteration, and the per-iteration overhead of the warp
selection mechanisms over naive selection, across population sizes.

Outputs: cec_gpu_throughput.csv, cec_gpu_throughput.png
"""
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gpumeta import gpu
from gpumeta.cec import check_opfunu
from gpumeta.config import (DEFAULT_DIM, DEFAULT_ITERS, EXP_POP_SIZES, MASTER_SEED, RESULTS_DIR, banner,
                            checkpoint, save_checkpoint, save_csv)
from gpumeta.gpu import CECG_FUNCS, load_cecg, run_gwo_gpu


def run():
    tag = 'cec_gpu_throughput'
    banner("On-device CEC functions: evaluation throughput and selection overhead (GWO)")
    check_opfunu()
    gpu.warmup()
    ck = checkpoint.setdefault(tag, {})
    dim = min(DEFAULT_DIM, 100)            # the device arrays hold at most 128 dimensions
    iters = DEFAULT_ITERS
    pops = EXP_POP_SIZES
    rows = []
    for fid, (cls, label) in CECG_FUNCS.items():
        for ps in pops:
            key = f"{fid}_{ps}"
            if key in ck:
                res = ck[key]
            else:
                print(f"  {label} | pop={ps}", flush=True)
                load_cecg(fid, dim)
                res = {}
                for m in ['naive', 'tournament', 'ballot']:
                    r = run_gwo_gpu(ps, iters, fid, (-100, 100), dim=dim, method=m, seed=MASTER_SEED)
                    res[m] = {'time_ms': r['time_ms']}
                # CPU reference: time to evaluate one population with opfunu
                fobj = getattr(__import__('opfunu.cec_based.cec2017', fromlist=[cls]), cls)(ndim=dim)
                Xs = np.random.RandomState(0).uniform(-100, 100, (min(ps, 2048), dim))
                t0 = time.time()
                for x in Xs:
                    fobj.evaluate(x)
                cpu_eval_ms = (time.time() - t0) * 1000 / len(Xs) * ps  # per iteration
                res['_cpu_eval_ms_per_iter'] = cpu_eval_ms
                ck[key] = res
                save_checkpoint()
            for m in ['naive', 'tournament', 'ballot']:
                gpu_total = res[m]['time_ms']
                gpu_per_iter = gpu_total / iters
                cpu_iter = res['_cpu_eval_ms_per_iter']
                rows.append({'function': CECG_FUNCS[fid][1], 'fid': fid, 'pop': ps,
                             'method': m,
                             'gpu_time_ms': gpu_total, 'gpu_ms_per_iter': gpu_per_iter,
                             'cpu_eval_ms_per_iter': cpu_iter,
                             'eval_speedup': cpu_iter / gpu_per_iter})
    df = save_csv(rows, tag)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label in df['function'].unique():
            d = df[(df['function'] == label) & (df.method == 'tournament')]
            ax.plot(d['pop'], d['eval_speedup'], marker='o', label=label)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('population size'); ax.set_ylabel('GPU vs CPU-eval speedup (per iteration)')
        ax.set_title('on-device CEC throughput')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(RESULTS_DIR, f"{tag}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped: {e}")
    # selection overhead relative to naive selection
    try:
        piv = df.pivot_table(index=['function', 'pop'], columns='method', values='gpu_time_ms')
        piv['tourn_overhead_%'] = 100 * (piv['tournament'] - piv['naive']) / piv['naive']
        piv['ballot_overhead_%'] = 100 * (piv['ballot'] - piv['naive']) / piv['naive']
        print("\n  mean selection overhead over naive:")
        print(f"    tournament: {piv['tourn_overhead_%'].mean():+.1f}%  |  ballot: {piv['ballot_overhead_%'].mean():+.1f}%")
    except Exception as e:
        print(f"  overhead summary skipped: {e}")
    return df


if __name__ == '__main__':
    run()
