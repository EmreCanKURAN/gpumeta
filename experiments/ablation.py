"""Component ablation: twelve selection variants (naive, warp tournament, global
tournaments, the three ballot variants, the elitist rules, direct global best
and two migration rates) on five functions and six algorithms.

The summary holds the mean end-to-end time of each variant, its difference to
naive selection and its mean quality rank (median error, floored at the
tolerance, ranked within each algorithm-function group).

Outputs: ablation_runs.csv, ablation_summary.csv
"""
import os

import pandas as pd

from gpumeta import gpu
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import (ALGORITHMS, CFG, RESULTS_DIR, STATS_DIM, STATS_POP, banner, checkpoint, make_seeds,
                            save_checkpoint, save_csv)
from gpumeta.gpu import GPU_FUNCS
from gpumeta.stats import floor_log10


def run():
    banner("Component ablation of the selection mechanisms")
    gpu.warmup()
    ck = checkpoint.setdefault('ablation', {})
    methods = ['naive', 'tournament', 'global3', 'global4', 'ballot_local', 'ballot', 'ballot_noswitch',
               'tournament_elitist', 'elitist_dev', 'gbest', 'mig_0.03125', 'mig_0.25']
    seeds = make_seeds(CFG['ablation_runs'], offset=200)
    rows = []
    for alg in ALGORITHMS:
        for fname in CFG['ablation_funcs']:
            key = f"{alg}_{fname}"
            fid = BENCHMARKS[fname]['id']; bnd = BENCHMARKS[fname]['bounds']
            if key not in ck:
                print(f"  {alg} | {fname}", flush=True)
                ck[key] = {}
                for m in methods:
                    rr = [GPU_FUNCS[alg](STATS_POP, CFG['ablation_iters'], fid, bnd, dim=STATS_DIM, method=m, seed=sd)
                          for sd in seeds]
                    ck[key][m] = [{'best': r['best'], 'time_ms': r['time_ms']} for r in rr]
                save_checkpoint()
            for m, rr in ck[key].items():
                for i, r in enumerate(rr):
                    rows.append({'algorithm': alg, 'function': fname, 'method': m, 'run': i, 'seed': seeds[i],
                                 'best': r['best'], 'time_ms': r['time_ms']})
    df = save_csv(rows, 'ablation_runs')
    t = df.groupby('method')['time_ms'].mean()
    # quality rank: rank methods per (alg, function) by median floored log error, then average
    med = df.groupby(['algorithm', 'function', 'method'])['best'].median().reset_index()
    med['lg'] = floor_log10(med['best'].values)
    med['rank'] = med.groupby(['algorithm', 'function'])['lg'].rank()
    rk = med.groupby('method')['rank'].mean()
    tab = pd.DataFrame({'mean_time_ms': t, 'delta_vs_naive_%': 100 * (t / t['naive'] - 1), 'quality_rank': rk})
    tab = tab.reindex(methods)
    tab.to_csv(os.path.join(RESULTS_DIR, 'ablation_summary.csv'))
    print(tab.round(3).to_string())


if __name__ == '__main__':
    run()
