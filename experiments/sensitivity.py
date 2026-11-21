"""Sensitivity of Ballot-Elite to the threshold factor alpha and of the warp
tournament to the tournament parameter ts.

Outputs: sensitivity.csv, sensitivity_alpha.png, sensitivity_ts.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import (ALGORITHMS, ALPHA_GRID, N_RUNS_SENS, RESULTS_DIR, SENS_FUNCS, STATS_DIM, STATS_ITERS,
                            STATS_POP, TS_GRID, banner, checkpoint, make_seeds, save_checkpoint, save_csv)
from gpumeta.gpu import GPU_FUNCS
from gpumeta.stats import summarize_runs


def run():
    tag = 'sensitivity'
    banner(f"Parameter sensitivity: alpha = {ALPHA_GRID} (Ballot-Elite), ts = {TS_GRID} (Tournament), "
           f"{N_RUNS_SENS} runs")
    ck = checkpoint.setdefault(tag, {})
    seeds = make_seeds(N_RUNS_SENS, offset=300)
    rows = []
    for alg in ALGORITHMS:
        for fname in SENS_FUNCS:
            fid = BENCHMARKS[fname]['id']; bnd = BENCHMARKS[fname]['bounds']
            for a in ALPHA_GRID:
                key = f"{alg}_{fname}_alpha_{a}"
                if key not in ck:
                    print(f"  {alg} | {fname} | alpha={a}", flush=True)
                    vals = [GPU_FUNCS[alg](STATS_POP, STATS_ITERS, fid, bnd, dim=STATS_DIM,
                                           method='ballot', alpha=a, seed=sd)['best'] for sd in seeds]
                    ck[key] = vals
                    save_checkpoint()
                s = summarize_runs(ck[key])
                rows.append({'algorithm': alg, 'function': fname, 'param': 'alpha',
                             'value': a, 'method': 'ballot', **s})
            for ts in TS_GRID:
                key = f"{alg}_{fname}_ts_{ts}"
                if key not in ck:
                    print(f"  {alg} | {fname} | ts={ts}", flush=True)
                    vals = [GPU_FUNCS[alg](STATS_POP, STATS_ITERS, fid, bnd, dim=STATS_DIM,
                                           method='tournament', ts=ts, seed=sd)['best'] for sd in seeds]
                    ck[key] = vals
                    save_checkpoint()
                s = summarize_runs(ck[key])
                rows.append({'algorithm': alg, 'function': fname, 'param': 'ts',
                             'value': ts, 'method': 'tournament', **s})
    df = save_csv(rows, tag)
    try:
        for param, gridv in [('alpha', ALPHA_GRID), ('ts', TS_GRID)]:
            sub = df[df.param == param]
            mat = np.zeros((len(ALGORITHMS), len(gridv)))
            for i, alg in enumerate(ALGORITHMS):
                for j, v in enumerate(gridv):
                    d = sub[(sub.algorithm == alg) & (sub.value == v)]
                    mat[i, j] = np.mean(np.log10(np.maximum(d['median'].values, 1e-300)))
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(mat, aspect='auto', cmap='viridis_r')
            ax.set_xticks(range(len(gridv))); ax.set_xticklabels(gridv)
            ax.set_yticks(range(len(ALGORITHMS))); ax.set_yticklabels(ALGORITHMS)
            ax.set_xlabel(param); ax.set_title(f"mean log10(median best) vs {param}")
            fig.colorbar(im)
            fig.savefig(os.path.join(RESULTS_DIR, f"sensitivity_{param}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        print(f"  plot skipped: {e}")
    return df


if __name__ == '__main__':
    run()
