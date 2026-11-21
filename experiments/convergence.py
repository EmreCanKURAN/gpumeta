"""Convergence curves of the GPU implementations at the default configuration
(best-so-far value every ten iterations; mean, standard deviation and minimum
over the seeded runs).

The curves are recorded with the kernels as written in metaheuristics.cu,
i.e. with the truncated constants, so converged Ackley curves end at the
level of about 1.5e-9 discussed in the paper.

Output: convergence.csv
"""
import numpy as np

from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import (ALGORITHMS, DEFAULT_DIM, DEFAULT_ITERS, DEFAULT_POP, N_RUNS_SENS, STATS_METHODS,
                            TEST_FUNCTIONS, banner, checkpoint, make_seeds, save_checkpoint, save_csv)
from gpumeta.gpu import GPU_FUNCS


def run():
    tag = 'convergence'
    CONV_POP, CONV_DIM, CONV_ITERS = DEFAULT_POP, DEFAULT_DIM, DEFAULT_ITERS
    banner(f"Convergence curves (pop={CONV_POP}, dim={CONV_DIM}, iters={CONV_ITERS})")
    ck = checkpoint.setdefault(tag, {})
    conv_runs = max(5, N_RUNS_SENS // 3)
    seeds = make_seeds(conv_runs, offset=700)
    rows = []
    for alg in ALGORITHMS:
        for fname in TEST_FUNCTIONS:
            key = f"{alg}_{fname}"
            if key in ck:
                curves = ck[key]
            else:
                print(f"  {alg} | {fname}", flush=True)
                fid = BENCHMARKS[fname]['id']; bnd = BENCHMARKS[fname]['bounds']
                curves = {}
                for m in STATS_METHODS:
                    per_iter = {}
                    for sd in seeds:
                        r = GPU_FUNCS[alg](CONV_POP, CONV_ITERS, fid, bnd, dim=CONV_DIM,
                                           method=m, seed=sd, track_convergence=True)
                        for it, bb in r['convergence']:
                            per_iter.setdefault(it, []).append(bb)
                    curves[m] = {it: (float(np.mean(v)), float(np.std(v)), float(np.min(v)))
                                 for it, v in sorted(per_iter.items())}
                ck[key] = curves
                save_checkpoint()
            for m, ci in curves.items():
                for it, (mu, sg, mn) in ci.items():
                    rows.append({'algorithm': alg, 'function': fname, 'method': m,
                                 'iter': it, 'mean_best': mu, 'std_best': sg, 'min_best': mn})
    save_csv(rows, tag)


if __name__ == '__main__':
    run()
