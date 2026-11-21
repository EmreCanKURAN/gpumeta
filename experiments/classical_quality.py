"""Solution quality on the ten classical functions.

The GPU kernels are compiled with exact double-precision constants and the
best position of every run is evaluated again with the CPU reference
evaluator (Numba), so GPU and CPU results are measured with the same
function. Errors are reported relative to the exact optimum.

run_stability  30 seed-paired runs per algorithm and function at N = 1024,
               D = 50, 1000 iterations for five GPU selection mechanisms, plus
               CPU Serial; preceded by a check of the GPU evaluator
run_large      one run at N = 8192, D = 100, 5000 iterations for naive,
               Tournament and Ballot-Elite

Outputs: evaluator_check.csv, classical_runs.csv, classical_large_runs.csv
"""
import math

import numpy as np
import pycuda.gpuarray as gpuarray
from scipy.optimize import minimize_scalar

from gpumeta import cpu, gpu
from gpumeta.benchmarks import BENCHMARKS, evaluate_numba, optimum_value
from gpumeta.config import (ALGORITHMS, CFG, DEFAULT_DIM, MASTER_SEED, STATS_DIM, STATS_POP, TEST_FUNCTIONS,
                            banner, checkpoint, make_seeds, save_checkpoint, save_csv)
from gpumeta.cpu import CPU_SERIAL_FUNCS, run_cpu_timed
from gpumeta.gpu import (BASE_HANDLES, BS, exact_constants_handles, grid, run_with_position,
                         track_best_position, use_kernels)

QUALITY_METHODS = ['naive', 'tournament', 'elitist_dev', 'mig_0.25', 'ballot']


def quality_runs(tag, pop, iters, dim, funcs, seeds, methods, cpu=True):
    banner(f"Classical functions, exact constants and CPU reference evaluator (pop={pop}, dim={dim}, "
           f"iters={iters}, runs={len(seeds)})")
    h = exact_constants_handles()
    ck = checkpoint.setdefault(tag, {})
    rows = []
    for fname in funcs:
        fid = BENCHMARKS[fname]['id']; bnd = BENCHMARKS[fname]['bounds']; fs = optimum_value(fname, dim)
        for alg in ALGORITHMS:
            key = f"{alg}_{fname}"
            if key not in ck:
                print(f"  [{tag}] {alg} | {fname}", flush=True)
                ck[key] = {}
                with use_kernels(h), track_best_position():
                    for m in methods:
                        lst = []
                        for sd in seeds:
                            r, x = run_with_position(alg, pop, iters, fid, bnd, dim, m, sd)
                            ref = float(evaluate_numba(np.clip(x, bnd[0], bnd[1]).astype(np.float64), fid))
                            lst.append({'gpu_value': r['best'], 'ref_value': ref, 'time_ms': r['time_ms']})
                        ck[key][m] = lst
                save_checkpoint()
            if cpu and 'cpu_serial' not in ck[key]:
                lb, ub = bnd
                lst = []
                for sd in seeds:
                    np.random.seed(sd)
                    bb, tt = run_cpu_timed(CPU_SERIAL_FUNCS[alg], fid, lb, ub, pop, iters, dim)
                    lst.append({'gpu_value': float('nan'), 'ref_value': float(bb), 'time_ms': tt})
                ck[key]['cpu_serial'] = lst
                save_checkpoint()
            for m, lst in ck[key].items():
                for i, d in enumerate(lst):
                    rows.append({'algorithm': alg, 'function': fname, 'method': m, 'run': i, 'seed': seeds[i],
                                 'gpu_value': d['gpu_value'], 'ref_value': d['ref_value'],
                                 'f_star': fs, 'error_ref': d['ref_value'] - fs,
                                 'evaluator_abs_diff': abs(d['gpu_value'] - d['ref_value'])
                                 if np.isfinite(d['gpu_value']) else float('nan'),
                                 'time_ms': d['time_ms']})
    df = save_csv(rows, f'{tag}_runs')
    diff = df[np.isfinite(df.evaluator_abs_diff)]
    if len(diff):
        print(f"  max |GPU value - CPU reference| = {diff.evaluator_abs_diff.max():.3e}  "
              f"(median {diff.evaluator_abs_diff.median():.3e})")
    return df


def evaluator_check(dim):
    """Objective values of the GPU evaluator with the original and with the
    exact constants against the CPU reference, at the known optimum and at 256
    random points of every function."""
    exact_handles = exact_constants_handles()
    rows = []
    xopt = {'Sphere': 0.0, 'Rastrigin': 0.0, 'Rosenbrock': 1.0, 'Ackley': 0.0, 'Griewank': 0.0,
            'Levy': 1.0, 'Zakharov': 0.0, 'SumSquares': 0.0}
    xopt['Schwefel'] = float(minimize_scalar(lambda x: -x * math.sin(math.sqrt(abs(x))), bounds=(400.0, 440.0),
                                             method='bounded', options={'xatol': 1e-12}).x)
    for fname in TEST_FUNCTIONS:
        fid = BENCHMARKS[fname]['id']; lb, ub = BENCHMARKS[fname]['bounds']
        rng = np.random.RandomState(7)
        P = rng.uniform(lb, ub, (256, dim))
        has_opt = fname in xopt
        if has_opt:
            P[0, :] = xopt[fname]
        d_pop = gpuarray.to_gpu(P.ravel().astype(np.float64)); d_o = gpuarray.zeros(256, np.float64)
        out = {}
        for label, kern in [('original', BASE_HANDLES['k_eval_pop']), ('exact', exact_handles['k_eval_pop'])]:
            kern(d_pop, d_o, np.int32(256), np.int32(dim), np.int32(fid), block=(BS, 1, 1), grid=grid(256))
            out[label] = d_o.get()
        cpu = np.array([evaluate_numba(P[i].copy(), fid) for i in range(256)])
        rel = lambda a: float(np.max(np.abs(a - cpu) / np.maximum(1.0, np.abs(cpu))))
        rows.append({'function': fname, 'dim': dim,
                     'f_at_optimum_cpu': float(cpu[0]) if has_opt else float('nan'),
                     'f_at_optimum_gpu_original': float(out['original'][0]) if has_opt else float('nan'),
                     'f_at_optimum_gpu_exact': float(out['exact'][0]) if has_opt else float('nan'),
                     'f_star_used': optimum_value(fname, dim),
                     'max_rel_diff_original_vs_cpu': rel(out['original']),
                     'max_rel_diff_exact_vs_cpu': rel(out['exact'])})
    df = save_csv(rows, 'evaluator_check')
    print(df.to_string(index=False))


def run_stability():
    """Seed-paired runs at the reduced configuration, with CPU Serial."""
    evaluator_check(STATS_DIM)
    return quality_runs('classical', STATS_POP, CFG['quality_iters'], STATS_DIM, CFG['quality_funcs'],
                        make_seeds(CFG['quality_runs'], offset=100), QUALITY_METHODS, cpu=True)


def run_large():
    """One seeded run per configuration at the largest population."""
    return quality_runs('classical_large', CFG['large_pop'], CFG['large_iters'], DEFAULT_DIM,
                        CFG['large_funcs'], [MASTER_SEED], ['naive', 'tournament', 'ballot'], cpu=False)


def run():
    cpu.warmup()
    gpu.warmup()
    run_stability()
    run_large()


if __name__ == '__main__':
    run()
