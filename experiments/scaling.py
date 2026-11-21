"""Scaling of the five execution platforms (CPU Serial, CPU Parallel, GPU Naive,
GPU Tournament, GPU Ballot-Elite) with population size, problem dimension and
iteration count, for six algorithms and ten test functions.

Outputs: scaling_population.csv, scaling_dimension.csv, scaling_iterations.csv

    python -m experiments.scaling                       # all three sweeps
    python -m experiments.scaling population            # one sweep
    python -m experiments.scaling population Rastrigin  # one sweep, one function
"""
import sys

import numpy as np

from gpumeta import cpu, gpu
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import (ALGORITHMS, DEFAULT_DIM, DEFAULT_ITERS, DEFAULT_POP, EXP_DIMENSIONS,
                            EXP_ITERATIONS, EXP_POP_SIZES, MASTER_SEED, SERIAL_CAP, TEST_FUNCTIONS,
                            banner, checkpoint, save_checkpoint, save_csv)
from gpumeta.cpu import CPU_PARALLEL_FUNCS, CPU_SERIAL_FUNCS, run_cpu_timed
from gpumeta.gpu import GPU_FUNCS

SWEEPS = {
    'population': ('ps', EXP_POP_SIZES, {'iters': DEFAULT_ITERS, 'dim': DEFAULT_DIM, 'ps': None}),
    'dimension': ('dim', EXP_DIMENSIONS, {'iters': DEFAULT_ITERS, 'ps': DEFAULT_POP, 'dim': None}),
    'iterations': ('iters', EXP_ITERATIONS, {'ps': DEFAULT_POP, 'dim': DEFAULT_DIM, 'iters': None}),
}


def sweep(tag, sweep_name, sweep_values, fixed, functions=TEST_FUNCTIONS):
    """Run all five platforms for every algorithm, function and value of the
    swept parameter. fixed holds the two parameters that stay constant."""
    banner(f"{tag}: {sweep_name} = {sweep_values}, {len(functions)} functions")
    ck = checkpoint.setdefault(tag, {})
    rows = []
    for fname in functions:
        fid = BENCHMARKS[fname]['id']; lb, ub = BENCHMARKS[fname]['bounds']
        for v in sweep_values:
            cfg = dict(fixed); cfg[sweep_name] = v
            skip_serial = (cfg['ps'] * cfg['dim'] * cfg['iters'] > SERIAL_CAP)
            for alg in ALGORITHMS:
                key = f"{alg}_{fname}_{v}"
                if key in ck:
                    res = ck[key]
                else:
                    print(f"  {alg} | {fname} | {sweep_name}={v}", flush=True)
                    res = {}
                    if not skip_serial:
                        np.random.seed(MASTER_SEED)
                        bb, tt = run_cpu_timed(CPU_SERIAL_FUNCS[alg], fid, lb, ub,
                                               cfg['ps'], cfg['iters'], cfg['dim'])
                        res['CPU_Serial'] = {'best': bb, 'time_ms': tt}
                    np.random.seed(MASTER_SEED)
                    bb, tt = run_cpu_timed(CPU_PARALLEL_FUNCS[alg], fid, lb, ub,
                                           cfg['ps'], cfg['iters'], cfg['dim'])
                    res['CPU_Parallel'] = {'best': bb, 'time_ms': tt}
                    for plat, m in [('GPU_Naive', 'naive'),
                                    ('GPU_Tournament', 'tournament'),
                                    ('GPU_Ballot', 'ballot')]:
                        res[plat] = GPU_FUNCS[alg](cfg['ps'], cfg['iters'], fid, (lb, ub),
                                                   dim=cfg['dim'], method=m, seed=MASTER_SEED)
                    ck[key] = res
                    save_checkpoint()
                for plat, r in res.items():
                    rows.append({'algorithm': alg, 'function': fname,
                                 sweep_name: v, **{k: cfg[k] for k in cfg},
                                 'platform': plat, 'best': r['best'], 'time_ms': r['time_ms'],
                                 'speedup_vs_serial': (res['CPU_Serial']['time_ms'] / r['time_ms'])
                                 if 'CPU_Serial' in res else float('nan')})
    df = save_csv(rows, tag)
    # function-averaged speedup of GPU Tournament over CPU Serial
    try:
        piv = (df[df.platform == 'GPU_Tournament']
               .pivot_table(index='algorithm', columns=sweep_name,
                            values='speedup_vs_serial', aggfunc='mean').round(2))
        print("\n  mean GPU Tournament speedup over CPU Serial:")
        print(piv)
    except Exception as e:
        print(f"  summary skipped: {e}")
    return df


def run(which=None, functions=None):
    cpu.warmup()
    gpu.warmup()
    for name in (which or list(SWEEPS)):
        sweep_name, values, fixed = SWEEPS[name]
        tag = f"scaling_{name}"
        if functions:
            tag += '_' + '_'.join(functions)
        sweep(tag, sweep_name, values, fixed, functions or TEST_FUNCTIONS)


if __name__ == '__main__':
    run(sys.argv[1:2] or None, sys.argv[2:] or None)
