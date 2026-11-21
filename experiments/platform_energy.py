"""Runtime, energy, average and peak power and energy-delay product of the five
platforms at the default configuration (GPU power sampled through NVML).

Output: platform_energy.csv
"""
import numpy as np

from gpumeta import cpu, gpu
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import (ALGORITHMS, DEFAULT_DIM, DEFAULT_ITERS, DEFAULT_POP, MASTER_SEED, TEST_FUNCTIONS,
                            banner, checkpoint, save_checkpoint, save_csv)
from gpumeta.cpu import CPU_PARALLEL_FUNCS, CPU_SERIAL_FUNCS, run_cpu_timed
from gpumeta.energy import NVML_OK, run_with_energy
from gpumeta.gpu import GPU_FUNCS


def run_all_platforms(alg, fname, ps, iters, dim, seed, with_energy=False):
    fid = BENCHMARKS[fname]['id']
    lb, ub = BENCHMARKS[fname]['bounds']
    out = {}
    np.random.seed(seed)
    best, t = run_cpu_timed(CPU_SERIAL_FUNCS[alg], fid, lb, ub, ps, iters, dim)
    out['CPU_Serial'] = {'best': best, 'time_ms': t,
                         'energy_j': float('nan'), 'avg_w': float('nan'), 'peak_w': float('nan'), 'edp': float('nan')}
    np.random.seed(seed)
    best, t = run_cpu_timed(CPU_PARALLEL_FUNCS[alg], fid, lb, ub, ps, iters, dim)
    out['CPU_Parallel'] = {'best': best, 'time_ms': t,
                           'energy_j': float('nan'), 'avg_w': float('nan'), 'peak_w': float('nan'), 'edp': float('nan')}
    for plat, m in [('GPU_Naive', 'naive'), ('GPU_Tournament', 'tournament'), ('GPU_Ballot', 'ballot')]:
        if with_energy:
            out[plat] = run_with_energy(GPU_FUNCS[alg], ps, iters, fid, (lb, ub), dim=dim, method=m, seed=seed)
        else:
            r = GPU_FUNCS[alg](ps, iters, fid, (lb, ub), dim=dim, method=m, seed=seed)
            r.update({'energy_j': float('nan'), 'avg_w': float('nan'), 'peak_w': float('nan'), 'edp': float('nan')})
            out[plat] = r
    return out


def run():
    tag = 'platform_energy'
    cpu.warmup()
    gpu.warmup()
    banner(f"Platform comparison with energy (pop={DEFAULT_POP}, dim={DEFAULT_DIM}, iters={DEFAULT_ITERS})")
    if not NVML_OK:
        print("  NVML is not available: GPU energy is reported as NaN (pip install nvidia-ml-py)")
    ck = checkpoint.setdefault(tag, {})
    rows = []
    for alg in ALGORITHMS:
        for fname in TEST_FUNCTIONS:
            key = f"{alg}_{fname}"
            if key in ck:
                res = ck[key]
            else:
                print(f"  {alg} | {fname}", flush=True)
                res = run_all_platforms(alg, fname, DEFAULT_POP, DEFAULT_ITERS, DEFAULT_DIM,
                                        MASTER_SEED, with_energy=True)
                ck[key] = res
                save_checkpoint()
            tser = res['CPU_Serial']['time_ms']
            for plat, r in res.items():
                t_s = r['time_ms'] / 1000.0
                e = r.get('energy_j', float('nan'))
                rows.append({'algorithm': alg, 'function': fname, 'platform': plat,
                             'best': r['best'], 'time_ms': r['time_ms'],
                             'speedup': tser / r['time_ms'],
                             'energy_j': e, 'avg_w': r.get('avg_w', float('nan')),
                             'peak_w': r.get('peak_w', float('nan')),
                             'perf_per_watt': (1.0 / t_s) / r['avg_w'] if np.isfinite(r.get('avg_w', np.nan)) and r.get('avg_w', 0) > 0 else float('nan'),
                             'edp': r.get('edp', float('nan'))})
    df = save_csv(rows, tag)
    piv = df[df.platform.str.startswith('GPU')].groupby('platform')[['time_ms', 'energy_j', 'edp']].mean()
    print("\n  GPU means over all algorithms and functions:")
    print(piv.to_string())
    return df


if __name__ == '__main__':
    run()
