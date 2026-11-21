"""Isolated cost of the selection operators: microseconds per call measured
with CUDA events for population sizes from 128 to about one million.

ballot_elite_full and ballot_local_full include the fitness transformation
(and the mean) that the operator needs; ballot_elite_select_only times the
selection kernel alone.

Outputs: selection_microbenchmark.csv, selection_microbenchmark_pivot.csv
"""
import os

import numpy as np
import pycuda.gpuarray as gpuarray

from gpumeta import gpu
from gpumeta.config import CFG, RESULTS_DIR, banner, save_csv
from gpumeta.gpu import BS, GPU_NAME, SELK, fill_selection, grid, seed32, time_events
from gpumeta.reference import emu_fit


def run():
    banner("Isolated cost of the selection operators (CUDA events, microseconds per call)")
    rows = []
    for n in CFG['micro_ns']:
        rng = np.random.RandomState(n)
        obj = rng.uniform(1.0, 2.0, n)
        d_obj = gpuarray.to_gpu(obj); d_fit = gpuarray.to_gpu(emu_fit(obj))
        d_sel = gpuarray.zeros(n, np.int32); d_avg = gpuarray.zeros(1, np.float64)
        d_cum = gpuarray.zeros(n, np.float64)
        d_bi = gpuarray.zeros(1, np.int32); d_bv = gpuarray.zeros(1, np.float64)
        g = grid(n); s32 = seed32(123)
        fill_selection('ballot_safe', d_obj, d_fit, d_sel, d_avg, n, 3, 1.2, 1)   # leaves a valid mean in d_avg
        V = {
            'roulette_serial_cumsum': lambda: (
                gpu.k_sel_naive(d_fit, d_cum, np.int32(n), block=(1, 1, 1), grid=(1, 1)),
                SELK['roulette_sample'](d_cum, d_sel, np.int32(n), s32, block=(BS, 1, 1), grid=g)),
            'roulette_parallel_scan': lambda: fill_selection('roulette_scan', d_obj, d_fit, d_sel,
                                                             d_avg, n, 3, 1.2, 123),
            'global_tournament_ts3': lambda: gpu.k_sel_tourn_global(d_obj, d_sel, np.int32(n), np.int32(3), s32,
                                                                    block=(BS, 1, 1), grid=g),
            'global_tournament_ts4': lambda: gpu.k_sel_tourn_global(d_obj, d_sel, np.int32(n), np.int32(4), s32,
                                                                    block=(BS, 1, 1), grid=g),
            'shared_block_tournament_ts3': lambda: SELK['sel_tournament_shared'](d_obj, d_sel, np.int32(n),
                                                                                 np.int32(3), s32, block=(BS, 1, 1),
                                                                                 grid=g, shared=BS * 8),
            'warp_tournament_ts3': lambda: SELK['sel_tournament_safe'](d_obj, d_sel, np.int32(n),
                                                                       np.int32(3), s32,
                                                                       block=(BS, 1, 1), grid=g),
            'ballot_elite_select_only': lambda: SELK['sel_ballot_elite_safe'](d_fit, d_sel, d_avg, np.int32(n),
                                                                              np.int32(3), np.float64(1.2), s32,
                                                                              block=(BS, 1, 1), grid=g),
            'ballot_elite_full': lambda: fill_selection('ballot_safe', d_obj, d_fit, d_sel,
                                                        d_avg, n, 3, 1.2, 123),
            'ballot_local_full': lambda: fill_selection('ballot_local_safe', d_obj, d_fit, d_sel,
                                                        d_avg, n, 3, 1.2, 123),
            'elitist_host_sync': lambda: fill_selection('tournament_elitist', d_obj, d_fit, d_sel,
                                                        d_avg, n, 3, 1.2, 123),
            'elitist_device': lambda: fill_selection('elitist_dev', d_obj, d_fit, d_sel, d_avg, n, 3,
                                                     1.2, 123),
            'global_best_broadcast': lambda: fill_selection('gbest', d_obj, d_fit, d_sel, d_avg, n, 3, 1.2, 123),
            'argmin_serial': lambda: gpu.k_find_best(d_obj, d_bi, d_bv, np.int32(n),
                                                     block=(1, 1, 1), grid=(1, 1)),
            'argmin_parallel_block': lambda: SELK['argmin_block'](d_obj, np.int32(n), d_bi, d_bv,
                                                                  block=(1024, 1, 1), grid=(1, 1)),
        }
        base = None
        res = {}
        for name, fn in V.items():
            us, reps = time_events(fn, budget_s=CFG['micro_budget_s'])
            res[name] = us
            rows.append({'gpu': GPU_NAME, 'n': n, 'variant': name, 'time_us': us, 'reps': reps})
        base = res['warp_tournament_ts3']
        for r in rows:
            if r['n'] == n:
                r['ratio_to_warp_tournament'] = r['time_us'] / base
        print(f"  n={n:8d}  " + "  ".join(f"{k}={v:.1f}" for k, v in res.items()))
    df = save_csv(rows, 'selection_microbenchmark')
    piv = df.pivot_table(index='variant', columns='n', values='time_us')
    piv.to_csv(os.path.join(RESULTS_DIR, 'selection_microbenchmark_pivot.csv'))
    print(piv.round(2).to_string())


if __name__ == '__main__':
    run()
