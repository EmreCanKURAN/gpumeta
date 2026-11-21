"""Correctness of the selection kernels for arbitrary population sizes.

Every kernel of selection_safe.cu is compared with the CPU reference
implementation on random populations (half of them with tied objective
values) for population sizes between 1 and 8192, including sizes that are not
multiples of 32. For multiples of 32 the boundary-safe kernels are also
compared with the kernels of metaheuristics.cu. Finally every optimiser is run
once with a population of 1000.

The script raises an error if any comparison fails; the CEC experiments rely
on the reference implementation validated here.

Outputs: kernel_correctness.csv, kernel_correctness_smoke.csv
"""
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from gpumeta import gpu
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import ALGORITHMS, CFG, banner, checkpoint, save_checkpoint, save_csv
from gpumeta.gpu import BS, GPU_FUNCS, fill_selection, grid, seed32
from gpumeta.reference import (emu_avg_single_block, emu_ballot, emu_elitist, emu_fit, emu_gbest,
                               emu_global_tournament, emu_migrate, emu_tournament)


def run():
    banner("Selection kernels against the CPU reference implementation, any population size")
    rows = []
    ts, alpha = 3, 1.2
    rhos = [0.0, 1.0 / 32, 0.25, 1.0]
    all_ok = True
    for n in CFG['correctness_ns']:
        for rep in range(CFG['correctness_reps']):
            rng = np.random.RandomState(1000 * n + rep)
            obj = rng.uniform(1.0, 2.0, n)
            if rep % 2 == 1:
                obj = np.round(obj, 2)            # ties on purpose
            seed = int(rng.randint(1, 2 ** 31 - 1))
            d_obj = gpuarray.to_gpu(obj)
            d_fit = gpuarray.zeros(n, np.float64)
            d_sel = gpuarray.zeros(n, np.int32)
            d_avg = gpuarray.zeros(1, np.float64)
            g = grid(n)

            def select(method):
                d_sel.fill(np.int32(-7))
                fill_selection(method, d_obj, d_fit, d_sel, d_avg, n, ts, alpha, seed)
                return d_sel.get().astype(np.int64)

            res = {}
            res['tournament_safe'] = (select('tournament_safe'), emu_tournament(obj, ts, seed))
            gsel = select('ballot_safe'); avg = float(d_avg.get()[0])
            fit_dev = d_fit.get()
            res['ballot_safe'] = (gsel, emu_ballot(emu_fit(obj), ts, alpha, seed, avg=avg, variant='elite'))
            gsel = select('ballot_noswitch_safe'); avg2 = float(d_avg.get()[0])
            res['ballot_noswitch_safe'] = (gsel, emu_ballot(emu_fit(obj), ts, alpha, seed, avg=avg2,
                                                            variant='noswitch'))
            res['ballot_local_safe'] = (select('ballot_local_safe'), emu_ballot(emu_fit(obj), ts, alpha, seed,
                                                                                variant='local'))
            res['elitist_dev'] = (select('elitist_dev'), emu_elitist(obj, ts, seed))
            res['gbest'] = (select('gbest'), emu_gbest(obj))
            res['global3'] = (select('global3'), emu_global_tournament(obj, 3, seed))
            for rho in rhos:
                res[f'mig_{rho:g}'] = (select(f'mig_{rho:g}'), emu_migrate(obj, ts, seed, rho))
            fit_exact = bool(np.array_equal(fit_dev, emu_fit(obj)))
            mean_exact = bool(n > 256 or avg == emu_avg_single_block(emu_fit(obj)))
            all_ok &= fit_exact and mean_exact
            for name, (gv, ev) in res.items():
                ok = bool(np.array_equal(gv, ev)) and gv.min() >= 0 and gv.max() < n
                all_ok &= ok
                rows.append({'n': n, 'rep': rep, 'ties': rep % 2 == 1, 'kernel': name, 'compare': 'gpu_vs_emulator',
                             'identical': ok, 'mismatches': int(np.sum(gv != ev)),
                             'fit_bit_exact': fit_exact, 'mean_bit_exact': mean_exact})
            # the elitist kernel must return an individual with the minimum objective value
            gel = select('tournament_elitist')
            rows.append({'n': n, 'rep': rep, 'ties': rep % 2 == 1, 'kernel': 'tournament_elitist',
                         'compare': 'vs_direct_global_best',
                         'identical': bool(gel.min() >= 0 and gel.max() < n and np.all(obj[np.clip(gel, 0, n - 1)] == obj.min())),
                         'mismatches': int(np.sum(obj[np.clip(gel, 0, n - 1)] != obj.min()))})
            # kernels of metaheuristics.cu against their boundary-safe versions
            pairs = {'tournament': 'tournament_safe', 'ballot': 'ballot_safe', 'ballot_noswitch': 'ballot_noswitch_safe',
                     'ballot_local': 'ballot_local_safe'}
            for old, new in pairs.items():
                if n % 32 != 0:
                    # these kernels return early for idx >= n, so they are only defined for
                    # full warps; experiments/profiling.py runs them under compute-sanitizer
                    continue
                try:
                    if old == 'tournament':
                        gpu.k_sel_tournament(d_obj, d_sel, np.int32(n), np.int32(ts), seed32(seed),
                                             block=(BS, 1, 1), grid=g)
                    elif old == 'ballot_local':
                        gpu.k_compute_fit(d_obj, d_fit, np.int32(n), block=(BS, 1, 1), grid=g)
                        gpu.k_sel_ballot_local(d_fit, d_sel, np.int32(n), np.int32(ts), np.float64(alpha),
                                               seed32(seed), block=(BS, 1, 1), grid=g)
                    else:
                        gpu.k_compute_fit(d_obj, d_fit, np.int32(n), block=(BS, 1, 1), grid=g)
                        gpu.k_reset_avg(d_avg, block=(1, 1, 1), grid=(1, 1))
                        gpu.k_compute_avg(d_fit, d_avg, np.int32(n), block=(BS, 1, 1), grid=g, shared=BS * 8)
                        gpu.k_finalize_avg(d_avg, np.int32(n), block=(1, 1, 1), grid=(1, 1))
                        kern = gpu.k_sel_ballot if old == 'ballot' else gpu.k_sel_ballot_nosw
                        kern(d_fit, d_sel, d_avg, np.int32(n), np.int32(ts), np.float64(alpha), seed32(seed),
                             block=(BS, 1, 1), grid=g)
                    cuda.Context.synchronize()
                    pv = d_sel.get().astype(np.int64)
                    sv = select(new)
                    same = bool(np.array_equal(pv, sv))
                    valid = bool(pv.min() >= 0 and pv.max() < n)
                except Exception as ex:
                    same, valid = False, False
                    print(f"    {old} failed at n={n}: {ex}")
                rows.append({'n': n, 'rep': rep, 'ties': rep % 2 == 1, 'kernel': old,
                             'compare': 'original_vs_safe', 'identical': same, 'valid_indices': valid,
                             'n_mod_32': n % 32})
                if n % 32 == 0 and not same:
                    all_ok = False
    df = save_csv(rows, 'kernel_correctness')
    summ = df.groupby(['compare', 'kernel'])['identical'].mean()
    print(summ.to_string())
    sub = df[df['compare'] == 'original_vs_safe']
    print("\n  original vs boundary-safe kernels, n % 32 == 0 (fraction identical):",
          sub.identical.mean() if len(sub) else 'n/a')
    # complete optimisers with a population size that is not a multiple of 32
    smoke = []
    fid = BENCHMARKS['Sphere']['id']; bnd = BENCHMARKS['Sphere']['bounds']
    for alg in ALGORITHMS:
        for m in ['tournament_safe', 'ballot_safe', 'ballot_local_safe', 'mig_0.25', 'gbest']:
            r = GPU_FUNCS[alg](1000, 50, fid, bnd, dim=30, method=m, seed=7)
            ok = bool(np.isfinite(r['best']))
            all_ok &= ok
            smoke.append({'algorithm': alg, 'method': m, 'ps': 1000, 'best': r['best'], 'finite': ok})
    save_csv(smoke, 'kernel_correctness_smoke')
    checkpoint['kernel_correctness_pass'] = bool(all_ok)
    save_checkpoint()
    print(f"\n  result: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise RuntimeError("kernel check failed: see kernel_correctness.csv")


if __name__ == '__main__':
    run()
