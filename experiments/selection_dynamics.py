"""Characterisation of the selection operators as operators, without an
optimiser around them.

run_dynamics          takeover time of a super individual, population diversity
                      of GWO and ABC, selection intensity of the warp and the
                      global tournament
run_global_best       identity of the elitist rule and direct global-best
                      selection, GWO leader statistics, takeover as a function
                      of the migration rate rho
run_matched_pressure  selection intensity of the warp tournament and the global
                      tournament at matched numbers of contestants

Outputs: takeover.csv, diversity.csv, selection_pressure.csv,
selection_dynamics.png, elitist_vs_global_best.csv, gwo_leader_statistics.csv,
takeover_migration.csv, matched_selection_pressure.csv
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycuda.gpuarray as gpuarray
from scipy import stats as sps

from gpumeta import gpu
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import (CFG, MASTER_SEED, RESULTS_DIR, STATS_DIM, STATS_ITERS, STATS_METHODS, STATS_POP,
                            banner, checkpoint, save_checkpoint, save_csv)
from gpumeta.gpu import BS, GPU_FUNCS, fill_selection, grid, run_gwo_gpu, use_kernels
from gpumeta.reference import emu_offsets


def takeover_experiment(method, ps=1024, ts=3, alpha=1.2, max_steps=300, seed=0):
    """Selection-only reproduction with one super individual.
    Warp-restricted mechanisms cannot propagate a solution across warp
    boundaries, so the following quantities are tracked:
      - global takeover step (>= 99% of population; capped at max_steps),
      - within-warp takeover step (>= 99% of the super individual's warp),
      - plateau: final proportion of super copies (expected ~32/ps for warp
        methods, 1.0 for global methods),
      - extinction: super individual lost by drift before takeover.
    """
    rng = np.random.RandomState(seed)
    obj = rng.uniform(1.0, 2.0, ps)
    si = rng.randint(ps)
    obj[si] = 0.0
    ws, we = (si // 32) * 32, (si // 32) * 32 + 32
    d_obj = gpuarray.to_gpu(obj.astype(np.float64))
    d_obj2 = gpuarray.zeros(ps, np.float64)
    d_fit = gpuarray.zeros(ps, np.float64)
    d_sel = gpuarray.zeros(ps, np.int32)
    d_avg = gpuarray.zeros(1, np.float64)
    curve = []
    takeover_global = max_steps
    takeover_warp = max_steps
    extinct = False
    prop = 1.0 / ps
    for step in range(max_steps):
        fill_selection(method, d_obj, d_fit, d_sel, d_avg, ps, ts, alpha, seed * 7919 + step)
        gpu.k_gather_obj(d_obj, d_obj2, d_sel, np.int32(ps), block=(BS, 1, 1), grid=grid(ps))
        d_obj, d_obj2 = d_obj2, d_obj
        o = d_obj.get()
        prop = float(np.mean(o < 0.5))
        prop_warp = float(np.mean(o[ws:we] < 0.5))
        curve.append(prop)
        if takeover_warp == max_steps and prop_warp >= 0.99:
            takeover_warp = step + 1
        if prop == 0.0:
            extinct = True
            break
        if prop >= 0.99:
            takeover_global = step + 1
            break
    return {'takeover_global': takeover_global, 'takeover_warp': takeover_warp,
            'plateau': prop, 'extinct': extinct, 'curve': curve}


def selection_pressure_experiment(ps=1024, ts=3, reps=200, seed=12345):
    """Distribution of selected objective values: warp vs global tournament.
    Reports the selection intensity and a two-sample KS test."""
    rng = np.random.RandomState(seed)
    obj = rng.uniform(0.0, 1.0, ps)
    d_obj = gpuarray.to_gpu(obj.astype(np.float64))
    d_fit = gpuarray.zeros(ps, np.float64)
    d_sel = gpuarray.zeros(ps, np.int32)
    d_avg = gpuarray.zeros(1, np.float64)
    selected = {}
    for m in ['tournament', 'tournament_global']:
        vals = []
        for r in range(reps):
            fill_selection(m, d_obj, d_fit, d_sel, d_avg, ps, ts, 1.2, seed + 31 * r)
            vals.append(obj[d_sel.get()])
        selected[m] = np.concatenate(vals)
    sd = obj.std()
    out = {}
    for m, v in selected.items():
        # minimisation: the intensity is measured on the negated objective
        out[m] = {'sel_mean_obj': float(v.mean()),
                  'selection_intensity': float(((-v).mean() - (-obj).mean()) / sd)}
    ks_stat, ks_p = sps.ks_2samp(selected['tournament'], selected['tournament_global'])
    return out, float(ks_stat), float(ks_p), selected


def run_dynamics():
    tag = 'selection_dynamics'
    banner("Takeover time, population diversity and selection intensity")
    ck = checkpoint.setdefault(tag, {})
    methods = ['tournament', 'tournament_elitist', 'tournament_global', 'ballot', 'ballot_local', 'ballot_noswitch']
    # (a) takeover time (global + within-warp + plateau + extinction)
    if 'takeover' not in ck:
        tk_rows, tk_curves = [], {}
        for m in methods:
            res = [takeover_experiment(m, ps=1024, seed=500 + rep) for rep in range(20)]
            alive = [r for r in res if not r['extinct']]
            tk_curves[m] = res[0]['curve']
            tk_rows.append({
                'method': m,
                'takeover_global_mean': float(np.mean([r['takeover_global'] for r in alive])) if alive else float('nan'),
                'takeover_global_std': float(np.std([r['takeover_global'] for r in alive])) if alive else float('nan'),
                'takeover_warp_mean': float(np.mean([r['takeover_warp'] for r in alive])) if alive else float('nan'),
                'takeover_warp_std': float(np.std([r['takeover_warp'] for r in alive])) if alive else float('nan'),
                'plateau_mean': float(np.mean([r['plateau'] for r in alive])) if alive else 0.0,
                'extinct_runs': int(sum(r['extinct'] for r in res)),
                'n': len(res), 'max_steps_cap': 300})
            r0 = tk_rows[-1]
            print(f"  takeover[{m}]: global={r0['takeover_global_mean']:.1f} (cap=300), "
                  f"warp={r0['takeover_warp_mean']:.1f}, plateau={r0['plateau_mean']:.3f}, "
                  f"extinct={r0['extinct_runs']}/20")
        ck['takeover'] = {'rows': tk_rows, 'curves': tk_curves}
        save_checkpoint()
    save_csv(ck['takeover']['rows'], 'takeover')
    # (b) diversity over time (GWO + ABC on Rastrigin)
    if 'diversity' not in ck:
        div_rows = {}
        fid = BENCHMARKS['Rastrigin']['id']; bnd = BENCHMARKS['Rastrigin']['bounds']
        for alg in ['GWO', 'ABC']:
            for m in STATS_METHODS:
                print(f"  diversity: {alg}/{m}", flush=True)
                r = GPU_FUNCS[alg](STATS_POP, STATS_ITERS, fid, bnd, dim=STATS_DIM,
                                   method=m, seed=MASTER_SEED, track_diversity=True)
                div_rows[f"{alg}_{m}"] = r['diversity']
        ck['diversity'] = div_rows
        save_checkpoint()
    drows = []
    for k, series in ck['diversity'].items():
        alg, m = k.split('_', 1)
        for it, v in series:
            drows.append({'algorithm': alg, 'method': m, 'iter': it, 'diversity': v})
    save_csv(drows, 'diversity')
    # (c) selection intensity of the warp and the global tournament
    if 'pressure' not in ck:
        out, ks_stat, ks_p, _ = selection_pressure_experiment()
        prows = [{'method': m, **v} for m, v in out.items()]
        for r in prows:
            r['ks_stat'] = ks_stat; r['ks_p'] = ks_p
        ck['pressure'] = prows
        save_checkpoint()
        print(f"  KS(warp vs global tournament): D={ks_stat:.4f}, p={ks_p:.4g}")
    save_csv(ck['pressure'], 'selection_pressure')
    # plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for m, curve in ck['takeover']['curves'].items():
            axes[0].plot(curve, label=m)
        axes[0].set_xlabel('selection step'); axes[0].set_ylabel('proportion of best individual')
        axes[0].set_title('takeover dynamics'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        for k, series in ck['diversity'].items():
            if k.startswith('GWO'):
                its = [x[0] for x in series]; vs = [x[1] for x in series]
                axes[1].plot(its, vs, label=k)
        axes[1].set_xlabel('iteration'); axes[1].set_ylabel('normalized diversity')
        axes[1].set_title('GWO population diversity'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig.savefig(os.path.join(RESULTS_DIR, f"{tag}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped: {e}")


def run_global_best():
    banner("Elitist rule and global best, GWO leader statistics, takeover under migration")
    # (a) the elitist rule selects the global best for every individual
    rows = []
    for n in [96, 1024, 8192]:
        same = 0; total = 0
        for rep in range(10 if CFG['quick'] else 200):
            rng = np.random.RandomState(rep + n)
            obj = rng.uniform(0, 1, n)
            d_obj = gpuarray.to_gpu(obj); d_fit = gpuarray.zeros(n, np.float64)
            d_sel = gpuarray.zeros(n, np.int32); d_avg = gpuarray.zeros(1, np.float64)
            fill_selection('tournament_elitist', d_obj, d_fit, d_sel, d_avg, n, 3, 1.2, rep)
            a = d_sel.get()
            fill_selection('gbest', d_obj, d_fit, d_sel, d_avg, n, 3, 1.2, rep)
            b = d_sel.get()
            same += int(np.array_equal(a, b)); total += 1
        rows.append({'n': n, 'populations': total, 'identical_selection_vectors': same,
                     'fraction': same / total})
    save_csv(rows, 'elitist_vs_global_best')
    print(pd.DataFrame(rows).to_string(index=False))

    # (b) leaders extracted by GWO from the selected multiset
    stats_rows = []
    rec = {'cur': None}
    orig_sel = gpu.k_gwo_leaders_sel
    orig_nai = gpu.k_gwo_leaders

    def w_sel(d_obj, d_sel, d_leaders, n, block=None, grid=None):
        orig_sel(d_obj, d_sel, d_leaders, n, block=block, grid=grid)
        L = d_leaders.get(); o = d_obj.get()
        top3 = set(np.argsort(o, kind='stable')[:3].tolist())
        rec['cur'].append((len(set(L.tolist())), int(L[0] == int(np.argmin(o))), int(set(L.tolist()) == top3)))

    def w_nai(d_obj, d_leaders, n, block=None, grid=None):
        orig_nai(d_obj, d_leaders, n, block=block, grid=grid)
        L = d_leaders.get()
        rec['cur'].append((len(set(L.tolist())), 1, 1))

    fid = BENCHMARKS['Rastrigin']['id']; bnd = BENCHMARKS['Rastrigin']['bounds']
    methods = ['naive', 'tournament', 'tournament_elitist', 'gbest', 'mig_0.03125', 'mig_0.25', 'ballot']
    with use_kernels({'k_gwo_leaders_sel': w_sel, 'k_gwo_leaders': w_nai}):
        for m in methods:
            for sd in range(CFG['leader_seeds']):
                rec['cur'] = []
                run_gwo_gpu(1024, CFG['leader_iters'], fid, bnd, dim=100, method=m, seed=100 + sd)
                arr = np.array(rec['cur'])
                stats_rows.append({'method': m, 'seed': sd, 'iterations': len(arr),
                                   'mean_distinct_leaders': float(arr[:, 0].mean()),
                                   'frac_iters_3_distinct': float(np.mean(arr[:, 0] == 3)),
                                   'frac_iters_1_distinct': float(np.mean(arr[:, 0] == 1)),
                                   'frac_alpha_is_global_best': float(arr[:, 1].mean()),
                                   'frac_leaders_equal_naive_top3': float(arr[:, 2].mean())})
    df = save_csv(stats_rows, 'gwo_leader_statistics')
    print(df.groupby('method').mean(numeric_only=True).round(4).to_string())

    # (c) takeover under selection only, as a function of the migration rate
    tk_rows = []
    for m in ['tournament', 'mig_0.03125', 'mig_0.1', 'mig_0.25', 'mig_0.5', 'gbest', 'tournament_elitist',
              'global3', 'global4']:
        res = [takeover_experiment(m, ps=1024, seed=500 + rep) for rep in range(CFG['takeover_reps'])]
        alive = [r for r in res if not r['extinct']]
        tk_rows.append({'method': m,
                        'takeover_global_mean': float(np.mean([r['takeover_global'] for r in alive])) if alive else float('nan'),
                        'takeover_global_std': float(np.std([r['takeover_global'] for r in alive])) if alive else float('nan'),
                        'takeover_warp_mean': float(np.mean([r['takeover_warp'] for r in alive])) if alive else float('nan'),
                        'plateau_mean': float(np.mean([r['plateau'] for r in alive])) if alive else 0.0,
                        'extinct_runs': int(sum(r['extinct'] for r in res)), 'n': len(res), 'cap': 300})
    df = save_csv(tk_rows, 'takeover_migration')
    print(df.to_string(index=False))


def run_matched_pressure():
    banner("Selection intensity at matched numbers of contestants")
    n = 1024
    rows = []
    d_sel = gpuarray.zeros(n, np.int32); d_fit = gpuarray.zeros(n, np.float64); d_avg = gpuarray.zeros(1, np.float64)
    for dist in ['uniform', 'normal']:
        rng = np.random.RandomState(12345)
        obj = rng.uniform(0, 1, n) if dist == 'uniform' else rng.normal(0, 1, n)
        d_obj = gpuarray.to_gpu(obj)
        sd = obj.std()
        configs = [('warp', w, 'tournament_safe', w) for w in [1, 2, 3, 4, 5]] + \
                  [('global', k, f'global{k}', 3) for k in [2, 3, 4, 5, 6]]
        samples = {}
        for kind, size, method, tsarg in configs:
            vals, lod, eff = [], [], []
            for r in range(CFG['pressure_reps']):
                seed = 12345 + 31 * r
                fill_selection(method, d_obj, d_fit, d_sel, d_avg, n, tsarg, 1.2, seed)
                s = d_sel.get()
                vals.append(obj[s]); lod.append(1.0 - len(np.unique(s)) / n)
                if kind == 'warp':
                    offs = np.stack(emu_offsets(n, seed, size))
                    eff.append(np.mean([1 + len(set(offs[:, i].tolist())) for i in range(0, n, 7)]))
            v = np.concatenate(vals)
            samples[(kind, size)] = v
            k_eff = float(np.mean(eff)) if eff else float(size)
            nominal = size + 1 if kind == 'warp' else size
            theo = (0.5 - 1.0 / (nominal + 1)) * math.sqrt(12.0) if dist == 'uniform' else float('nan')
            rows.append({'distribution': dist, 'mechanism': kind, 'ts_parameter': size,
                         'nominal_contestants': nominal, 'mean_distinct_contestants': k_eff,
                         'selection_intensity': float((obj.mean() - v.mean()) / sd),
                         'theoretical_intensity_uniform': theo,
                         'selection_variance_ratio': float(v.var() / obj.var()),
                         'loss_of_diversity': float(np.mean(lod))})
        for w in [1, 2, 3, 4, 5]:
            k = w + 1
            if ('global', k) in samples:
                D, p = sps.ks_2samp(samples[('warp', w)], samples[('global', k)])
                rows.append({'distribution': dist, 'mechanism': f'KS warp ts={w} vs global ts={k}',
                             'ts_parameter': w, 'nominal_contestants': k, 'ks_D': float(D), 'ks_p': float(p)})
    df = save_csv(rows, 'matched_selection_pressure')
    print(df.round(4).to_string(index=False))


def run():
    run_dynamics()
    run_global_best()
    run_matched_pressure()


if __name__ == '__main__':
    run()
