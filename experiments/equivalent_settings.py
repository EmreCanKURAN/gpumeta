"""End-to-end runtime under equivalent algorithmic settings: all kernels of an
algorithm stay the same and only the selection operator is exchanged. The
warp mechanisms are compared with naive selection, a roulette wheel with a
parallel prefix sum, global tournaments and direct global-best selection.

Outputs: equivalent_settings_runs.csv, equivalent_settings_summary.csv
"""
from gpumeta import gpu
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import ALGORITHMS, CFG, banner, checkpoint, make_seeds, save_checkpoint, save_csv
from gpumeta.gpu import GPU_FUNCS

METHODS_GUIDE = ['naive', 'tournament', 'global3', 'global4', 'gbest', 'tournament_elitist', 'elitist_dev',
                 'ballot']
METHODS_ABC = ['naive', 'roulette_scan', 'tournament', 'global3', 'global4', 'gbest', 'ballot']
BASELINES = ['naive', 'roulette_scan', 'global3', 'global4', 'gbest']


def run():
    banner("End-to-end runtime with only the selection operator exchanged")
    gpu.warmup()
    ck = checkpoint.setdefault('equivalent_settings', {})
    seeds = make_seeds(CFG['equiv_seeds'], offset=900)
    rows = []
    for ps in CFG['equiv_pops']:
        for fname in CFG['equiv_funcs']:
            fid = BENCHMARKS[fname]['id']; bnd = BENCHMARKS[fname]['bounds']
            for alg in ALGORITHMS:
                key = f"{alg}_{fname}_{ps}"
                if key not in ck:
                    print(f"  {alg} | {fname} | ps={ps}", flush=True)
                    res = {}
                    for m in (METHODS_ABC if alg == 'ABC' else METHODS_GUIDE):
                        res[m] = [GPU_FUNCS[alg](ps, CFG['equiv_iters'], fid, bnd, dim=CFG['equiv_dim'],
                                                 method=m, seed=sd) for sd in seeds]
                        res[m] = [{'best': r['best'], 'time_ms': r['time_ms']} for r in res[m]]
                    ck[key] = res
                    save_checkpoint()
                for m, runs in ck[key].items():
                    for i, r in enumerate(runs):
                        rows.append({'algorithm': alg, 'function': fname, 'ps': ps, 'method': m, 'run': i,
                                     'best': r['best'], 'time_ms': r['time_ms']})
    df = save_csv(rows, 'equivalent_settings_runs')
    agg = df.groupby(['algorithm', 'ps', 'method'])['time_ms'].mean().reset_index()
    out = []
    for (alg, ps), sub in agg.groupby(['algorithm', 'ps']):
        t = dict(zip(sub.method, sub.time_ms))
        bases = {b: t[b] for b in BASELINES if b in t}
        best_base = min(bases, key=bases.get)
        for m in t:
            out.append({'algorithm': alg, 'ps': ps, 'method': m, 'mean_time_ms': t[m],
                        'speedup_vs_naive': t['naive'] / t[m],
                        'best_gpu_baseline': best_base, 'speedup_vs_best_gpu_baseline': bases[best_base] / t[m]})
    df2 = save_csv(out, 'equivalent_settings_summary')
    print(df2.pivot_table(index=['algorithm', 'ps'], columns='method', values='mean_time_ms').round(1).to_string())


if __name__ == '__main__':
    run()
