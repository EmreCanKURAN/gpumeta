"""Hyperparameter optimisation of an RBF SVM and of XGBoost with a host-loop GWO
whose leaders are chosen by the GPU selection kernels, compared with random
search under the same evaluation budget.

Nested protocol: each run draws a stratified 75/25 split; the optimiser
minimises the three-fold cross-validation error on the training part, and the
tuned model is then fitted on the training part and scored on the held-out
part. All methods of a run share the split, the folds and the optimiser seed.

Outputs: ml_search_space.csv, ml_runs.csv, ml_statistics.csv, ml_summary.csv
"""
import itertools
import os
import time

import numpy as np
import pycuda.gpuarray as gpuarray
from joblib import Parallel, delayed
from scipy import stats as sps
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedShuffleSplit

from gpumeta.config import CFG, MASTER_SEED, N_WORKERS, RESULTS_DIR, banner, checkpoint, save_checkpoint, save_csv
from gpumeta.gpu import fill_selection
from gpumeta.ml import SEARCH_SPACE, build_model, cv_error, xgb
from gpumeta.reference import leaders_from_selection
from gpumeta.stats import a12, holm, wilcoxon_p


METHODS = ['random', 'naive', 'tournament', 'elitist_dev', 'mig_0.25', 'ballot']


def gwo_host(eval_batch, dim, ps, iters, seed, method):
    rng = np.random.RandomState(seed)
    pos = rng.rand(ps, dim)
    obj = eval_batch(pos); evals = ps
    d_obj = gpuarray.zeros(ps, np.float64); d_fit = gpuarray.zeros(ps, np.float64)
    d_sel = gpuarray.zeros(ps, np.int32); d_avg = gpuarray.zeros(1, np.float64)
    bi = int(np.argmin(obj)); best_obj, best_pos = float(obj[bi]), pos[bi].copy()
    nlead = []
    for it in range(iters):
        if method == 'naive':
            ld = np.argsort(obj)[:3]; nlead.append(3)
        else:
            d_obj.set(obj.astype(np.float64))
            fill_selection(method, d_obj, d_fit, d_sel, d_avg, ps, 3, 1.2, seed * 131 + it)
            ld, nd = leaders_from_selection(d_sel.get(), obj); nlead.append(nd)
        a_p, b_p, dl_p = pos[ld[0]], pos[ld[1]], pos[ld[2]]
        a = 2.0 - 2.0 * it / iters
        r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
        X1 = a_p - (2 * a * r1 - a) * np.abs(2 * r2 * a_p - pos)
        r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
        X2 = b_p - (2 * a * r1 - a) * np.abs(2 * r2 * b_p - pos)
        r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
        X3 = dl_p - (2 * a * r1 - a) * np.abs(2 * r2 * dl_p - pos)
        pos = np.clip((X1 + X2 + X3) / 3.0, 0, 1)
        obj = eval_batch(pos); evals += ps
        bi = int(np.argmin(obj))
        if obj[bi] < best_obj:
            best_obj, best_pos = float(obj[bi]), pos[bi].copy()
    return best_obj, best_pos, evals, float(np.mean(nlead))


def run():
    banner("Hyperparameter optimisation with nested cross-validation")
    space_rows = [{'model': k, 'hyperparameter': a, 'scale': b, 'low': c, 'high': d}
                  for k, v in SEARCH_SPACE.items() for a, b, c, d in v]
    save_csv(space_rows, 'ml_search_space')
    datasets = {'breast_cancer': load_breast_cancer(), 'digits': load_digits()}
    kinds = ['SVM_RBF'] + (['XGBoost'] if xgb is not None else [])
    ck = checkpoint.setdefault('ml', {})
    rows = []
    with Parallel(n_jobs=N_WORKERS, backend='loky') as par:
        for dname, data in datasets.items():
            X, y = data.data.astype(np.float64), data.target
            ps, iters = CFG['ml_budget'][dname]
            for kind in kinds:
                dim = 2 if kind == 'SVM_RBF' else 8
                runs = CFG['ml_runs_svm'] if kind == 'SVM_RBF' else CFG['ml_runs_xgb']
                task = f"{kind}_{dname}"
                for r in range(runs):
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1000 + r)
                    tr, te = next(sss.split(X, y))
                    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
                    iseed = 2000 + r
                    oseed = MASTER_SEED + 17 * r

                    def eval_batch(P):
                        return np.array(par(delayed(cv_error)(kind, p, Xtr, ytr, iseed) for p in P))

                    for m in METHODS:
                        key = f"{task}_{m}_{r}"
                        if key not in ck:
                            t0 = time.time()
                            if m == 'random':
                                rng = np.random.RandomState(oseed)
                                bo, bp, ev = np.inf, None, 0
                                for _ in range(iters + 1):
                                    V = rng.rand(ps, dim); o = eval_batch(V); ev += ps
                                    i = int(np.argmin(o))
                                    if o[i] < bo:
                                        bo, bp = float(o[i]), V[i].copy()
                                nl = float('nan')
                            else:
                                bo, bp, ev, nl = gwo_host(eval_batch, dim, ps, iters, oseed, m)
                            clf = build_model(kind, bp).fit(Xtr, ytr)
                            test_acc = float((clf.predict(Xte) == yte).mean())
                            ck[key] = {'cv_acc': 1.0 - bo, 'test_acc': test_acc, 'evals': ev,
                                       'time_s': time.time() - t0, 'best_vec': bp.tolist(), 'n_leaders': nl}
                            save_checkpoint()
                            print(f"    {task:24s} run {r:2d} {m:12s} CV={1 - bo:.4f} test={test_acc:.4f} "
                                  f"({ck[key]['time_s']:.0f}s)", flush=True)
                        d = ck[key]
                        rows.append({'task': task, 'run': r, 'method': m, 'cv_acc_inner': d['cv_acc'],
                                     'test_acc_outer': d['test_acc'], 'evals': d['evals'], 'time_s': d['time_s'],
                                     'mean_distinct_leaders': d['n_leaders'],
                                     'split_seed': 1000 + r, 'inner_cv_seed': 2000 + r, 'optimizer_seed': oseed,
                                     'ps': ps, 'iters': iters})
    df = save_csv(rows, 'ml_runs')
    st = []
    for task, sub in df.groupby('task'):
        for metric in ['cv_acc_inner', 'test_acc_outer']:
            W = sub.pivot_table(index='run', columns='method', values=metric)
            W = W[[m for m in METHODS if m in W.columns]].dropna()
            try:
                chi2, pf = sps.friedmanchisquare(*[W[c].values for c in W.columns])
            except Exception:
                chi2, pf = float('nan'), float('nan')
            pairs = [(a, b) for a, b in itertools.combinations(W.columns, 2)]
            ps_ = [wilcoxon_p(W[a].values, W[b].values) for a, b in pairs]
            adj = holm(ps_)
            for (a, b), p, pa in zip(pairs, ps_, adj):
                st.append({'task': task, 'metric': metric, 'm1': a, 'm2': b, 'n_runs': len(W),
                           'mean_m1': W[a].mean(), 'mean_m2': W[b].mean(), 'std_m1': W[a].std(ddof=1),
                           'std_m2': W[b].std(ddof=1), 'A12_m1_higher': 1 - a12(W[a].values, W[b].values),
                           'p_wilcoxon': p, 'p_holm': pa, 'friedman_chi2': chi2, 'friedman_p': pf})
    save_csv(st, 'ml_statistics')
    summ = df.groupby(['task', 'method'])[['cv_acc_inner', 'test_acc_outer', 'time_s']].agg(['mean', 'std'])
    summ.to_csv(os.path.join(RESULTS_DIR, 'ml_summary.csv'))
    print(summ.round(4).to_string())


if __name__ == '__main__':
    run()
