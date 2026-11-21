"""Statistical comparison of selection mechanisms: Wilcoxon signed-rank and
Mann-Whitney tests with Holm correction, TOST equivalence, Vargha-Delaney A12,
rank-biserial correlation, Friedman ranks and the Nemenyi critical difference."""
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import EPS_TOL, TOST_MARGIN, save_csv


def summarize_runs(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'median': float(np.median(a)), 'min': float(np.min(a)), 'max': float(np.max(a))}


NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164}


def floor_log10(x, eps=EPS_TOL):
    x = np.asarray(x, np.float64)
    return np.log10(np.maximum(x, eps))


def a12(x, y):
    """Vargha-Delaney A for MINIMISATION: P(x < y) + 0.5 P(x == y). >0.5 -> x better."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    gt = (x[:, None] < y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def rank_biserial(d):
    d = np.asarray(d, np.float64)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def wilcoxon_p(x, y):
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    if np.all(d == 0):
        return 1.0
    try:
        return float(sps.wilcoxon(x, y).pvalue)
    except Exception:
        return 1.0


def tost_p(d, margin=TOST_MARGIN):
    """Paired TOST on differences d (log10 errors). p < 0.05 -> equivalent within +-margin."""
    d = np.asarray(d, np.float64)
    if np.all(np.abs(d) < 1e-15):
        return 0.0
    try:
        p1 = sps.wilcoxon(d + margin, alternative='greater').pvalue
        p2 = sps.wilcoxon(d - margin, alternative='less').pvalue
        return float(max(p1, p2))
    except Exception:
        return 1.0


def holm(pvals):
    p = np.asarray(pvals, np.float64)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def stable_mean_std(x):
    x = np.asarray(x, np.float64)
    s = np.max(np.abs(x)) if len(x) else 0.0
    if s == 0 or not np.isfinite(s):
        return float(np.mean(x)) if len(x) else float('nan'), 0.0
    y = x / s
    return float(s * y.mean()), float(s * y.std(ddof=1)) if len(x) > 1 else 0.0


# The random number generator of Numba is not controlled by np.random.seed,
# so CPU runs are independent samples and are compared with an unpaired test.
UNPAIRED = {'cpu_serial'}


def mechanism_analysis(data, tag, methods, ref_method='naive'):
    """data[(alg, func)][method] -> list of ERRORS (to optimum), seed-paired.
    Writes pairwise / per-algorithm summary / Friedman+Nemenyi tables."""
    pair_rows, sum_rows, fr_rows, desc_rows = [], [], [], []
    algs = sorted({a for a, _ in data})
    for alg in algs:
        funcs = [f for a, f in data if a == alg]
        ms = [m for m in methods if all(m in data[(alg, f)] for f in funcs)]
        if len(ms) < 2:
            continue
        # descriptive
        for f in funcs:
            for m in ms:
                v = np.asarray(data[(alg, f)][m], np.float64)
                lv = floor_log10(v)
                mu, sd = stable_mean_std(v)
                desc_rows.append({'algorithm': alg, 'function': f, 'method': m, 'n': len(v),
                                  'median': float(np.median(v)), 'q1': float(np.percentile(v, 25)),
                                  'q3': float(np.percentile(v, 75)), 'mean_stable': mu, 'std_stable': sd,
                                  'median_log10_floor': float(np.median(lv)),
                                  'iqr_log10_floor': float(np.percentile(lv, 75) - np.percentile(lv, 25)),
                                  'success_rate': float(np.mean(v <= EPS_TOL))})
        # pairwise
        rows_alg = []
        for m1, m2 in itertools.combinations(ms, 2):
            for f in funcs:
                x = np.asarray(data[(alg, f)][m1], np.float64)
                y = np.asarray(data[(alg, f)][m2], np.float64)
                n = min(len(x), len(y)); x, y = x[:n], y[:n]
                lx, ly = floor_log10(x), floor_log10(y)
                d = lx - ly
                unpaired = (m1 in UNPAIRED) or (m2 in UNPAIRED)
                if unpaired:
                    try:
                        p_w = float(sps.mannwhitneyu(lx, ly, alternative='two-sided').pvalue) \
                            if not np.all(lx == ly) else 1.0
                    except Exception:
                        p_w = 1.0
                else:
                    p_w = wilcoxon_p(lx, ly)
                rows_alg.append({'algorithm': alg, 'function': f, 'm1': m1, 'm2': m2, 'n': n,
                                 'test': 'mann-whitney' if unpaired else 'wilcoxon-signed-rank',
                                 'median_m1': float(np.median(x)), 'median_m2': float(np.median(y)),
                                 'median_log10_diff': float(np.median(ly) - np.median(lx)),
                                 'better': m1 if np.median(lx) < np.median(ly) else (
                                     m2 if np.median(ly) < np.median(lx) else 'tie'),
                                 'A12_m1_better': a12(lx, ly),
                                 'cliffs_delta': 2 * a12(lx, ly) - 1,
                                 'rank_biserial': float('nan') if unpaired else -rank_biserial(d),
                                 'p_wilcoxon': p_w,
                                 'p_tost': 1.0 if unpaired else tost_p(d)})
        # Holm family = one mechanism pair across all functions of this algorithm
        for m1, m2 in itertools.combinations(ms, 2):
            fam = [r for r in rows_alg if r['m1'] == m1 and r['m2'] == m2]
            adj = holm([r['p_wilcoxon'] for r in fam])
            adj_t = holm([r['p_tost'] for r in fam])
            for r, a, at in zip(fam, adj, adj_t):
                r['p_holm'] = float(a)
                r['p_tost_holm'] = float(at)
                r['significant_holm'] = bool(a < 0.05)
                r['equivalent_holm'] = bool(at < 0.05)
        pair_rows.extend(rows_alg)
        for m1, m2 in itertools.combinations(ms, 2):
            sub = [r for r in rows_alg if r['m1'] == m1 and r['m2'] == m2]
            nf = len(sub)
            sum_rows.append({'algorithm': alg, 'm1': m1, 'm2': m2, 'n_functions': nf,
                             'm1_sig_better': sum(r['significant_holm'] and r['better'] == m1 for r in sub),
                             'm2_sig_better': sum(r['significant_holm'] and r['better'] == m2 for r in sub),
                             'not_significant': sum(not r['significant_holm'] for r in sub),
                             'equivalent_tost': sum(r['equivalent_holm'] for r in sub),
                             'median_of_median_log10_diff': float(np.median([r['median_log10_diff'] for r in sub])),
                             'median_A12_m1_better': float(np.median([r['A12_m1_better'] for r in sub]))})
        # Friedman across mechanisms, blocks = functions (median log error)
        M = np.array([[np.median(floor_log10(data[(alg, f)][m])) for m in ms] for f in funcs])
        ranks = np.apply_along_axis(sps.rankdata, 1, M)
        mean_ranks = ranks.mean(axis=0)
        try:
            chi2, pf = sps.friedmanchisquare(*[M[:, j] for j in range(len(ms))]) if len(ms) >= 3 else (
                float('nan'), float('nan'))
        except Exception:
            chi2, pf = float('nan'), float('nan')
        q = NEMENYI_Q05.get(len(ms), float('nan'))
        cd = q * math.sqrt(len(ms) * (len(ms) + 1) / (6.0 * len(funcs)))
        for m, rk in zip(ms, mean_ranks):
            fr_rows.append({'algorithm': alg, 'method': m, 'mean_rank': float(rk), 'n_functions': len(funcs),
                            'friedman_chi2': float(chi2), 'friedman_p': float(pf), 'nemenyi_CD_0.05': cd})
    save_csv(desc_rows, f"stats_{tag}_descriptive")
    save_csv(pair_rows, f"stats_{tag}_pairwise")
    save_csv(sum_rows, f"stats_{tag}_summary")
    save_csv(fr_rows, f"stats_{tag}_friedman_nemenyi")
    return pd.DataFrame(sum_rows), pd.DataFrame(fr_rows)
