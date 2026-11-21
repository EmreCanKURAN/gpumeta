"""CEC2017 (29 functions) and CEC2020 (10 functions) with six algorithms and the
selection mechanisms naive, Tournament, Ballot-Elite, two migration rates and,
where applicable, the elitist rule (D = 30, population 96, 500 iterations,
15 seed-paired runs). See gpumeta/cec.py for the host-loop implementation.

This script runs on the CPU only. experiments/kernel_correctness.py verifies
on the GPU that the reference selection used here matches the kernels.

Outputs: cec_runs.csv, cec_convergence.csv, cec_elitist_identity_check.csv,
cec_winner_counts.csv
"""
import os

import numpy as np
import pandas as pd

from gpumeta.cec import cec_experiment, cec_methods, check_opfunu
from gpumeta.config import ALGORITHMS, CFG, RESULTS_DIR, banner, save_csv
from gpumeta.stats import floor_log10


def run():
    banner("CEC2017 and CEC2020: six algorithms, six selection mechanisms")
    check_opfunu()
    ck = cec_experiment('cec', CFG['cec_suites'], ALGORITHMS, cec_methods, CFG['cec_runs'],
                        CFG['cec_iters'])
    rows, conv_rows, ident = [], [], []
    data = {}
    for key, mv in ck.items():
        suite, dim, F, alg = key.split('_')
        fname = f"{suite.upper()}_{F}"
        for m, d in mv.items():
            for i, e in enumerate(d['errs']):
                rows.append({'suite': suite, 'function': fname, 'algorithm': alg, 'method': m, 'run': i,
                             'error': e, 'n_distinct_leaders': d['n_leaders'][i]})
            for it, v in (d['conv'] or []):
                conv_rows.append({'suite': suite, 'function': fname, 'algorithm': alg, 'method': m, 'iter': it,
                                  'error': v})
        if alg not in ('GWO', 'ABC') and 'elitist' in mv:
            ident.append({'function': fname, 'algorithm': alg,
                          'elitist_identical_to_naive_runs': int(np.sum(np.array(mv['elitist']['errs']) ==
                                                                        np.array(mv['naive']['errs']))),
                          'runs': len(mv['naive']['errs'])})
        data[(alg, fname)] = {m: d['errs'] for m, d in mv.items()}
    save_csv(rows, 'cec_runs')
    save_csv(conv_rows, 'cec_convergence')
    if ident:
        save_csv(ident, 'cec_elitist_identity_check')
        print(pd.DataFrame(ident).to_string(index=False))
    # sole-winner counts: lowest median error after flooring at the tolerance
    wc = []
    for (alg, fname), mv in data.items():
        ms = [m for m in ['naive', 'tournament', 'elitist', 'mig_0.03125', 'mig_0.25', 'ballot'] if m in mv]
        med = {m: np.median(floor_log10(mv[m])) for m in ms}
        lo = min(med.values())
        winners = [m for m in ms if med[m] == lo]
        for m in ms:
            wc.append({'algorithm': alg, 'function': fname, 'method': m, 'win': int(m in winners),
                       'sole_win': int(winners == [m])})
    wcd = pd.DataFrame(wc)
    tab = wcd.pivot_table(index='algorithm', columns='method', values='sole_win', aggfunc='sum')
    tab.to_csv(os.path.join(RESULTS_DIR, 'cec_winner_counts.csv'))
    print(tab.to_string())
    return data


if __name__ == '__main__':
    run()
