"""Statistical comparison of the selection mechanisms from the stored runs of
experiments/cec_quality.py and experiments/classical_quality.py.

For every algorithm and pair of mechanisms: direction and size of the
difference (median log10 error), Vargha-Delaney A12, rank-biserial
correlation, Wilcoxon signed-rank test (Mann-Whitney U when CPU Serial is
involved) with Holm correction across functions, TOST equivalence, and
Friedman ranks with the Nemenyi critical difference. Errors are floored at the
tolerance of 1e-8 before taking logarithms.

Outputs: stats_cec_*.csv, stats_classical_*.csv, cpu_serial_stability.csv
"""
import numpy as np

from gpumeta.benchmarks import optimum_value
from gpumeta.config import EPS_TOL, RESULTS_DIR, STATS_DIM, banner, checkpoint, save_csv
from gpumeta.stats import floor_log10, mechanism_analysis, stable_mean_std


def run():
    banner("Statistics: effect sizes, Holm-corrected tests, TOST, Friedman and Nemenyi")
    if 'cec' in checkpoint:
        data = {}
        for key, mv in checkpoint['cec'].items():
            suite, dim, F, alg = key.split('_')
            data[(alg, f"{suite.upper()}_{F}")] = {m: d['errs'] for m, d in mv.items()}
        mechanism_analysis(data, 'cec', ['naive', 'tournament', 'elitist', 'mig_0.03125', 'mig_0.25', 'ballot'])
    else:
        print("  no CEC runs in the checkpoint (run experiments.cec_quality first)")
    runs = checkpoint.get('classical')
    if not runs:
        print("  no classical runs in the checkpoint (run experiments.classical_quality first)")
        return
    # errors of the classical functions relative to the exact optimum
    data = {}
    for key, mv in runs.items():
        alg, fname = key.split('_', 1)
        fs = optimum_value(fname, STATS_DIM)
        data[(alg, fname)] = {m: [d['ref_value'] - fs for d in lst] for m, lst in mv.items()}
    mechanism_analysis(data, 'classical', ['naive', 'tournament', 'elitist_dev', 'mig_0.25', 'ballot', 'cpu_serial'])
    # CPU Serial: mean and standard deviation computed on rescaled values, which
    # stays accurate when the errors are many orders of magnitude below one
    rows = []
    for key, mv in runs.items():
        alg, fname = key.split('_', 1)
        if 'cpu_serial' in mv:
            v = np.array([d['ref_value'] for d in mv['cpu_serial']])
            err = v - optimum_value(fname, STATS_DIM)
            mu, sd = stable_mean_std(v)
            rows.append({'algorithm': alg, 'function': fname, 'mean_stable': mu, 'std_stable': sd,
                         'numpy_std_naive': float(np.std(v, ddof=1)), 'median': float(np.median(v)),
                         'median_log10_floor': float(np.median(floor_log10(err))),
                         'success_rate_eps': float(np.mean(err <= EPS_TOL))})
    if rows:
        save_csv(rows, 'cpu_serial_stability')
    print("  tables written to", RESULTS_DIR)


if __name__ == '__main__':
    run()
