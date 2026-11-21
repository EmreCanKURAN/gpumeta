"""ABC onlooker-coupling ablation on CEC2017 and CEC2020.

Eight ways of choosing the food source that an onlooker bee exploits:

    roulette        fitness-proportional roulette wheel (naive)
    warp_direct     onlooker i exploits the warp-tournament winner sel[i]
    global4_direct  the same with a global tournament of four contestants
    orig_warp       original coupling with warp-tournament winners: distinct
                    winners, filled up with the best individuals, assigned in
                    fitness order (Algorithm B.1 of the paper)
    orig_elitist    original coupling with the elitist rule as winner source
    orig_global4    original coupling with global-tournament winners
    orig_random     original coupling with a random winner set of equal size
    rank_order      fitness-ordered assignment only, no selection kernel

This script runs on the CPU only.

Outputs: abc_coupling_runs.csv, stats_abc_coupling_*.csv
"""
from gpumeta.cec import cec_experiment, check_opfunu
from gpumeta.config import CFG, banner, save_csv
from gpumeta.stats import mechanism_analysis

VARIANTS = ['roulette', 'warp_direct', 'global4_direct', 'orig_warp', 'orig_elitist', 'orig_global4',
            'orig_random', 'rank_order']


def run():
    banner("ABC onlooker-coupling ablation (CEC2017 and CEC2020)")
    check_opfunu()
    ck = cec_experiment('abc_coupling', CFG['abc_suites'], ['ABC'], lambda a, s, f: VARIANTS, CFG['abc_runs'],
                        CFG['abc_iters'])
    rows = []
    data = {}
    for key, mv in ck.items():
        suite, dim, F, alg = key.split('_')
        fname = f"{suite.upper()}_{F}"
        data[('ABC', fname)] = {m: d['errs'] for m, d in mv.items()}
        for m, d in mv.items():
            for i, e in enumerate(d['errs']):
                rows.append({'suite': suite, 'function': fname, 'variant': m, 'run': i, 'error': e})
    save_csv(rows, 'abc_coupling_runs')
    mechanism_analysis(data, 'abc_coupling', VARIANTS)
    return data


if __name__ == '__main__':
    run()
