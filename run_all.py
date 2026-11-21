"""Run the experiments of the paper one after another.

    python run_all.py --list
    python run_all.py --quick                    # complete dry run with small sizes
    python run_all.py                            # paper-scale runs
    python run_all.py --only scaling ablation
    python run_all.py --skip profiling ml_tuning

Results are written to ./results (./results_quick with --quick) or to the
directory given with --results. Every script stores its progress in
checkpoint.pkl inside that directory, so an interrupted run can simply be
started again.
"""
import argparse
import importlib
import os
import time
import traceback

STEPS = [
    'kernel_correctness',
    'scaling',
    'platform_energy',
    'convergence',
    'sensitivity',
    'selection_dynamics',
    'microbenchmark',
    'profiling',
    'equivalent_settings',
    'ablation',
    'function_evaluations',
    'fitness_cost',
    'library_baseline',
    'cec_gpu_throughput',
    'classical_quality',
    'cec_quality',
    'abc_coupling',
    'ml_tuning',
    'analysis',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--quick', action='store_true', help='small problem sizes')
    ap.add_argument('--only', nargs='+', metavar='STEP', help='run only these steps')
    ap.add_argument('--skip', nargs='+', metavar='STEP', default=[], help='leave these steps out')
    ap.add_argument('--results', metavar='DIR', help='output directory')
    ap.add_argument('--list', action='store_true', help='print the step names and exit')
    args = ap.parse_args()
    if args.list:
        print('\n'.join(STEPS))
        return
    unknown = [s for s in (args.only or []) + args.skip if s not in STEPS]
    if unknown:
        ap.error(f"unknown step(s): {', '.join(unknown)}")

    # gpumeta.config reads these when it is first imported
    if args.quick:
        os.environ['GPUMETA_QUICK'] = '1'
    if args.results:
        os.environ['GPUMETA_RESULTS'] = args.results

    t0 = time.time()
    failures = []
    for name in STEPS:
        if (args.only and name not in args.only) or name in args.skip:
            continue
        ts = time.time()
        try:
            importlib.import_module(f'experiments.{name}').run()
            print(f"[{name}] finished in {(time.time() - ts) / 60:.1f} min")
        except Exception as ex:
            traceback.print_exc()
            failures.append(name)
            print(f"[{name}] failed: {ex}")
            if name == 'kernel_correctness':
                break   # nothing else is meaningful if the kernels are wrong
    print(f"\ntotal {(time.time() - t0) / 3600:.2f} h | failed steps: {', '.join(failures) or 'none'}")


if __name__ == '__main__':
    main()
