"""Experiment sizes, seeds, result paths and checkpointing.

Two environment variables control a run:

    GPUMETA_QUICK=1      small problem sizes (a complete dry run of every script)
    GPUMETA_RESULTS=dir  output directory (default: ./results, or
                         ./results_quick in quick mode)
"""
import math
import os
import pickle

import pandas as pd

QUICK_MODE = os.environ.get('GPUMETA_QUICK', '0').lower() not in ('', '0', 'false', 'no')

MASTER_SEED = 42


def make_seeds(n, offset=0):
    """Run-indexed seeds. Every compared method receives the same list."""
    return [int(MASTER_SEED + 1009 * i + offset) for i in range(n)]


ALGORITHMS = ['ABC', 'GWO', 'WOA', 'SCA', 'SSA', 'HHO']
TEST_FUNCTIONS = ['Sphere', 'Rastrigin', 'Rosenbrock', 'Ackley', 'Griewank', 'Schwefel',
                  'Levy', 'Zakharov', 'SumSquares', 'Michalewicz']

if QUICK_MODE:
    EXP_POP_SIZES = [128, 1024, 8192]
    EXP_DIMENSIONS = [10, 100]
    EXP_ITERATIONS = [500, 2000]
    DEFAULT_DIM, DEFAULT_POP, DEFAULT_ITERS = 50, 512, 500
    N_RUNS_STATS, N_RUNS_ABL, N_RUNS_SENS, N_RUNS_CEC = 5, 5, 3, 3
    STATS_ITERS, STATS_DIM, STATS_POP = 300, 30, 512
    CEC_ITERS, CEC_POP = 100, 64
else:
    EXP_POP_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
    EXP_DIMENSIONS = [10, 50, 100, 500, 1000]
    EXP_ITERATIONS = [1000, 5000, 10000]
    DEFAULT_DIM, DEFAULT_POP, DEFAULT_ITERS = 100, 1024, 5000
    N_RUNS_STATS, N_RUNS_ABL, N_RUNS_SENS, N_RUNS_CEC = 30, 30, 15, 15
    STATS_ITERS, STATS_DIM, STATS_POP = 1000, 50, 1024
    CEC_ITERS, CEC_POP = 500, 96

CEC_DIM = 30
CEC2017_FUNCTIONS = list(range(1, 30))   # the 29 functions of the suite (opfunu numbering)
CEC2020_FUNCTIONS = list(range(1, 11))   # the 10 functions of the suite

ALPHA_GRID = [1.0, 1.1, 1.2, 1.3, 1.5]   # Ballot-Elite threshold factor
TS_GRID = [2, 3, 4, 5]                   # tournament parameter
COST_GRID = [1, 10, 100, 1000]           # fitness-cost multiplier
STATS_METHODS = ['naive', 'tournament', 'tournament_elitist', 'ballot']
SENS_FUNCS = ['Rastrigin', 'Ackley', 'Schwefel']
ABL_FUNCS = ['Sphere', 'Rastrigin', 'Rosenbrock', 'Ackley', 'Schwefel']

# CPU Serial is skipped when ps * dim * iters exceeds this value. The largest
# configuration of the paper (8192 x 100 x 5000 = 4.1e12) stays below it.
SERIAL_CAP = 6e12

EPS_TOL = 1e-8                    # success tolerance on the error to the optimum
TOST_MARGIN = math.log10(2.0)     # equivalence margin: a factor of two in error
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Sizes of the operator-level and quality experiments.
if QUICK_MODE:
    CFG = dict(
        quick=True,
        correctness_ns=[1, 31, 32, 33, 96, 100, 1000, 1024, 1025], correctness_reps=2,
        takeover_reps=3, leader_iters=60, leader_seeds=1,
        pressure_reps=20,
        micro_ns=[128, 1024, 8192], micro_budget_s=0.05,
        equiv_pops=[1024], equiv_funcs=['Sphere', 'Rastrigin'], equiv_iters=100, equiv_seeds=1, equiv_dim=30,
        ablation_funcs=['Sphere'], ablation_runs=2, ablation_iters=100,
        fe_funcs=['Sphere', 'Rastrigin'], fe_iters=100, fe_seeds=1, fe_dim=30,
        quality_funcs=['Sphere', 'Ackley', 'Schwefel'], quality_runs=3, quality_iters=100,
        large_iters=100, large_pop=1024, large_funcs=['Sphere', 'Ackley'],
        cec_suites=[('cec2017', [1, 4]), ('cec2020', [1])], cec_runs=2, cec_iters=20,
        abc_suites=[('cec2017', [1]), ('cec2020', [1])], abc_runs=2, abc_iters=20,
        ml_runs_svm=2, ml_runs_xgb=1, ml_budget={'breast_cancer': (64, 3), 'digits': (32, 2)},
    )
else:
    CFG = dict(
        quick=False,
        correctness_ns=[1, 31, 32, 33, 64, 96, 100, 255, 256, 257, 1000, 1023, 1024, 1025, 4097, 8191, 8192],
        correctness_reps=10,
        takeover_reps=20, leader_iters=500, leader_seeds=3,
        pressure_reps=200,
        micro_ns=[128, 256, 512, 1024, 2048, 4096, 8192, 32768, 131072, 1048576], micro_budget_s=0.5,
        equiv_pops=[1024, 8192], equiv_funcs=list(TEST_FUNCTIONS), equiv_iters=DEFAULT_ITERS, equiv_seeds=3,
        equiv_dim=DEFAULT_DIM,
        ablation_funcs=list(ABL_FUNCS), ablation_runs=N_RUNS_ABL, ablation_iters=STATS_ITERS,
        fe_funcs=list(TEST_FUNCTIONS), fe_iters=DEFAULT_ITERS, fe_seeds=3, fe_dim=DEFAULT_DIM,
        quality_funcs=list(TEST_FUNCTIONS), quality_runs=N_RUNS_STATS, quality_iters=STATS_ITERS,
        large_iters=DEFAULT_ITERS, large_pop=8192, large_funcs=list(TEST_FUNCTIONS),
        cec_suites=[('cec2017', list(CEC2017_FUNCTIONS)), ('cec2020', list(CEC2020_FUNCTIONS))],
        cec_runs=N_RUNS_CEC, cec_iters=CEC_ITERS,
        abc_suites=[('cec2017', list(CEC2017_FUNCTIONS)), ('cec2020', list(CEC2020_FUNCTIONS))],
        abc_runs=N_RUNS_CEC, abc_iters=CEC_ITERS,
        ml_runs_svm=30, ml_runs_xgb=10, ml_budget={'breast_cancer': (64, 20), 'digits': (32, 15)},
    )

RESULTS_DIR = os.environ.get('GPUMETA_RESULTS', './results_quick' if QUICK_MODE else './results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Checkpoint: every experiment stores its raw results here after each
# configuration, so an interrupted run continues where it stopped. Delete the
# file to start again from scratch.
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'checkpoint.pkl')


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'rb') as f:
                data = pickle.load(f)
            print(f"[checkpoint] resumed from {CHECKPOINT_PATH} ({len(data)} sections)")
            return data
        except Exception as e:
            print(f"[checkpoint] could not be read ({e}); starting fresh")
    return {}


checkpoint = _load_checkpoint()


def save_checkpoint():
    tmp = CHECKPOINT_PATH + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(checkpoint, f)
    os.replace(tmp, CHECKPOINT_PATH)


def save_csv(rows, name):
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  [csv] {path}  ({len(df)} rows)")
    return df


def banner(title):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
