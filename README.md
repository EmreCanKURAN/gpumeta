# gpumeta

Code for the paper

> E. C. Kuran, U. Kuran, E. Ecemiş, "Register-Level Selection via Warp Primitives: Shuffle-Based Tournament and Ballot-Elite Mechanisms for GPU-Accelerated Metaheuristics," *Swarm and Evolutionary Computation*, 2026.

The paper introduces two selection operators for population-based metaheuristics on GPUs that exchange competitor values in the registers of a 32-thread warp:

- **GPU Tournament** selection, built on `__shfl_xor_sync`;
- **GPU Ballot-Elite** selection, which adds elite qualification with `__ballot_sync` and a strategy switch based on the elite count of the warp.

Boundary-safe versions of both kernels work for any population size, and a migration rate ρ controls how often a thread also compares its winner with the global best. The operators are evaluated inside six algorithms (ABC, GWO, WOA, SCA, SSA, HHO) written as native CUDA kernels and compared with single-threaded and multi-threaded CPU implementations.

This repository contains the kernels, the CPU implementations and one script for every experiment of the paper.

## Layout

```
gpumeta/
  kernels/metaheuristics.cu    algorithms, test functions and selection operators
  kernels/selection_safe.cu    boundary-safe selection kernels, migration, baseline operators
  gpu.py                       PyCUDA: compilation, selection dispatcher, GPU runners
  cpu.py                       Numba implementations (serial and parallel)
  reference.py                 CPU reference implementation of the selection kernels
  cec.py                       host-loop CEC2017 / CEC2020 experiments
  ml.py                        models and search spaces of the hyperparameter experiment
  stats.py, energy.py, benchmarks.py, config.py
experiments/                   one script per experiment (see the table below)
run_all.py                     runs the experiments one after another
colab.ipynb                    the same on Google Colab
tools/system_info.py           prints the hardware and software configuration
```

## Requirements

- an NVIDIA GPU with a CUDA 12 toolkit (`nvcc` is needed by PyCUDA);
- Python 3.12 (the pinned packages also have wheels for 3.10 and 3.11).

```
pip install -r requirements.txt
pip install --no-deps --ignore-requires-python opfunu==1.0.4
```

The second line is needed because opfunu 1.0.4 declares Python < 3.12, so on Python 3.12 pip would otherwise fall back to version 1.0.1. That version has a sign error in the modified Schwefel function and wrong mixing proportions in two CEC2017 hybrid functions, which affects 18 of the 39 CEC functions. The scripts refuse to run the CEC experiments with an older version.

`cupy-cuda12x` is only used by `experiments/library_baseline.py`, and `xgboost` only by `experiments/ml_tuning.py`. Both scripts skip the corresponding part when the package is missing. On Google Colab it is enough to install `pycuda`, `nvidia-ml-py`, `xgboost` and `opfunu` on top of the preinstalled packages; `colab.ipynb` does this.

## Running

A complete dry run of every script with small problem sizes:

```
python run_all.py --quick
```

Paper-scale runs:

```
python run_all.py                           # everything
python run_all.py --only scaling ablation   # selected steps
python run_all.py --list                    # names of the steps
python -m experiments.scaling population    # a single script
```

Results are written as CSV files (and a few PNG figures) to `./results`, or `./results_quick` in quick mode. The directory can be changed with `--results DIR` or the environment variable `GPUMETA_RESULTS`; quick mode can also be switched on with `GPUMETA_QUICK=1` when a script is started directly. Every script saves its raw results to `checkpoint.pkl` in the results directory after each configuration, so an interrupted run continues where it stopped when it is started again. Delete that file to start from scratch.

The paper-scale runs are long. The three scaling sweeps alone take roughly 18 hours on an A100, most of it spent in the CPU Serial baseline, and the two CEC scripts take several hours on a multi-core CPU.

## Experiments

| Script | Content | Tables and figures | Main output files |
| --- | --- | --- | --- |
| `kernel_correctness` | GPU kernels against the CPU reference for population sizes 1 to 8192 | Section 4 | `kernel_correctness.csv` |
| `scaling` | five platforms, sweeps over population, dimension and iterations | Tables 9-12, 29, 40-43, 50; Fig. 5 | `scaling_*.csv` |
| `platform_energy` | runtime, energy, power, energy-delay product | Tables 13, 14 | `platform_energy.csv` |
| `convergence` | convergence curves | Figs. 8, 9 | `convergence.csv` |
| `sensitivity` | α and tournament-parameter sensitivity | Tables 23, 24 | `sensitivity.csv` |
| `selection_dynamics` | takeover, diversity, selection intensity, GWO leaders | Tables 20-22, B.6 | `takeover*.csv`, `diversity.csv`, `matched_selection_pressure.csv`, `gwo_leader_statistics.csv` |
| `microbenchmark` | isolated cost of the selection operators | Table B.1 | `selection_microbenchmark.csv` |
| `profiling` | SASS, compute-sanitizer, Nsight Compute, register usage | Tables 3, B.2 | `profiling_*.csv` |
| `equivalent_settings` | end-to-end runtime with only the selection operator exchanged | Table B.3 | `equivalent_settings_*.csv` |
| `ablation` | twelve selection variants | Table 25 | `ablation_*.csv` |
| `function_evaluations` | evaluations per run | Table B.4 | `function_evaluations.csv` |
| `fitness_cost` | break-even against the evaluation cost | Table 26; Fig. 7 | `fitness_cost.csv` |
| `library_baseline` | native kernels against CuPy | Table 27 | `library_baseline.csv` |
| `cec_gpu_throughput` | shifted and rotated functions on the device | Tables 18, 19; Fig. 6 | `cec_gpu_throughput.csv` |
| `classical_quality` | solution quality on the classical functions | Tables 30-39, 45-49, B.8 | `classical_runs.csv`, `classical_large_runs.csv`, `evaluator_check.csv` |
| `cec_quality` | CEC2017 and CEC2020 | Tables 16, 17; Figs. A.1-A.4 | `cec_runs.csv`, `cec_convergence.csv` |
| `abc_coupling` | ABC onlooker-coupling ablation | Table B.5 | `abc_coupling_runs.csv` |
| `ml_tuning` | SVM and XGBoost hyperparameter optimisation with nested CV | Table 28 | `ml_runs.csv`, `ml_statistics.csv` |
| `analysis` | statistical comparison of the mechanisms | Tables 17, 45, B.7, B.9 | `stats_cec_*.csv`, `stats_classical_*.csv` |

Notes:

- The CPU columns of Tables 30-39 come from the population sweep of `scaling` at N = 8192.
- Table 15 (other GPUs) is the population sweep restricted to one function: `python -m experiments.scaling population Rastrigin`.
- `cec_quality`, `abc_coupling` and `analysis` do not need a GPU. Their selection step uses `gpumeta/reference.py`, which `kernel_correctness` checks against the GPU kernels.
- The files `stats_cec_*.csv` and `stats_classical_*.csv` correspond to the files `A_E10R_cec_*.csv` and `A_X10_classical_*.csv` of Supplementary Material 2 of the article.
- `profiling` needs `nvcc`, `cuobjdump`, `compute-sanitizer` and `ncu` from the CUDA toolkit and skips the tools it cannot find. Nsight Compute needs access to the GPU performance counters, which shared machines such as Colab usually do not grant; the script writes the driver and the command line to `results/profiling` so that this step can be repeated on a local machine.

## Reproducibility

- All runs are seeded from `MASTER_SEED = 42` in `gpumeta/config.py`, and every compared selection mechanism receives the same seeds. GPU runs with the same seed start from the same initial population, so comparisons between GPU mechanisms are paired.
- The random number generator of the Numba CPU implementations is not controlled by the NumPy seed. CPU results therefore vary between executions and are treated as independent samples in the statistical tests.
- Runtimes depend on the hardware. The paper reports an A100 (Colab), a T4 and an RTX 4060 Laptop GPU.
- `metaheuristics.cu` writes π as 3.14159265 and e as 2.71828183. The timing experiments and the convergence curves use the kernels as they are. `classical_quality` compiles the same source with exact constants and evaluates the best position of every run again with the CPU reference evaluator, so the reported errors do not contain the Ackley floor of 1.5e-9 that the truncated e produces.
- Griewank is evaluated in a log-product form: the product of the cosine terms is accumulated as a sum of logarithms, which keeps the value accurate close to the optimum. A logarithm needs a positive argument, so the product is set to 0 as soon as one cosine term is zero or negative, and the function then returns `sum(x_i^2) / 4000 + 1`. The textbook formula printed in the paper keeps the sign of the product. The two forms are identical wherever every cosine term is positive, which includes the neighbourhood of the global optimum (value 0 at the origin), and elsewhere they differ by the absolute value of the product, which is at most 1. The GPU kernels (`griewank` in `metaheuristics.cu`), the CPU implementations and the reference evaluator (`griewank_numba` in `gpumeta/benchmarks.py`) all use this form, so every compared method optimises the same function. Griewank values obtained with this code should be compared with other studies with this in mind.
- `tournament` and `ballot` are the kernels of `metaheuristics.cu`, which expect a population size that is a multiple of 32. `tournament_safe`, `ballot_safe` and the migration kernels (`mig_<rho>`) accept any population size and give identical results for multiples of 32.

## Using the kernels directly

```python
from gpumeta.benchmarks import BENCHMARKS
from gpumeta.gpu import run_gwo_gpu

f = BENCHMARKS['Rastrigin']
res = run_gwo_gpu(1024, 1000, f['id'], f['bounds'], dim=50, method='mig_0.25', seed=1)
print(res['best'], res['time_ms'])
```

`gpumeta.gpu.fill_selection` applies a selection operator to any device array of objective values; the list of method names is in the docstring of `gpumeta/gpu.py`.

## Citation

```bibtex
@article{kuran2026warpselection,
  author  = {Kuran, Emre Can and Kuran, Umut and Ecemi{\c{s}}, Ece},
  title   = {Register-Level Selection via Warp Primitives: Shuffle-Based Tournament and
             Ballot-Elite Mechanisms for {GPU}-Accelerated Metaheuristics},
  journal = {Swarm and Evolutionary Computation},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
