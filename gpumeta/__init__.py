"""Warp-level selection for GPU-accelerated metaheuristics.

    gpumeta.gpu         CUDA kernels (PyCUDA), selection dispatcher, GPU runners
    gpumeta.cpu         Numba CPU implementations (serial and parallel)
    gpumeta.reference   CPU reference implementation of the selection kernels
    gpumeta.cec         host-loop CEC2017 / CEC2020 experiments
    gpumeta.ml          models of the hyperparameter optimisation experiment
    gpumeta.stats       statistical tests and effect sizes
    gpumeta.energy      NVML energy measurement
    gpumeta.benchmarks  classical test functions
    gpumeta.config      experiment sizes, seeds, output paths

gpumeta.gpu creates a CUDA context when it is imported; the other modules do
not need a GPU.
"""

__version__ = '1.0.0'
