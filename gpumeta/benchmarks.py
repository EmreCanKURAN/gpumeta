"""Classical test functions (Numba) used by the CPU implementations and as the
reference evaluator of the GPU results."""
import math

import numpy as np
from numba import jit, prange
from scipy.optimize import minimize_scalar


@jit(nopython=True)
def sphere_numba(x):
    return np.sum(x**2)


@jit(nopython=True)
def rastrigin_numba(x):
    # Numerically stable: sum(x^2) + 20*sum(sin^2(pi*x))
    n = len(x)
    sum_sq = 0.0
    sum_sin = 0.0
    for i in range(n):
        sum_sq += x[i] * x[i]
        sin_val = np.sin(np.pi * x[i])
        sum_sin += sin_val * sin_val
    return sum_sq + 20.0 * sum_sin


@jit(nopython=True)
def rosenbrock_numba(x):
    s = 0.0
    for i in range(len(x) -1):
        s += 100 *(x[i +1] - x[i]**2)**2 + (x[i] - 1)**2
    return s


@jit(nopython=True)
def ackley_numba(x):
    n = len(x)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        sum1 += x[i] * x[i]
        sum2 += np.cos(2.0 * np.pi * x[i])
    return -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20.0 + np.e


@jit(nopython=True)
def griewank_numba(x):
    # Numerically stable via log-product
    n = len(x)
    sum_part = 0.0
    log_prod = 0.0
    for i in range(n):
        sum_part += x[i] * x[i]
        theta = x[i] / np.sqrt(i + 1.0)
        cos_val = np.cos(theta)
        if cos_val > 0.9999999999:
            log_prod += -theta * theta / 2.0
        elif cos_val > 0:
            log_prod += np.log(cos_val)
        else:
            log_prod = -1e100
    if log_prod > -1e-10:
        one_minus_prod = -log_prod
    else:
        one_minus_prod = 1.0 - np.exp(log_prod)
    return sum_part / 4000.0 + one_minus_prod


@jit(nopython=True)
def schwefel_numba(x):
    n = len(x)
    s = 0.0
    for i in range(n):
        s += x[i] * np.sin(np.sqrt(np.abs(x[i])))
    return 418.9829 * n - s


@jit(nopython=True)
def levy_numba(x):
    n = len(x)
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        w[i] = 1.0 + (x[i] - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[n -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[n -1]) ** 2)
    sum_part = 0.0
    for i in range(n - 1):
        sum_part += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[i] + 1.0) ** 2)
    return term1 + sum_part + term3


@jit(nopython=True)
def zakharov_numba(x):
    n = len(x)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        sum1 += x[i] ** 2
        sum2 += 0.5 * (i + 1) * x[i]
    return sum1 + sum2 ** 2 + sum2 ** 4


@jit(nopython=True)
def sum_squares_numba(x):
    n = len(x)
    s = 0.0
    for i in range(n):
        s += (i + 1) * x[i] ** 2
    return s


@jit(nopython=True)
def michalewicz_numba(x):
    n = len(x)
    m = 10.0
    s = 0.0
    for i in range(n):
        s += np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** (2 * m)
    return -s + n


@jit(nopython=True)
def evaluate_numba(x, fid):
    if fid == 0:
        return sphere_numba(x)
    elif fid == 1:
        return rastrigin_numba(x)
    elif fid == 2:
        return rosenbrock_numba(x)
    elif fid == 3:
        return ackley_numba(x)
    elif fid == 4:
        return griewank_numba(x)
    elif fid == 5:
        return schwefel_numba(x)
    elif fid == 6:
        return levy_numba(x)
    elif fid == 7:
        return zakharov_numba(x)
    elif fid == 8:
        return sum_squares_numba(x)
    elif fid == 9:
        return michalewicz_numba(x)
    else:
        return sphere_numba(x)


@jit(nopython=True, parallel=True)
def evaluate_batch_parallel(pop, fid):
    n = pop.shape[0]
    obj = np.empty(n, dtype=np.float64)
    for i in prange(n):
        obj[i] = evaluate_numba(pop[i], fid)
    return obj


BENCHMARKS = {
    'Sphere':      {'id': 0, 'bounds': (-100, 100), 'optimal': 0.0},
    'Rastrigin':   {'id': 1, 'bounds': (-5.12, 5.12), 'optimal': 0.0},
    'Rosenbrock':  {'id': 2, 'bounds': (-30, 30), 'optimal': 0.0},
    'Ackley':      {'id': 3, 'bounds': (-32.768, 32.768), 'optimal': 0.0},
    'Griewank':    {'id': 4, 'bounds': (-600, 600), 'optimal': 0.0},
    'Schwefel':    {'id': 5, 'bounds': (-500, 500), 'optimal': 0.0},
    'Levy':        {'id': 6, 'bounds': (-10, 10), 'optimal': 0.0},
    'Zakharov':    {'id': 7, 'bounds': (-5, 10), 'optimal': 0.0},
    'SumSquares':  {'id': 8, 'bounds': (-10, 10), 'optimal': 0.0},
    'Michalewicz': {'id': 9, 'bounds': (0, np.pi), 'optimal': 0.0},
}

EXPENSIVE_FID = 60   # sphere with a scalable evaluation cost (GPU side: d_fitness_cost)


def optimum_value(fname, dim):
    """Optimum value of a test function. Schwefel is implemented with the
    customary constant 418.9829, so its optimum is d * 1.2728e-5 and not 0."""
    if fname == 'Schwefel':
        r = minimize_scalar(lambda x: -x * math.sin(math.sqrt(abs(x))), bounds=(400.0, 440.0),
                            method='bounded', options={'xatol': 1e-12})
        return dim * (418.9829 + r.fun)
    return 0.0
