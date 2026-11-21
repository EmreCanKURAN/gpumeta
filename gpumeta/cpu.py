"""CPU implementations of the six algorithms (Numba): a single-threaded
version and a multi-threaded version (prange) of each."""
import multiprocessing as mp
import time

import numpy as np
from numba import jit, prange, set_num_threads

from .benchmarks import evaluate_batch_parallel, evaluate_numba

CPU_CORES = mp.cpu_count()
N_THREADS = min(CPU_CORES, 8)
set_num_threads(N_THREADS)


# --------------------------- ABC (Karaboga 2005) -----------------------------
@jit(nopython=True)
def abc_numba_serial(fid, lb, ub, ps, iters, dim):
    limit = min(ps * dim, 100 * ps)
    pop = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pop[i, j] = lb + np.random.random() * (ub - lb)
    obj = np.empty(ps, dtype=np.float64)
    fitness = np.empty(ps, dtype=np.float64)
    for i in range(ps):
        obj[i] = evaluate_numba(pop[i], fid)
        if obj[i] >= 0:
            fitness[i] = 1.0 / (1.0 + obj[i])
        else:
            fitness[i] = 1.0 + np.abs(obj[i])
    trial = np.zeros(ps, dtype=np.int32)
    best_obj = obj[0]
    for i in range(1, ps):
        if obj[i] < best_obj:
            best_obj = obj[i]
    for it in range(iters):
        # Employed bees
        for i in range(ps):
            j = np.random.randint(0, dim)
            k = np.random.randint(0, ps)
            while k == i:
                k = np.random.randint(0, ps)
            Xnew = pop[i].copy()
            phi = -1.0 + 2.0 * np.random.random()
            Xnew[j] = pop[i, j] + phi * (pop[i, j] - pop[k, j])
            if Xnew[j] < lb: Xnew[j] = lb
            if Xnew[j] > ub: Xnew[j] = ub
            new_obj = evaluate_numba(Xnew, fid)
            new_fit = 1.0 / (1.0 + new_obj) if new_obj >= 0 else 1.0 + np.abs(new_obj)
            if new_fit > fitness[i]:
                for d in range(dim):
                    pop[i, d] = Xnew[d]
                obj[i] = new_obj
                fitness[i] = new_fit
                trial[i] = 0
            else:
                trial[i] += 1
        # Onlooker bees
        max_fit = fitness[0]
        for i in range(1, ps):
            if fitness[i] > max_fit:
                max_fit = fitness[i]
        prob = np.empty(ps, dtype=np.float64)
        for i in range(ps):
            prob[i] = 0.9 * (fitness[i] / max_fit) + 0.1
        m = 0
        n = 0
        while m < ps:
            if np.random.random() < prob[n]:
                j = np.random.randint(0, dim)
                k = np.random.randint(0, ps)
                while k == n:
                    k = np.random.randint(0, ps)
                Xnew = pop[n].copy()
                phi = -1.0 + 2.0 * np.random.random()
                Xnew[j] = pop[n, j] + phi * (pop[n, j] - pop[k, j])
                if Xnew[j] < lb: Xnew[j] = lb
                if Xnew[j] > ub: Xnew[j] = ub
                new_obj = evaluate_numba(Xnew, fid)
                new_fit = 1.0 / (1.0 + new_obj) if new_obj >= 0 else 1.0 + np.abs(new_obj)
                if new_fit > fitness[n]:
                    for d in range(dim):
                        pop[n, d] = Xnew[d]
                    obj[n] = new_obj
                    fitness[n] = new_fit
                    trial[n] = 0
                else:
                    trial[n] += 1
                m += 1
            n = (n + 1) % ps
        for i in range(ps):
            if obj[i] < best_obj:
                best_obj = obj[i]
        # Scout (single, max trial)
        max_trial_idx = 0
        max_trial_val = trial[0]
        for i in range(1, ps):
            if trial[i] > max_trial_val:
                max_trial_val = trial[i]
                max_trial_idx = i
        if max_trial_val > limit:
            trial[max_trial_idx] = 0
            for j in range(dim):
                pop[max_trial_idx, j] = lb + np.random.random() * (ub - lb)
            obj[max_trial_idx] = evaluate_numba(pop[max_trial_idx], fid)
            if obj[max_trial_idx] >= 0:
                fitness[max_trial_idx] = 1.0 / (1.0 + obj[max_trial_idx])
            else:
                fitness[max_trial_idx] = 1.0 + np.abs(obj[max_trial_idx])
    for i in range(ps):
        if obj[i] < best_obj:
            best_obj = obj[i]
    return best_obj


@jit(nopython=True, parallel=True)
def abc_employed_parallel(pop, obj, fitness, trial, lb, ub, dim, ps, fid):
    for i in prange(ps):
        j = np.random.randint(0, dim)
        k = np.random.randint(0, ps)
        while k == i:
            k = np.random.randint(0, ps)
        Xnew = pop[i].copy()
        phi = -1.0 + 2.0 * np.random.random()
        Xnew[j] = pop[i, j] + phi * (pop[i, j] - pop[k, j])
        if Xnew[j] < lb: Xnew[j] = lb
        if Xnew[j] > ub: Xnew[j] = ub
        new_obj = evaluate_numba(Xnew, fid)
        new_fit = 1.0 / (1.0 + new_obj) if new_obj >= 0 else 1.0 + np.abs(new_obj)
        if new_fit > fitness[i]:
            for d in range(dim):
                pop[i, d] = Xnew[d]
            obj[i] = new_obj
            fitness[i] = new_fit
            trial[i] = 0
        else:
            trial[i] += 1


@jit(nopython=True)
def abc_numba_parallel(fid, lb, ub, ps, iters, dim):
    limit = min(ps * dim, 100 * ps)
    pop = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pop[i, j] = lb + np.random.random() * (ub - lb)
    obj = evaluate_batch_parallel(pop, fid)
    fitness = np.empty(ps, dtype=np.float64)
    for i in range(ps):
        fitness[i] = 1.0 / (1.0 + obj[i]) if obj[i] >= 0 else 1.0 + np.abs(obj[i])
    trial = np.zeros(ps, dtype=np.int32)
    best_obj = obj[0]
    for i in range(1, ps):
        if obj[i] < best_obj:
            best_obj = obj[i]
    for it in range(iters):
        abc_employed_parallel(pop, obj, fitness, trial, lb, ub, dim, ps, fid)
        max_fit = fitness[0]
        for i in range(1, ps):
            if fitness[i] > max_fit:
                max_fit = fitness[i]
        prob = np.empty(ps, dtype=np.float64)
        for i in range(ps):
            prob[i] = 0.9 * (fitness[i] / max_fit) + 0.1
        m = 0
        n = 0
        while m < ps:
            if np.random.random() < prob[n]:
                j = np.random.randint(0, dim)
                k = np.random.randint(0, ps)
                while k == n:
                    k = np.random.randint(0, ps)
                Xnew = pop[n].copy()
                phi = -1.0 + 2.0 * np.random.random()
                Xnew[j] = pop[n, j] + phi * (pop[n, j] - pop[k, j])
                if Xnew[j] < lb: Xnew[j] = lb
                if Xnew[j] > ub: Xnew[j] = ub
                new_obj = evaluate_numba(Xnew, fid)
                new_fit = 1.0 / (1.0 + new_obj) if new_obj >= 0 else 1.0 + np.abs(new_obj)
                if new_fit > fitness[n]:
                    for d in range(dim):
                        pop[n, d] = Xnew[d]
                    obj[n] = new_obj
                    fitness[n] = new_fit
                    trial[n] = 0
                else:
                    trial[n] += 1
                m += 1
            n = (n + 1) % ps
        for i in range(ps):
            if obj[i] < best_obj:
                best_obj = obj[i]
        max_trial_idx = 0
        max_trial_val = trial[0]
        for i in range(1, ps):
            if trial[i] > max_trial_val:
                max_trial_val = trial[i]
                max_trial_idx = i
        if max_trial_val > limit:
            trial[max_trial_idx] = 0
            for j in range(dim):
                pop[max_trial_idx, j] = lb + np.random.random() * (ub - lb)
            obj[max_trial_idx] = evaluate_numba(pop[max_trial_idx], fid)
            if obj[max_trial_idx] >= 0:
                fitness[max_trial_idx] = 1.0 / (1.0 + obj[max_trial_idx])
            else:
                fitness[max_trial_idx] = 1.0 + np.abs(obj[max_trial_idx])
    return best_obj

# --------------------------- GWO (Mirjalili 2014) -----------------------------


@jit(nopython=True)
def gwo_numba_serial(fid, lb, ub, ps, iters, dim):
    alpha_pos = np.zeros(dim, dtype=np.float64)
    alpha_score = np.float64(np.inf)
    beta_pos = np.zeros(dim, dtype=np.float64)
    beta_score = np.float64(np.inf)
    delta_pos = np.zeros(dim, dtype=np.float64)
    delta_score = np.float64(np.inf)
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    for it in range(iters):
        for i in range(ps):
            for j in range(dim):
                if pos[i, j] > ub: pos[i, j] = ub
                if pos[i, j] < lb: pos[i, j] = lb
            fitness = evaluate_numba(pos[i], fid)
            if fitness < alpha_score:
                delta_score = beta_score
                for j in range(dim):
                    delta_pos[j] = beta_pos[j]
                beta_score = alpha_score
                for j in range(dim):
                    beta_pos[j] = alpha_pos[j]
                alpha_score = fitness
                for j in range(dim):
                    alpha_pos[j] = pos[i, j]
            elif fitness < beta_score:
                delta_score = beta_score
                for j in range(dim):
                    delta_pos[j] = beta_pos[j]
                beta_score = fitness
                for j in range(dim):
                    beta_pos[j] = pos[i, j]
            elif fitness < delta_score:
                delta_score = fitness
                for j in range(dim):
                    delta_pos[j] = pos[i, j]
        a = 2.0 - float(it) * (2.0 / iters)
        for i in range(ps):
            for j in range(dim):
                r1 = np.random.random(); r2 = np.random.random()
                A1 = 2.0 * a * r1 - a; C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_pos[j] - pos[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                r1 = np.random.random(); r2 = np.random.random()
                A2 = 2.0 * a * r1 - a; C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos[j] - pos[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                r1 = np.random.random(); r2 = np.random.random()
                A3 = 2.0 * a * r1 - a; C3 = 2.0 * r2
                D_delta = np.abs(C3 * delta_pos[j] - pos[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                pos[i, j] = (X1 + X2 + X3) / 3.0
    return alpha_score


@jit(nopython=True, parallel=True)
def gwo_boundary_check_parallel(pos, lb, ub, ps, dim):
    for i in prange(ps):
        for j in range(dim):
            if pos[i, j] > ub: pos[i, j] = ub
            if pos[i, j] < lb: pos[i, j] = lb


@jit(nopython=True, parallel=True)
def gwo_position_update_parallel(pos, alpha_pos, beta_pos, delta_pos, lb, ub, dim, ps, a):
    for i in prange(ps):
        for j in range(dim):
            r1 = np.random.random(); r2 = np.random.random()
            A1 = 2.0 * a * r1 - a; C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos[j] - pos[i, j])
            X1 = alpha_pos[j] - A1 * D_alpha
            r1 = np.random.random(); r2 = np.random.random()
            A2 = 2.0 * a * r1 - a; C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta_pos[j] - pos[i, j])
            X2 = beta_pos[j] - A2 * D_beta
            r1 = np.random.random(); r2 = np.random.random()
            A3 = 2.0 * a * r1 - a; C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta_pos[j] - pos[i, j])
            X3 = delta_pos[j] - A3 * D_delta
            pos[i, j] = (X1 + X2 + X3) / 3.0


@jit(nopython=True)
def gwo_numba_parallel(fid, lb, ub, ps, iters, dim):
    alpha_pos = np.zeros(dim, dtype=np.float64)
    alpha_score = np.float64(np.inf)
    beta_pos = np.zeros(dim, dtype=np.float64)
    beta_score = np.float64(np.inf)
    delta_pos = np.zeros(dim, dtype=np.float64)
    delta_score = np.float64(np.inf)
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    for it in range(iters):
        gwo_boundary_check_parallel(pos, lb, ub, ps, dim)
        obj = evaluate_batch_parallel(pos, fid)
        for i in range(ps):
            fitness = obj[i]
            if fitness < alpha_score:
                delta_score = beta_score
                for j in range(dim):
                    delta_pos[j] = beta_pos[j]
                beta_score = alpha_score
                for j in range(dim):
                    beta_pos[j] = alpha_pos[j]
                alpha_score = fitness
                for j in range(dim):
                    alpha_pos[j] = pos[i, j]
            elif fitness < beta_score:
                delta_score = beta_score
                for j in range(dim):
                    delta_pos[j] = beta_pos[j]
                beta_score = fitness
                for j in range(dim):
                    beta_pos[j] = pos[i, j]
            elif fitness < delta_score:
                delta_score = fitness
                for j in range(dim):
                    delta_pos[j] = pos[i, j]
        a = 2.0 - float(it) * (2.0 / iters)
        gwo_position_update_parallel(pos, alpha_pos, beta_pos, delta_pos, lb, ub, dim, ps, a)
    return alpha_score

# --------------------------- WOA (Mirjalili 2016) -----------------------------


@jit(nopython=True)
def woa_numba_serial(fid, lb, ub, ps, iters, dim):
    leader_pos = np.zeros(dim, dtype=np.float64)
    leader_score = np.float64(np.inf)
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    b = 1.0
    for it in range(iters):
        for i in range(ps):
            for j in range(dim):
                if pos[i, j] > ub: pos[i, j] = ub
                if pos[i, j] < lb: pos[i, j] = lb
            fitness = evaluate_numba(pos[i], fid)
            if fitness < leader_score:
                leader_score = fitness
                for j in range(dim):
                    leader_pos[j] = pos[i, j]
        a = 2.0 - float(it) * (2.0 / iters)
        a2 = -1.0 + float(it) * ((-1.0) / iters)
        for i in range(ps):
            r1 = np.random.random(); r2 = np.random.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            l = (a2 - 1.0) * np.random.random() + 1.0
            p = np.random.random()
            for j in range(dim):
                if p < 0.5:
                    if np.abs(A) >= 1.0:
                        rand_leader_index = np.random.randint(0, ps)
                        X_rand = pos[rand_leader_index, j]
                        D_X_rand = np.abs(C * X_rand - pos[i, j])
                        pos[i, j] = X_rand - A * D_X_rand
                    else:
                        D_Leader = np.abs(C * leader_pos[j] - pos[i, j])
                        pos[i, j] = leader_pos[j] - A * D_Leader
                else:
                    distance2Leader = np.abs(leader_pos[j] - pos[i, j])
                    pos[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2.0 * np.pi) + leader_pos[j]
    return leader_score


@jit(nopython=True, parallel=True)
def woa_boundary_check_parallel(pos, lb, ub, ps, dim):
    for i in prange(ps):
        for j in range(dim):
            if pos[i, j] > ub: pos[i, j] = ub
            if pos[i, j] < lb: pos[i, j] = lb


@jit(nopython=True, parallel=True)
def woa_position_update_parallel(pos, leader_pos, lb, ub, dim, ps, a, a2, b):
    for i in prange(ps):
        r1 = np.random.random(); r2 = np.random.random()
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        l = (a2 - 1.0) * np.random.random() + 1.0
        p = np.random.random()
        for j in range(dim):
            if p < 0.5:
                if np.abs(A) >= 1.0:
                    rand_idx = np.random.randint(0, ps)
                    X_rand = pos[rand_idx, j]
                    D_X_rand = np.abs(C * X_rand - pos[i, j])
                    pos[i, j] = X_rand - A * D_X_rand
                else:
                    D_Leader = np.abs(C * leader_pos[j] - pos[i, j])
                    pos[i, j] = leader_pos[j] - A * D_Leader
            else:
                distance2Leader = np.abs(leader_pos[j] - pos[i, j])
                pos[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2.0 * np.pi) + leader_pos[j]


@jit(nopython=True)
def woa_numba_parallel(fid, lb, ub, ps, iters, dim):
    leader_pos = np.zeros(dim, dtype=np.float64)
    leader_score = np.float64(np.inf)
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    b = np.float64(1.0)
    for it in range(iters):
        woa_boundary_check_parallel(pos, lb, ub, ps, dim)
        obj = evaluate_batch_parallel(pos, fid)
        for i in range(ps):
            if obj[i] < leader_score:
                leader_score = obj[i]
                for j in range(dim):
                    leader_pos[j] = pos[i, j]
        a = 2.0 - float(it) * (2.0 / iters)
        a2 = -1.0 + float(it) * ((-1.0) / iters)
        woa_position_update_parallel(pos, leader_pos, lb, ub, dim, ps, a, a2, b)
    return leader_score

# --------------------------- SCA (Mirjalili 2016) -----------------------------


@jit(nopython=True)
def sca_numba_serial(fid, lb, ub, ps, iters, dim):
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    obj = np.empty(ps, dtype=np.float64)
    for i in range(ps):
        obj[i] = evaluate_numba(pos[i], fid)
    best_idx = np.argmin(obj)
    best_pos = pos[best_idx].copy()
    best_obj = obj[best_idx]
    for it in range(iters):
        r1 = 2.0 - 2.0 * (it + 1) / iters
        for i in range(ps):
            for j in range(dim):
                r2 = 2.0 * np.pi * np.random.random()
                r3 = 2.0 * np.random.random()
                r4 = np.random.random()
                if r4 < 0.5:
                    pos[i, j] = pos[i, j] + r1 * np.sin(r2) * np.abs(r3 * best_pos[j] - pos[i, j])
                else:
                    pos[i, j] = pos[i, j] + r1 * np.cos(r2) * np.abs(r3 * best_pos[j] - pos[i, j])
        for i in range(ps):
            for j in range(dim):
                if pos[i, j] > ub: pos[i, j] = ub
                if pos[i, j] < lb: pos[i, j] = lb
            obj[i] = evaluate_numba(pos[i], fid)
            if obj[i] < best_obj:
                best_obj = obj[i]
                for j in range(dim):
                    best_pos[j] = pos[i, j]
    return best_obj


@jit(nopython=True, parallel=True)
def sca_update_parallel(pos, best_pos, lb, ub, dim, ps, r1):
    for i in prange(ps):
        for j in range(dim):
            r2 = 2.0 * np.pi * np.random.random()
            r3 = 2.0 * np.random.random()
            r4 = np.random.random()
            if r4 < 0.5:
                pos[i, j] = pos[i, j] + r1 * np.sin(r2) * np.abs(r3 * best_pos[j] - pos[i, j])
            else:
                pos[i, j] = pos[i, j] + r1 * np.cos(r2) * np.abs(r3 * best_pos[j] - pos[i, j])


@jit(nopython=True, parallel=True)
def sca_boundary_eval_parallel(pos, obj, lb, ub, dim, ps, fid):
    for i in prange(ps):
        for j in range(dim):
            if pos[i, j] > ub: pos[i, j] = ub
            if pos[i, j] < lb: pos[i, j] = lb
        obj[i] = evaluate_numba(pos[i], fid)


@jit(nopython=True)
def sca_numba_parallel(fid, lb, ub, ps, iters, dim):
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    obj = evaluate_batch_parallel(pos, fid)
    best_idx = np.argmin(obj)
    best_pos = pos[best_idx].copy()
    best_obj = obj[best_idx]
    for it in range(iters):
        r1 = 2.0 - 2.0 * (it + 1) / iters
        sca_update_parallel(pos, best_pos, lb, ub, dim, ps, r1)
        sca_boundary_eval_parallel(pos, obj, lb, ub, dim, ps, fid)
        min_idx = np.argmin(obj)
        if obj[min_idx] < best_obj:
            best_obj = obj[min_idx]
            for j in range(dim):
                best_pos[j] = pos[min_idx, j]
    return best_obj

# --------------------------- SSA (Mirjalili 2017) -----------------------------


@jit(nopython=True)
def ssa_numba_serial(fid, lb, ub, ps, iters, dim):
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    obj = np.empty(ps, dtype=np.float64)
    for i in range(ps):
        obj[i] = evaluate_numba(pos[i], fid)
    best_idx = np.argmin(obj)
    food_pos = pos[best_idx].copy()
    best_obj = obj[best_idx]
    for it in range(iters):
        c1 = 2 * np.exp(-((4 * it / iters) ** 2))
        half_ps = ps // 2
        for i in range(ps):
            if i < half_ps:
                for j in range(dim):
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if c3 < 0.5:
                        pos[i, j] = food_pos[j] + c1 * ((ub - lb) * c2 + lb)
                    else:
                        pos[i, j] = food_pos[j] - c1 * ((ub - lb) * c2 + lb)
            else:
                for j in range(dim):
                    pos[i, j] = (pos[i, j] + pos[i -1, j]) / 2.0
        for i in range(ps):
            for j in range(dim):
                if pos[i, j] > ub: pos[i, j] = ub
                if pos[i, j] < lb: pos[i, j] = lb
            obj[i] = evaluate_numba(pos[i], fid)
        for i in range(ps):
            if obj[i] < best_obj:
                best_obj = obj[i]
                for j in range(dim):
                    food_pos[j] = pos[i, j]
    return best_obj


@jit(nopython=True, parallel=True)
def ssa_leader_update_parallel(pos, food_pos, lb, ub, dim, half_ps, c1):
    for i in prange(half_ps):
        for j in range(dim):
            c2 = np.random.random()
            c3 = np.random.random()
            if c3 < 0.5:
                pos[i, j] = food_pos[j] + c1 * ((ub - lb) * c2 + lb)
            else:
                pos[i, j] = food_pos[j] - c1 * ((ub - lb) * c2 + lb)
            if pos[i, j] > ub: pos[i, j] = ub
            if pos[i, j] < lb: pos[i, j] = lb


@jit(nopython=True)
def ssa_numba_parallel(fid, lb, ub, ps, iters, dim):
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    obj = evaluate_batch_parallel(pos, fid)
    best_idx = np.argmin(obj)
    food_pos = pos[best_idx].copy()
    best_obj = obj[best_idx]
    half_ps = ps // 2
    for it in range(iters):
        c1 = 2.0 * np.exp(-((4.0 * it / iters) ** 2))
        ssa_leader_update_parallel(pos, food_pos, lb, ub, dim, half_ps, c1)
        for i in range(half_ps, ps):
            for j in range(dim):
                pos[i, j] = (pos[i, j] + pos[i -1, j]) / 2.0
                if pos[i, j] > ub: pos[i, j] = ub
                if pos[i, j] < lb: pos[i, j] = lb
        obj = evaluate_batch_parallel(pos, fid)
        for i in range(ps):
            if obj[i] < best_obj:
                best_obj = obj[i]
                for j in range(dim):
                    food_pos[j] = pos[i, j]
    return best_obj

# --------------------------- HHO (Heidari 2019) -----------------------------


@jit(nopython=True)
def hho_numba_serial(fid, lb, ub, ps, iters, dim):
    beta = 1.5
    sigma = 0.6966
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    obj = np.empty(ps, dtype=np.float64)
    rabbit_pos = np.empty(dim, dtype=np.float64)
    best_obj = np.float64(np.inf)
    for it in range(iters):
        for i in range(ps):
            for j in range(dim):
                if pos[i, j] > ub: pos[i, j] = ub
                if pos[i, j] < lb: pos[i, j] = lb
            obj[i] = evaluate_numba(pos[i], fid)
            if obj[i] < best_obj:
                best_obj = obj[i]
                for j in range(dim):
                    rabbit_pos[j] = pos[i, j]
        X_mean = np.zeros(dim, dtype=np.float64)
        for i in range(ps):
            for j in range(dim):
                X_mean[j] += pos[i, j]
        for j in range(dim):
            X_mean[j] /= ps
        E1 = 2.0 * (1.0 - (it / iters))
        for i in range(ps):
            E0 = 2.0 * np.random.random() - 1.0
            Escaping_Energy = E1 * E0
            if np.abs(Escaping_Energy) >= 1.0:
                q = np.random.random()
                rand_idx = np.random.randint(0, ps)
                if q < 0.5:
                    for j in range(dim):
                        pos[i, j] = pos[rand_idx, j] - np.random.random() * np.abs(pos[rand_idx, j] - 2.0 * np.random.random() * pos[i, j])
                else:
                    for j in range(dim):
                        pos[i, j] = (rabbit_pos[j] - X_mean[j]) - np.random.random() * ((ub - lb) * np.random.random() + lb)
            else:
                r = np.random.random()
                J = 2.0 * (1.0 - np.random.random())
                if r >= 0.5 and np.abs(Escaping_Energy) < 0.5:
                    for j in range(dim):
                        pos[i, j] = rabbit_pos[j] - Escaping_Energy * np.abs(rabbit_pos[j] - pos[i, j])
                elif r >= 0.5 and np.abs(Escaping_Energy) >= 0.5:
                    for j in range(dim):
                        pos[i, j] = (rabbit_pos[j] - pos[i, j]) - Escaping_Energy * np.abs(J * rabbit_pos[j] - pos[i, j])
                elif r < 0.5 and np.abs(Escaping_Energy) >= 0.5:
                    X1 = np.empty(dim, dtype=np.float64)
                    for j in range(dim):
                        X1[j] = rabbit_pos[j] - Escaping_Energy * np.abs(J * rabbit_pos[j] - pos[i, j])
                        if X1[j] > ub: X1[j] = ub
                        if X1[j] < lb: X1[j] = lb
                    fit1 = evaluate_numba(X1, fid)
                    if fit1 < obj[i]:
                        for j in range(dim):
                            pos[i, j] = X1[j]
                        obj[i] = fit1
                    else:
                        X2 = np.empty(dim, dtype=np.float64)
                        for j in range(dim):
                            u = np.random.randn() * sigma
                            v = np.random.randn()
                            step = u / (np.abs(v) ** (1.0 / beta))
                            X2[j] = rabbit_pos[j] - Escaping_Energy * np.abs(J * rabbit_pos[j] - pos[i, j]) + np.random.random() * step
                            if X2[j] > ub: X2[j] = ub
                            if X2[j] < lb: X2[j] = lb
                        fit2 = evaluate_numba(X2, fid)
                        if fit2 < obj[i]:
                            for j in range(dim):
                                pos[i, j] = X2[j]
                            obj[i] = fit2
                else:
                    X1 = np.empty(dim, dtype=np.float64)
                    for j in range(dim):
                        X1[j] = rabbit_pos[j] - Escaping_Energy * np.abs(J * rabbit_pos[j] - X_mean[j])
                        if X1[j] > ub: X1[j] = ub
                        if X1[j] < lb: X1[j] = lb
                    fit1 = evaluate_numba(X1, fid)
                    if fit1 < obj[i]:
                        for j in range(dim):
                            pos[i, j] = X1[j]
                        obj[i] = fit1
                    else:
                        X2 = np.empty(dim, dtype=np.float64)
                        for j in range(dim):
                            u = np.random.randn() * sigma
                            v = np.random.randn()
                            step = u / (np.abs(v) ** (1.0 / beta))
                            X2[j] = rabbit_pos[j] - Escaping_Energy * np.abs(J * rabbit_pos[j] - X_mean[j]) + np.random.random() * step
                            if X2[j] > ub: X2[j] = ub
                            if X2[j] < lb: X2[j] = lb
                        fit2 = evaluate_numba(X2, fid)
                        if fit2 < obj[i]:
                            for j in range(dim):
                                pos[i, j] = X2[j]
                            obj[i] = fit2
    return best_obj


@jit(nopython=True, parallel=True)
def hho_boundary_eval_parallel(pos, obj, lb, ub, dim, ps, fid):
    for i in prange(ps):
        for j in range(dim):
            if pos[i, j] > ub: pos[i, j] = ub
            if pos[i, j] < lb: pos[i, j] = lb
        obj[i] = evaluate_numba(pos[i], fid)


@jit(nopython=True, parallel=True)
def hho_exploration_parallel(pos, obj, rabbit_pos, X_mean, lb, ub, dim, ps, E1):
    for i in prange(ps):
        E0 = 2.0 * np.random.random() - 1.0
        Escaping_Energy = E1 * E0
        if np.abs(Escaping_Energy) < 1.0:
            continue
        q = np.random.random()
        rand_idx = np.random.randint(0, ps)
        if q < 0.5:
            for j in range(dim):
                pos[i, j] = pos[rand_idx, j] - np.random.random() * np.abs(
                    pos[rand_idx, j] - 2.0 * np.random.random() * pos[i, j])
        else:
            for j in range(dim):
                pos[i, j] = (rabbit_pos[j] - X_mean[j]) - np.random.random() * (
                    (ub - lb) * np.random.random() + lb)


@jit(nopython=True)
def hho_numba_parallel(fid, lb, ub, ps, iters, dim):
    sigma = 0.6966
    beta = 1.5
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    obj = np.empty(ps, dtype=np.float64)
    rabbit_pos = np.empty(dim, dtype=np.float64)
    best_obj = np.float64(np.inf)
    for it in range(iters):
        hho_boundary_eval_parallel(pos, obj, lb, ub, dim, ps, fid)
        for i in range(ps):
            if obj[i] < best_obj:
                best_obj = obj[i]
                for j in range(dim):
                    rabbit_pos[j] = pos[i, j]
        X_mean = np.zeros(dim, dtype=np.float64)
        for i in range(ps):
            for j in range(dim):
                X_mean[j] += pos[i, j]
        for j in range(dim):
            X_mean[j] /= ps
        E1 = 2.0 * (1.0 - (it / iters))
        hho_exploration_parallel(pos, obj, rabbit_pos, X_mean, lb, ub, dim, ps, E1)
        for i in range(ps):
            E0 = 2.0 * np.random.random() - 1.0
            Escaping_Energy = E1 * E0
            if np.abs(Escaping_Energy) >= 1.0:
                continue
            r = np.random.random()
            J = 2.0 * (1.0 - np.random.random())
            if r >= 0.5 and np.abs(Escaping_Energy) < 0.5:
                for j in range(dim):
                    pos[i, j] = rabbit_pos[j] - Escaping_Energy * np.abs(rabbit_pos[j] - pos[i, j])
            elif r >= 0.5 and np.abs(Escaping_Energy) >= 0.5:
                for j in range(dim):
                    pos[i, j] = (rabbit_pos[j] - pos[i, j]) - Escaping_Energy * np.abs(
                        J * rabbit_pos[j] - pos[i, j])
            elif r < 0.5 and np.abs(Escaping_Energy) >= 0.5:
                X1 = np.empty(dim, dtype=np.float64)
                for j in range(dim):
                    X1[j] = rabbit_pos[j] - Escaping_Energy * np.abs(J * rabbit_pos[j] - pos[i, j])
                    if X1[j] > ub: X1[j] = ub
                    if X1[j] < lb: X1[j] = lb
                fit1 = evaluate_numba(X1, fid)
                if fit1 < obj[i]:
                    for j in range(dim):
                        pos[i, j] = X1[j]
                    obj[i] = fit1
                else:
                    X2 = np.empty(dim, dtype=np.float64)
                    for j in range(dim):
                        u = np.random.randn() * sigma
                        v = np.random.randn()
                        step = u / (np.abs(v) ** (1.0 / beta))
                        X2[j] = rabbit_pos[j] - Escaping_Energy * np.abs(
                            J * rabbit_pos[j] - pos[i, j]) + np.random.random() * step
                        if X2[j] > ub: X2[j] = ub
                        if X2[j] < lb: X2[j] = lb
                    fit2 = evaluate_numba(X2, fid)
                    if fit2 < obj[i]:
                        for j in range(dim):
                            pos[i, j] = X2[j]
                        obj[i] = fit2
            else:
                X1 = np.empty(dim, dtype=np.float64)
                for j in range(dim):
                    X1[j] = rabbit_pos[j] - Escaping_Energy * np.abs(J * rabbit_pos[j] - X_mean[j])
                    if X1[j] > ub: X1[j] = ub
                    if X1[j] < lb: X1[j] = lb
                fit1 = evaluate_numba(X1, fid)
                if fit1 < obj[i]:
                    for j in range(dim):
                        pos[i, j] = X1[j]
                    obj[i] = fit1
                else:
                    X2 = np.empty(dim, dtype=np.float64)
                    for j in range(dim):
                        u = np.random.randn() * sigma
                        v = np.random.randn()
                        step = u / (np.abs(v) ** (1.0 / beta))
                        X2[j] = rabbit_pos[j] - Escaping_Energy * np.abs(
                            J * rabbit_pos[j] - X_mean[j]) + np.random.random() * step
                        if X2[j] > ub: X2[j] = ub
                        if X2[j] < lb: X2[j] = lb
                    fit2 = evaluate_numba(X2, fid)
                    if fit2 < obj[i]:
                        for j in range(dim):
                            pos[i, j] = X2[j]
                        obj[i] = fit2
    return best_obj

# ------------------ Sphere with a scalable evaluation cost -------------------


@jit(nopython=True)
def sphere_expensive_numba(x, cost):
    s = 0.0
    for r in range(cost):
        t = 0.0
        for i in range(len(x)):
            t += x[i] * x[i] + 1e-15 * np.sin((r + 1) * x[i])
        s = t
    return s


@jit(nopython=True)
def gwo_serial_expensive(lb, ub, ps, iters, dim, cost, seed):
    """Serial GWO on the cost-scalable sphere (GPU counterpart: fid 60)."""
    np.random.seed(seed)
    alpha_pos = np.zeros(dim, dtype=np.float64); alpha_score = np.float64(np.inf)
    beta_pos = np.zeros(dim, dtype=np.float64); beta_score = np.float64(np.inf)
    delta_pos = np.zeros(dim, dtype=np.float64); delta_score = np.float64(np.inf)
    pos = np.empty((ps, dim), dtype=np.float64)
    for i in range(ps):
        for j in range(dim):
            pos[i, j] = lb + np.random.random() * (ub - lb)
    for it in range(iters):
        for i in range(ps):
            for j in range(dim):
                if pos[i, j] > ub: pos[i, j] = ub
                if pos[i, j] < lb: pos[i, j] = lb
            fitness = sphere_expensive_numba(pos[i], cost)
            if fitness < alpha_score:
                delta_score = beta_score
                for j in range(dim): delta_pos[j] = beta_pos[j]
                beta_score = alpha_score
                for j in range(dim): beta_pos[j] = alpha_pos[j]
                alpha_score = fitness
                for j in range(dim): alpha_pos[j] = pos[i, j]
            elif fitness < beta_score:
                delta_score = beta_score
                for j in range(dim): delta_pos[j] = beta_pos[j]
                beta_score = fitness
                for j in range(dim): beta_pos[j] = pos[i, j]
            elif fitness < delta_score:
                delta_score = fitness
                for j in range(dim): delta_pos[j] = pos[i, j]
        a = 2.0 - float(it) * (2.0 / iters)
        for i in range(ps):
            for j in range(dim):
                r1 = np.random.random(); r2 = np.random.random()
                A1 = 2.0 * a * r1 - a; C1 = 2.0 * r2
                X1 = alpha_pos[j] - A1 * np.abs(C1 * alpha_pos[j] - pos[i, j])
                r1 = np.random.random(); r2 = np.random.random()
                A2 = 2.0 * a * r1 - a; C2 = 2.0 * r2
                X2 = beta_pos[j] - A2 * np.abs(C2 * beta_pos[j] - pos[i, j])
                r1 = np.random.random(); r2 = np.random.random()
                A3 = 2.0 * a * r1 - a; C3 = 2.0 * r2
                X3 = delta_pos[j] - A3 * np.abs(C3 * delta_pos[j] - pos[i, j])
                pos[i, j] = (X1 + X2 + X3) / 3.0
    return alpha_score


CPU_SERIAL_FUNCS = {
    'ABC': abc_numba_serial, 'GWO': gwo_numba_serial, 'WOA': woa_numba_serial,
    'SCA': sca_numba_serial, 'SSA': ssa_numba_serial, 'HHO': hho_numba_serial
}
CPU_PARALLEL_FUNCS = {
    'ABC': abc_numba_parallel, 'GWO': gwo_numba_parallel, 'WOA': woa_numba_parallel,
    'SCA': sca_numba_parallel, 'SSA': ssa_numba_parallel, 'HHO': hho_numba_parallel
}


def run_cpu_timed(func, fid, lb, ub, ps, iters, dim):
    start = time.time()
    best = func(fid, np.float64(lb), np.float64(ub), ps, iters, dim)
    return float(best), (time.time() - start) * 1000


_warm = False


def warmup():
    """Compile every implementation once so that timings exclude the JIT."""
    global _warm
    if _warm:
        return
    for alg in CPU_SERIAL_FUNCS:
        CPU_SERIAL_FUNCS[alg](0, -100.0, 100.0, 8, 3, 5)
        CPU_PARALLEL_FUNCS[alg](0, -100.0, 100.0, 8, 3, 5)
    gwo_serial_expensive(-100.0, 100.0, 8, 3, 5, 1, 1)
    _warm = True
