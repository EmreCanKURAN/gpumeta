"""CEC2017 and CEC2020 experiments in a host loop.

The fitness is evaluated on the CPU with the official shift and rotation data
(opfunu); the selection indices come from the CPU reference implementation of
the warp kernels (gpumeta.reference). The coupling between selection and
update is the same as in the GPU kernels:

    WOA, SCA, SSA, HHO : individual i is guided by pos[sel[i]]; naive: argmin
    GWO                : leaders are the three best distinct members of the
                         selected multiset; naive: the global top three
    ABC                : onlooker i exploits source sel[i]; naive: roulette

No GPU is needed for this module, so the runs of one function are distributed
over CPU processes.
"""
import multiprocessing as mp
import time

import numpy as np

from .config import CEC_DIM, CEC_POP, N_WORKERS, checkpoint, make_seeds, save_checkpoint
from .reference import emu_fit, emu_select, leaders_from_selection


def check_opfunu():
    """The CEC results require opfunu 1.0.4. Version 1.0.1, which pip selects on
    Python 3.12 because later versions declare Python < 3.12, has a sign error
    in the modified Schwefel function and wrong mixing proportions in two
    CEC2017 hybrid functions."""
    import opfunu
    v = tuple(int(x) for x in opfunu.__version__.split('.')[:3])
    if v < (1, 0, 4):
        raise RuntimeError(f"opfunu {opfunu.__version__} is installed, but version 1.0.4 is required:\n"
                           "  pip install --no-deps --ignore-requires-python opfunu==1.0.4")


def make_cec_fitness(suite, fnum, dim):
    """Return (fitness, f_opt). The optimiser searches the unit cube, which
    fitness maps to the box [lb, ub] of the function."""
    if suite == 'cec2017':
        from opfunu.cec_based import cec2017 as mod_cec
        cls = getattr(mod_cec, f"F{fnum}2017")
    elif suite == 'cec2020':
        from opfunu.cec_based import cec2020 as mod_cec
        cls = getattr(mod_cec, f"F{fnum}2020")
    else:
        raise ValueError(suite)
    fobj = cls(ndim=dim)
    lb = np.asarray(fobj.lb, dtype=np.float64)
    ub = np.asarray(fobj.ub, dtype=np.float64)
    f_opt = float(fobj.f_global)

    def fitness(v):
        v = np.clip(v, 0.0, 1.0)
        x = lb + v * (ub - lb)
        return float(fobj.evaluate(x))
    return fitness, f_opt


CEC_METHODS = ['naive', 'tournament', 'mig_0.03125', 'mig_0.25', 'ballot']
_fitness_cache = {}


def cec_methods(alg, suite, fnum):
    m = list(CEC_METHODS)
    if alg in ('GWO', 'ABC') or (suite == 'cec2017' and fnum in (1, 4, 10)):
        m.append('elitist')          # for single-leader algorithms this equals naive (identity check)
    return m


def cec_fitness(suite, fnum, dim):
    k = (suite, fnum, dim)
    if k not in _fitness_cache:
        _fitness_cache[k] = make_cec_fitness(suite, fnum, dim)
    return _fitness_cache[k]


def abc_sources(variant, obj, pos, seed, it, rng):
    """Onlooker sources of ABC for a selection mechanism or coupling variant."""
    ps = len(obj)
    sseed = (int(seed) * 131 + it) & 0xFFFFFFFF
    if variant in ('naive', 'roulette'):
        fit = emu_fit(obj)
        return rng.choice(ps, ps, p=fit / fit.sum())
    if variant == 'rank_order':
        return np.argsort(obj, kind='stable')
    direct = {'tournament': 'tournament', 'warp_direct': 'tournament', 'global4_direct': 'global4',
              'elitist': 'elitist', 'ballot': 'ballot', 'mig_0.03125': 'mig_0.03125', 'mig_0.25': 'mig_0.25'}
    if variant in direct:
        return emu_select(direct[variant], obj, sseed)
    if variant.startswith('orig_'):
        # original coupling: distinct winners, filled up with the best individuals,
        # then assigned in fitness order
        src_kind = variant[len('orig_'):]
        if src_kind == 'random':
            nw = len(np.unique(emu_select('tournament', obj, sseed)))
            winners = np.sort(rng.choice(ps, nw, replace=False))
        else:
            winners = np.unique(emu_select({'warp': 'tournament', 'elitist': 'elitist',
                                            'global4': 'global4'}[src_kind], obj, sseed))
        if len(winners) >= ps:
            guides = rng.choice(winners, ps, replace=False)
        else:
            guides = np.concatenate([winners, np.argsort(obj)[:ps]])[:ps]
        return guides[np.argsort(obj[guides])]
    raise ValueError(variant)


def cec_host_run(alg, fitness, dim, ps, iters, seed, method, track=False):
    rng = np.random.RandomState(seed)
    pos = rng.rand(ps, dim)
    obj = np.array([fitness(p) for p in pos])
    best = float(obj.min())
    conv, nlead = [], []
    trial = np.zeros(ps, dtype=np.int64) if alg == 'ABC' else None
    rows_idx = np.arange(ps)
    for it in range(iters):
        a = 2.0 - 2.0 * it / iters
        sseed = (int(seed) * 131 + it) & 0xFFFFFFFF
        if alg != 'ABC':
            if method == 'naive':
                sel = None
                gb = int(np.argmin(obj))
            else:
                sel = emu_select(method, obj, sseed)
        if alg == 'GWO':
            if sel is None:
                ld = np.argsort(obj)[:3]; nlead.append(3)
            else:
                ld, nd = leaders_from_selection(sel, obj); nlead.append(nd)
            A, B, D = pos[ld[0]], pos[ld[1]], pos[ld[2]]
            r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
            X1 = A - (2 * a * r1 - a) * np.abs(2 * r2 * A - pos)
            r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
            X2 = B - (2 * a * r1 - a) * np.abs(2 * r2 * B - pos)
            r1 = rng.rand(ps, dim); r2 = rng.rand(ps, dim)
            X3 = D - (2 * a * r1 - a) * np.abs(2 * r2 * D - pos)
            pos = np.clip((X1 + X2 + X3) / 3.0, 0.0, 1.0)
        elif alg == 'WOA':
            L = pos[gb][None, :] if sel is None else pos[sel]
            a2 = -1.0 - it / iters
            r1 = rng.rand(ps, 1); r2 = rng.rand(ps, 1)
            Acoef = 2 * a * r1 - a; Ccoef = 2 * r2
            p = rng.rand(ps, 1); l = (a2 - 1.0) * rng.rand(ps, 1) + 1.0
            rand_idx = rng.randint(0, ps, ps)
            Xrand = pos[rand_idx]
            spiral = np.abs(L - pos) * np.exp(l) * np.cos(2 * np.pi * l) + L
            enc = np.where(np.abs(Acoef) >= 1.0, Xrand - Acoef * np.abs(Ccoef * Xrand - pos),
                           L - Acoef * np.abs(Ccoef * L - pos))
            pos = np.clip(np.where(p < 0.5, enc, spiral), 0.0, 1.0)
        elif alg == 'SCA':
            G = pos[gb][None, :] if sel is None else pos[sel]
            r1c = 2.0 - 2.0 * (it + 1) / iters
            r2 = 2 * np.pi * rng.rand(ps, dim); r3 = 2 * rng.rand(ps, dim); r4 = rng.rand(ps, dim)
            upd = pos + r1c * np.where(r4 < 0.5, np.sin(r2), np.cos(r2)) * np.abs(r3 * G - pos)
            pos = np.clip(upd, 0.0, 1.0)
        elif alg == 'SSA':
            half = ps // 2
            F = pos[gb][None, :] if sel is None else pos[sel[:half]]
            c1 = 2.0 * np.exp(-((4.0 * it / iters) ** 2))
            new = pos.copy()
            c2 = rng.rand(half, dim); c3 = rng.rand(half, dim)
            new[:half] = np.where(c3 < 0.5, F + c1 * c2, F - c1 * c2)
            new[half:] = (pos[half:] + pos[half - 1:ps - 1]) / 2.0
            pos = np.clip(new, 0.0, 1.0)
        elif alg == 'HHO':
            R = pos[gb][None, :] if sel is None else pos[sel]
            Xm = pos.mean(axis=0)
            E1 = 2.0 * (1.0 - it / iters)
            E = E1 * (2 * rng.rand(ps, 1) - 1)
            J = 2 * (1 - rng.rand(ps, 1))
            q = rng.rand(ps, 1)
            ridx = rng.randint(0, ps, ps)
            explore = np.where(q < 0.5, pos[ridx] - rng.rand(ps, 1) * np.abs(pos[ridx] - 2 * rng.rand(ps, 1) * pos),
                               (R - Xm) - rng.rand(ps, 1) * rng.rand(ps, 1))
            soft = R - E * np.abs(J * R - pos)
            hard = R - E * np.abs(R - pos)
            exploit = np.where(np.abs(E) >= 0.5, soft, hard)
            pos = np.clip(np.where(np.abs(E) >= 1.0, explore, exploit), 0.0, 1.0)
        elif alg == 'ABC':
            k_idx = rng.randint(0, ps, ps); j_idx = rng.randint(0, dim, ps); phi = -1 + 2 * rng.rand(ps)
            cand = pos.copy()
            cand[rows_idx, j_idx] = pos[rows_idx, j_idx] + phi * (pos[rows_idx, j_idx] - pos[k_idx, j_idx])
            cand = np.clip(cand, 0.0, 1.0)
            cobj = np.array([fitness(c) for c in cand])
            improved = cobj < obj
            pos[improved] = cand[improved]; obj[improved] = cobj[improved]
            trial[improved] = 0; trial[~improved] += 1
            src = np.asarray(abc_sources(method, obj, pos, seed, it, rng), dtype=np.int64)
            k2 = rng.randint(0, ps, ps); j2 = rng.randint(0, dim, ps); phi2 = -1 + 2 * rng.rand(ps)
            cand2 = pos[src].copy()
            cand2[rows_idx, j2] = pos[src, j2] + phi2 * (pos[src, j2] - pos[k2, j2])
            cand2 = np.clip(cand2, 0.0, 1.0)
            cobj2 = np.array([fitness(c) for c in cand2])
            better = cobj2 < obj
            pos[better] = cand2[better]; obj[better] = cobj2[better]
            trial[better] = 0
            scout = trial > ps * dim
            if scout.any():
                pos[scout] = rng.rand(int(scout.sum()), dim)
                obj[scout] = np.array([fitness(pp) for pp in pos[scout]])
                trial[scout] = 0
        if alg != 'ABC':
            obj = np.array([fitness(pp) for pp in pos])
        bi = int(np.argmin(obj))
        if obj[bi] < best:
            best = float(obj[bi])
        if track and it % 10 == 0:
            conv.append((it, best))
    return best, conv, (float(np.mean(nlead)) if nlead else float('nan'))


def _cec_task(t):
    suite, fnum, dim, alg, method, seed, ps, iters, track = t
    fitness, f_opt = cec_fitness(suite, fnum, dim)
    best, conv, nl = cec_host_run(alg, fitness, dim, ps, iters, seed, method, track)
    return (suite, fnum, alg, method, seed, best - f_opt, [(i, v - f_opt) for i, v in conv], nl)


def _worker_init():
    import gc
    gc.disable()   # workers are CPU-only and must never release CUDA objects of the parent


def _pool_map(tasks):
    if N_WORKERS <= 1:
        return [_cec_task(t) for t in tasks]
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        return [_cec_task(t) for t in tasks]
    import gc
    gc.collect()
    gc.freeze()    # keep parent objects (e.g. CUDA handles) out of the children's collector
    try:
        with ctx.Pool(N_WORKERS, initializer=_worker_init, maxtasksperchild=50) as pool:
            return pool.map(_cec_task, tasks, chunksize=1)
    finally:
        gc.unfreeze()


def cec_experiment(tag, suites, algs, methods_fn, runs, iters):
    ck = checkpoint.setdefault(tag, {})
    seeds = make_seeds(runs, offset=400)
    dim = CEC_DIM; ps = CEC_POP
    t0 = time.time(); done = 0
    total = sum(len(f) for _, f in suites)
    for suite, fnums in suites:
        for fnum in fnums:
            todo_keys = [f"{suite}_{dim}_F{fnum}_{alg}" for alg in algs]
            if all(k in ck for k in todo_keys):
                done += 1; continue
            tasks = []
            for alg in algs:
                if f"{suite}_{dim}_F{fnum}_{alg}" in ck:
                    continue
                for m in methods_fn(alg, suite, fnum):
                    for ri, sd in enumerate(seeds):
                        tasks.append((suite, fnum, dim, alg, m, sd, ps, iters, ri == 0))
            tb = time.time()
            res = _pool_map(tasks)
            for suite_, fnum_, alg, m, sd, err, conv, nl in res:
                key = f"{suite_}_{dim}_F{fnum_}_{alg}"
                d = ck.setdefault(key, {}).setdefault(m, {'errs': [None] * len(seeds), 'conv': None,
                                                          'n_leaders': [None] * len(seeds)})
                i = seeds.index(sd)
                d['errs'][i] = float(err); d['n_leaders'][i] = nl
                if conv:
                    d['conv'] = conv
            save_checkpoint()
            done += 1
            el = time.time() - t0
            print(f"  [{tag}] {suite} F{fnum}: {len(tasks)} runs in {time.time() - tb:.0f}s | "
                  f"{done}/{total} functions | ETA {el / max(1, done) * (total - done) / 3600:.1f} h", flush=True)
    return ck
