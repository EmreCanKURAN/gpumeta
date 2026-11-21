"""Number of objective function evaluations per run under a fixed iteration
budget, counted on the device for naive, Tournament and Ballot-Elite
selection.

Output: function_evaluations.csv
"""
import numpy as np
import pycuda.driver as cuda

from gpumeta.benchmarks import BENCHMARKS
from gpumeta.config import ALGORITHMS, CFG, banner, checkpoint, make_seeds, save_checkpoint, save_csv
from gpumeta.gpu import BASE_SRC, GPU_FUNCS, build_module, counting_source, use_kernels


def run():
    banner("Function evaluations per run (fixed iteration budget)")
    mod, h = build_module(counting_source(BASE_SRC), 'evaluation counter')
    ptr = mod.get_global('d_fe_count')[0]
    ck = checkpoint.setdefault('function_evaluations', {})
    seeds = make_seeds(CFG['fe_seeds'], offset=800)
    ps, it, dim = 1024, CFG['fe_iters'], CFG['fe_dim']
    rows = []
    with use_kernels(h):
        for alg in ALGORITHMS:
            for fname in CFG['fe_funcs']:
                key = f"{alg}_{fname}"
                fid = BENCHMARKS[fname]['id']; bnd = BENCHMARKS[fname]['bounds']
                if key not in ck:
                    ck[key] = {}
                    for m in ['naive', 'tournament', 'ballot']:
                        ck[key][m] = []
                        for sd in seeds:
                            cuda.memcpy_htod(ptr, np.zeros(1, np.uint64))
                            GPU_FUNCS[alg](ps, it, fid, bnd, dim=dim, method=m, seed=sd)
                            cuda.Context.synchronize()
                            c = np.zeros(1, np.uint64); cuda.memcpy_dtoh(c, ptr)
                            ck[key][m].append(int(c[0]))
                    save_checkpoint()
                for m, v in ck[key].items():
                    for i, c in enumerate(v):
                        rows.append({'algorithm': alg, 'function': fname, 'method': m, 'run': i, 'ps': ps,
                                     'iters': it, 'fe': c, 'fe_per_ps_iter': c / (ps * (it + 1))})
    df = save_csv(rows, 'function_evaluations')
    print(df.groupby(['algorithm', 'method'])['fe'].agg(['mean', 'std', 'min', 'max']).to_string())


if __name__ == '__main__':
    run()
