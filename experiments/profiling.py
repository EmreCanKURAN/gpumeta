"""Low-level inspection of the selection kernels.

1. SASS: both kernel files are compiled with nvcc and the generated
   instructions are counted per kernel (shuffles, votes, branches, divergence
   barriers, predicated instructions, global loads and stores).
2. compute-sanitizer: synccheck, racecheck and memcheck on a small driver
   that launches every selection kernel once, for a population of 1000 (not a
   multiple of 32) and of 1024.
3. Nsight Compute: branch uniformity, active threads per instruction, global
   memory traffic and kernel duration at N = 8192.
4. Register and local-memory usage of the compiled kernels.

Steps 1 to 3 need the CUDA toolkit (nvcc, cuobjdump, compute-sanitizer, ncu).
Tools that are not found are skipped. Nsight Compute needs access to the GPU
performance counters, which shared machines often do not grant; the driver and
the command line are written to results/profiling so that the step can be
repeated on a local machine.

Outputs: profiling_sass_instruction_counts.csv, profiling_compute_sanitizer.csv,
profiling_ncu_base.csv, profiling_ncu_safe.csv, profiling_register_usage.csv
and the raw tool outputs in profiling/
"""
import io
import os
import re
import shutil
import subprocess

import pandas as pd

from gpumeta import gpu
from gpumeta.config import CFG, RESULTS_DIR, banner, save_csv
from gpumeta.gpu import BASE_SRC, CC, SELECTION_KERNELS, SELECTION_SRC, SELK

# Stand-alone script that launches every selection kernel once. It is run in
# a separate process under compute-sanitizer and Nsight Compute.
DRIVER_SRC = r'''
import sys, argparse, numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
ap = argparse.ArgumentParser()
ap.add_argument('--set', default='safe', choices=['base', 'safe']); ap.add_argument('--n', type=int, default=8192)
ap.add_argument('--base', required=True); ap.add_argument('--safe', required=True)
a = ap.parse_args()
n = a.n; BS = 256; g = ((n + BS - 1) // BS, 1)
rng = np.random.RandomState(1)
obj = rng.uniform(1, 2, n); fit = 1.0 / (1.0 + obj)
d_obj = gpuarray.to_gpu(obj); d_fit = gpuarray.to_gpu(fit)
d_sel = gpuarray.zeros(n, np.int32); d_avg = gpuarray.to_gpu(np.array([fit.mean()]))
d_bi = gpuarray.zeros(1, np.int32); d_bv = gpuarray.zeros(1, np.float64)
d_cum = gpuarray.to_gpu(np.cumsum(fit))
s = np.uint32(12345); ts = np.int32(3); al = np.float64(1.2); N = np.int32(n)
if a.set == 'base':
    m = SourceModule(open(a.base).read())
    K = lambda k: m.get_function(k)
    calls = [
        lambda: K('sel_tournament')(d_obj, d_sel, N, ts, s, block=(BS,1,1), grid=g),
        lambda: K('sel_tournament_global')(d_obj, d_sel, N, ts, s, block=(BS,1,1), grid=g),
        lambda: K('sel_tournament_elitist')(d_obj, d_sel, N, ts, np.int32(0), s, block=(BS,1,1), grid=g),
        lambda: K('sel_ballot_elite')(d_fit, d_sel, d_avg, N, ts, al, s, block=(BS,1,1), grid=g),
        lambda: K('sel_ballot_local')(d_fit, d_sel, N, ts, al, s, block=(BS,1,1), grid=g),
        lambda: K('sel_ballot_noswitch')(d_fit, d_sel, d_avg, N, ts, al, s, block=(BS,1,1), grid=g),
        lambda: K('compute_avg_fitness')(d_fit, d_avg, N, block=(BS,1,1), grid=g, shared=BS*8),
    ]
else:
    m = SourceModule(open(a.safe).read())
    K = lambda k: m.get_function(k)
    calls = [
        lambda: K('sel_tournament_safe')(d_obj, d_sel, N, ts, s, block=(BS,1,1), grid=g),
        lambda: K('sel_ballot_elite_safe')(d_fit, d_sel, d_avg, N, ts, al, s, block=(BS,1,1), grid=g),
        lambda: K('sel_ballot_local_safe')(d_fit, d_sel, N, ts, al, s, block=(BS,1,1), grid=g),
        lambda: K('sel_ballot_noswitch_safe')(d_fit, d_sel, d_avg, N, ts, al, s, block=(BS,1,1), grid=g),
        lambda: K('sel_tournament_elitist_dev')(d_obj, d_sel, N, ts, d_bi, s, block=(BS,1,1), grid=g),
        lambda: K('sel_globalbest')(d_sel, N, d_bi, block=(BS,1,1), grid=g),
        lambda: K('sel_tournament_migrate')(d_obj, d_sel, N, ts, d_bi, np.float64(0.25), s, block=(BS,1,1), grid=g),
        lambda: K('argmin_block')(d_obj, N, d_bi, d_bv, block=(1024,1,1), grid=(1,1)),
        lambda: K('roulette_sample')(d_cum, d_sel, N, s, block=(BS,1,1), grid=g),
        lambda: K('sel_tournament_shared')(d_obj, d_sel, N, ts, s, block=(BS,1,1), grid=g, shared=BS*8),
    ]
for c in calls:
    c()
cuda.Context.synchronize()
print("DRIVER_OK", a.set, n)
'''

NCU_METRICS = ','.join([
    'smsp__sass_average_branch_targets_threads_uniform.pct',
    'smsp__sass_branch_targets_threads_divergent.sum',
    'smsp__sass_branch_targets.sum',
    'smsp__thread_inst_executed_per_inst_executed.ratio',
    'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
    'dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum'])


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    for c in [f'/usr/local/cuda/bin/{name}', f'/usr/local/cuda/nsight-compute/{name}']:
        if os.path.exists(c):
            return c
    for root in ['/usr/local', '/opt']:
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    return None


def run_tools():
    banner("SASS inspection, compute-sanitizer and Nsight Compute")
    d = os.path.join(RESULTS_DIR, 'profiling'); os.makedirs(d, exist_ok=True)
    base_cu = os.path.join(d, 'base_kernels.cu'); safe_cu = os.path.join(d, 'safe_kernels.cu')
    with open(base_cu, 'w') as f:
        f.write('extern "C" {\n' + BASE_SRC + '\n}\n')
    with open(safe_cu, 'w') as f:
        f.write('extern "C" {\n' + SELECTION_SRC + '\n}\n')
    drv = os.path.join(d, 'profile_driver.py')
    with open(drv, 'w') as f:
        f.write(DRIVER_SRC)
    arch = f"sm_{CC[0]}{CC[1]}"
    nvcc = find_tool('nvcc'); cuobj = find_tool('cuobjdump')
    sel_kernels = ['sel_tournament', 'sel_tournament_elitist', 'sel_ballot_elite', 'sel_ballot_local',
                   'sel_ballot_noswitch', 'sel_tournament_global'] + SELECTION_KERNELS
    sass_rows = []
    if nvcc and cuobj:
        for label, cu in [('base', base_cu), ('safe', safe_cu)]:
            cub = cu.replace('.cu', '.cubin')
            r = subprocess.run([nvcc, '-cubin', f'-arch={arch}', '-O3', '-o', cub, cu], capture_output=True,
                               text=True)
            if r.returncode != 0:
                print(f"  nvcc failed for {label}: {r.stderr[:500]}"); continue
            sass = subprocess.run([cuobj, '-sass', cub], capture_output=True, text=True).stdout
            with open(cu.replace('.cu', '.sass.txt'), 'w') as f:
                f.write(sass)
            blocks = re.split(r'\n\s*Function : ', sass)
            for blk in blocks[1:]:
                name = blk.split('\n', 1)[0].strip()
                if name not in sel_kernels:
                    continue
                ins = [l for l in blk.split('\n') if re.match(r'\s*/\*[0-9a-f]{4}\*/', l)]
                txt = '\n'.join(ins)
                cnt = lambda pat: len(re.findall(pat, txt))
                sass_rows.append({'module': label, 'kernel': name, 'instructions': len(ins),
                                  'SHFL': cnt(r'\bSHFL\.'), 'VOTE': cnt(r'\bVOTE\.'),
                                  'BRA': cnt(r'\bBRA\b'),
                                  'BRA_on_uniform_predicate': len([l for l in ins if 'BRA' in l and 'UP' in l]),
                                  'BSSY_divergence_barrier': cnt(r'\bBSSY\b'), 'BSYNC': cnt(r'\bBSYNC\b'),
                                  'WARPSYNC': cnt(r'\bWARPSYNC\b'), 'predicated_instr': cnt(r'@!?P\d'),
                                  'SEL_FSEL': cnt(r'\b(F?SEL)\b'), 'LDG': cnt(r'\bLDG\.'), 'STG': cnt(r'\bSTG\.'),
                                  'LDS_STS': cnt(r'\b(LDS|STS)\b'), 'EXIT': cnt(r'\bEXIT\b')})
        save_csv(sass_rows, 'profiling_sass_instruction_counts')
    else:
        print("  nvcc or cuobjdump not found: SASS step skipped")
    # compute-sanitizer: synccheck (misuse of warp collectives), racecheck, memcheck
    cs = find_tool('compute-sanitizer')
    cs_rows = []
    if cs:
        tools = ['synccheck'] if CFG['quick'] else ['synccheck', 'racecheck', 'memcheck']
        nlist = [1000] if CFG['quick'] else [1000, 1024]
        for tool in tools:
            for kset in ['base', 'safe']:
                for n in nlist:
                    r = subprocess.run([cs, '--tool', tool, 'python3', drv, '--set', kset, '--n', str(n),
                                        '--base', base_cu, '--safe', safe_cu], capture_output=True, text=True,
                                       timeout=1800)
                    txt = r.stdout + r.stderr
                    m = (re.search(r'ERROR SUMMARY:\s*(\d+)', txt)
                         or re.search(r'RACECHECK SUMMARY:.*?\((\d+) errors', txt))
                    cs_rows.append({'tool': tool, 'kernels': kset, 'n': n,
                                    'errors': int(m.group(1)) if m else None,
                                    'driver_ok': 'DRIVER_OK' in txt})
                    with open(os.path.join(d, f'sanitizer_{tool}_{kset}_{n}.txt'), 'w') as f:
                        f.write(txt)
        save_csv(cs_rows, 'profiling_compute_sanitizer')
        print(pd.DataFrame(cs_rows).to_string(index=False))
    else:
        print("  compute-sanitizer not found: skipped")
    ncu = find_tool('ncu')
    if ncu:
        for kset in ['base', 'safe']:
            cmd = [ncu, '--csv', '--page', 'raw', '--metrics', NCU_METRICS, 'python3', drv, '--set', kset,
                   '--n', '8192', '--base', base_cu, '--safe', safe_cu]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            txt = r.stdout
            with open(os.path.join(d, f'ncu_{kset}_raw.txt'), 'w') as f:
                f.write(r.stdout + '\n--- stderr ---\n' + r.stderr)
            lines = [l for l in txt.split('\n') if l.startswith('"')]
            if len(lines) > 2:
                try:
                    df = pd.read_csv(io.StringIO('\n'.join(lines)))
                    df = df.iloc[1:]
                    df.to_csv(os.path.join(RESULTS_DIR, f'profiling_ncu_{kset}.csv'), index=False)
                    keep = [c for c in df.columns if c in ('Kernel Name',) or c in NCU_METRICS.split(',')]
                    print(df[keep].to_string(index=False))
                except Exception as ex:
                    print(f"  ncu CSV parse failed: {ex}")
            else:
                print(f"  ncu produced no metrics for '{kset}' (GPU performance counters are often blocked on "
                      f"shared machines, ERR_NVGPUCTRPERM). Run locally:\n    {' '.join(cmd)}")
    else:
        print("  ncu not found: run the driver with Nsight Compute on a local machine (see profiling/README.txt)")
    with open(os.path.join(d, 'README.txt'), 'w') as f:
        f.write("Nsight Compute (needs GPU counter access):\n"
                f"ncu --csv --page raw --metrics {NCU_METRICS} python profile_driver.py --set safe "
                "--n 8192 --base base_kernels.cu --safe safe_kernels.cu > ncu_safe.csv\n"
                "compute-sanitizer --tool synccheck python profile_driver.py --set base --n 1000 "
                "--base base_kernels.cu --safe safe_kernels.cu\n")


def register_usage():
    """Registers per thread, local (spill) memory and shared memory of the
    compiled selection kernels, as reported by the CUDA driver."""
    kernels = [
        ('tournament', gpu.k_sel_tournament),
        ('tournament_global', gpu.k_sel_tourn_global),
        ('tournament_elitist', gpu.k_sel_tourn_elite),
        ('ballot', gpu.k_sel_ballot),
        ('ballot_local', gpu.k_sel_ballot_local),
        ('ballot_noswitch', gpu.k_sel_ballot_nosw),
        ('naive_cumulative_sum', gpu.k_sel_naive),
        ('abc_onlooker_roulette', gpu.k_abc_onl_roul),
    ] + [(name, SELK[name]) for name in SELECTION_KERNELS]
    rows = [{'kernel': name, 'registers_per_thread': k.num_regs, 'local_bytes': k.local_size_bytes,
             'shared_bytes': k.shared_size_bytes} for name, k in kernels]
    print(save_csv(rows, 'profiling_register_usage').to_string(index=False))


def run():
    run_tools()
    register_usage()


if __name__ == '__main__':
    run()
