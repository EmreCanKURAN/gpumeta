"""Print the hardware and software configuration of the machine."""
import platform
import re
import subprocess


def _smi(field):
    return subprocess.check_output(['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader,nounits'],
                                   text=True).strip().split('\n')[0]


def _proc(path, pattern):
    try:
        m = re.search(pattern, open(path).read())
        return m.group(1).strip() if m else 'n/a'
    except OSError:
        return 'n/a'


def main():
    rows = []
    try:
        rows += [('GPU', _smi('name')),
                 ('GPU memory', f"{int(_smi('memory.total')) / 1024:.0f} GB"),
                 ('Power limit', f"{float(_smi('power.max_limit')):.0f} W"),
                 ('Driver', _smi('driver_version'))]
        m = re.search(r'CUDA Version:\s+([\d.]+)', subprocess.check_output(['nvidia-smi'], text=True))
        rows.append(('CUDA (driver)', m.group(1) if m else 'n/a'))
    except Exception as e:
        rows.append(('GPU', f'not found ({e})'))
    try:
        import pycuda
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(0)
        cc = dev.compute_capability()
        rows += [('Compute capability', f'{cc[0]}.{cc[1]}'),
                 ('Multiprocessors', dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)),
                 ('PyCUDA', pycuda.VERSION_TEXT)]
    except Exception as e:
        rows.append(('PyCUDA', f'not available ({e})'))
    rows.append(('CPU', _proc('/proc/cpuinfo', r'model name\s*:\s*(.+)')))
    kb = _proc('/proc/meminfo', r'MemTotal:\s+(\d+)')
    rows.append(('RAM', f'{int(kb) / 1024 ** 2:.1f} GB' if kb.isdigit() else 'n/a'))
    rows.append(('Python', platform.python_version()))
    for name in ['numpy', 'scipy', 'numba', 'pandas', 'sklearn', 'xgboost', 'cupy', 'opfunu']:
        try:
            rows.append((name, __import__(name).__version__))
        except Exception:
            rows.append((name, 'not installed'))
    for label, value in rows:
        print(f'{label:<20} {value}')


if __name__ == '__main__':
    main()
