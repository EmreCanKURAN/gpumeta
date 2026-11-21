"""GPU energy measurement through NVML."""
import threading
import time

import numpy as np

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


class GPUEnergyMonitor:
    """Samples the GPU power draw through NVML in a background thread and
    integrates it (trapezoidal rule) to energy in joules."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self.ok = NVML_OK
        if self.ok:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.ok = False
        self._thread = None
        self._stop = threading.Event()
        self.samples = []

    def _loop(self):
        while not self._stop.is_set():
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append((time.time(), p_mw / 1000.0))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if not self.ok:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.ok or self._thread is None:
            return {'energy_j': float('nan'), 'avg_w': float('nan'), 'peak_w': float('nan')}
        self._stop.set()
        self._thread.join(timeout=2.0)
        if len(self.samples) < 2:
            return {'energy_j': float('nan'), 'avg_w': float('nan'), 'peak_w': float('nan')}
        t = np.array([s[0] for s in self.samples])
        w = np.array([s[1] for s in self.samples])
        energy = float(_trapezoid(w, t))
        return {'energy_j': energy, 'avg_w': float(np.mean(w)), 'peak_w': float(np.max(w))}


_ENERGY_MON = GPUEnergyMonitor()


def run_with_energy(fn, *args, **kwargs):
    """Run fn and add energy (J), average and peak power (W) and the
    energy-delay product to its result dictionary."""
    _ENERGY_MON.start()
    res = fn(*args, **kwargs)
    e = _ENERGY_MON.stop()
    res.update(e)
    t_s = res.get('time_ms', 0) / 1000.0
    res['edp'] = res['energy_j'] * t_s if np.isfinite(res.get('energy_j', float('nan'))) else float('nan')
    return res
