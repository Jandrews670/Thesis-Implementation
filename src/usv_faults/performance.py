from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from typing import Dict, List, Optional


class PerformanceSampler:
    """Measure process CPU time and sampled resident memory for a code block."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._start_memory = process_memory_mb()
        self._peak_sampled_rss_mb = self._start_memory.get("rss_mb")

    def start(self) -> "PerformanceSampler":
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_memory = process_memory_mb()
        self._peak_sampled_rss_mb = self._start_memory.get("rss_mb")
        self._thread = threading.Thread(target=self._sample_memory, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, object]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        end_memory = process_memory_mb()
        wall_s = max(0.0, end_wall - self._start_wall)
        cpu_s = max(0.0, end_cpu - self._start_cpu)
        logical_cores = int(os.cpu_count() or 1)
        cpu_percent_all_cores = (cpu_s / wall_s / logical_cores * 100.0) if wall_s > 0.0 else None
        cpu_percent_one_core = (cpu_s / wall_s * 100.0) if wall_s > 0.0 else None
        peak_candidates = [
            self._peak_sampled_rss_mb,
            self._start_memory.get("peak_rss_mb"),
            end_memory.get("peak_rss_mb"),
            end_memory.get("rss_mb"),
        ]
        peak_rss_mb = max(value for value in peak_candidates if value is not None)
        return {
            "wall_time_s": wall_s,
            "process_cpu_time_s": cpu_s,
            "cpu_usage_percent_all_cores": cpu_percent_all_cores,
            "cpu_usage_percent_one_core": cpu_percent_one_core,
            "logical_cpu_count": logical_cores,
            "rss_start_mb": self._start_memory.get("rss_mb"),
            "rss_end_mb": end_memory.get("rss_mb"),
            "peak_rss_mb": peak_rss_mb,
            "memory_measurement_method": end_memory.get("method", self._start_memory.get("method")),
        }

    def _sample_memory(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            rss_mb = process_memory_mb().get("rss_mb")
            if rss_mb is not None:
                if self._peak_sampled_rss_mb is None:
                    self._peak_sampled_rss_mb = rss_mb
                else:
                    self._peak_sampled_rss_mb = max(self._peak_sampled_rss_mb, rss_mb)


def process_memory_mb() -> Dict[str, object]:
    if sys.platform.startswith("win"):
        return _windows_process_memory_mb()
    linux_memory = _linux_proc_status_memory_mb()
    if linux_memory is not None:
        return linux_memory
    return _resource_memory_mb()


def _linux_proc_status_memory_mb() -> Optional[Dict[str, object]]:
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return None
    values: Dict[str, float] = {}
    with open(status_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmHWM:")):
                parts = line.split()
                if len(parts) >= 2:
                    values[parts[0].rstrip(":")] = float(parts[1]) / 1024.0
    return {
        "rss_mb": values.get("VmRSS"),
        "peak_rss_mb": values.get("VmHWM", values.get("VmRSS")),
        "method": "/proc/self/status",
    }


def _resource_memory_mb() -> Dict[str, object]:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
        peak_mb = float(usage.ru_maxrss) / divisor
        return {"rss_mb": None, "peak_rss_mb": peak_mb, "method": "resource.getrusage"}
    except Exception:
        return {"rss_mb": None, "peak_rss_mb": None, "method": "not_available"}


def _windows_process_memory_mb() -> Dict[str, object]:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return {"rss_mb": None, "peak_rss_mb": None, "method": "windows_psapi_failed"}
    divisor = 1024.0 * 1024.0
    return {
        "rss_mb": float(counters.WorkingSetSize) / divisor,
        "peak_rss_mb": float(counters.PeakWorkingSetSize) / divisor,
        "method": "windows_psapi.GetProcessMemoryInfo",
    }


def sdae_compute_estimates(
    model_config: Dict[str, object],
    train_windows: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict[str, object]:
    dimensions = sdae_linear_dimensions(model_config)
    linear_macs = int(sum(in_dim * out_dim for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])))
    bias_adds = int(sum(dimensions[1:]))
    forward_linear_flops = int(2 * linear_macs + bias_adds)
    parameter_count = int(sum((in_dim * out_dim) + out_dim for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])))
    training_flops_per_window = int(3 * forward_linear_flops)
    train_window_epochs = None
    training_flops_total = None
    if train_windows is not None and epochs is not None:
        train_window_epochs = int(train_windows) * int(epochs)
        training_flops_total = int(training_flops_per_window * train_window_epochs)
    return {
        "linear_layer_dimensions": dimensions,
        "parameter_count": parameter_count,
        "estimated_forward_linear_macs_per_window": linear_macs,
        "estimated_forward_linear_flops_per_window": forward_linear_flops,
        "estimated_training_linear_flops_per_window": training_flops_per_window,
        "train_window_epochs": train_window_epochs,
        "estimated_training_linear_flops_total": training_flops_total,
        "flop_estimate_note": (
            "Linear-layer estimate only. Forward FLOPs count multiply-adds as 2 FLOPs plus bias adds; "
            "training is approximated as 3x forward FLOPs. Activations, optimizer bookkeeping, data loading, "
            "and Python overhead are excluded."
        ),
    }


def sdae_linear_dimensions(model_config: Dict[str, object]) -> List[int]:
    input_dim = int(model_config["input_dim"])
    hidden_dims = [int(value) for value in model_config["hidden_dims"]]
    latent_dim = int(model_config["latent_dim"])
    return [input_dim, *hidden_dims, latent_dim, *reversed(hidden_dims), input_dim]
