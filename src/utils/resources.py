"""
Resource monitoring without extra dependencies (psutil is not in requirements.txt).
  process_ram_mib()  current / peak resident memory of this process (Windows: working set via
                     GetProcessMemoryInfo; Linux/Colab: /proc/self/status and getrusage)
  gpu_peak_mib()     peak memory allocated by PyTorch on the current CUDA device
"""
from __future__ import annotations

import sys


def process_ram_mib() -> dict:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return {"current": round(c.WorkingSetSize / 2**20, 1), "peak": round(c.PeakWorkingSetSize / 2**20, 1)}
    import resource
    cur = None
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                cur = round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return {"current": cur, "peak": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}


def gpu_peak_mib() -> float | None:
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None
