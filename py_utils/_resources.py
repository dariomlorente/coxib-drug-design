from __future__ import annotations

# RESOURCES
# Shared utility functions for hardware-aware resource management

import multiprocessing as mp
import os


def _get_ram_budget_gb(reserve_gb: float = 4.0) -> float:
    """Return usable RAM in GB (total physical RAM minus reserve).

    Detects total physical memory and subtracts a safety reserve for the
    OS and background applications.  Analogous to how ``_get_n_workers``
    detects CPU cores and leaves one free.

    Parameters:
        reserve_gb: GB to keep free for OS + apps (default: 4.0).

    Returns:
        Available GB for computation (minimum 1.0).
        MBP M3 16 GB -> 12.0, iMac M4 24 GB -> 20.0.
    """
    total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    total_gb = total_bytes / (1024 ** 3)
    return max(total_gb - reserve_gb, 1.0)


def _get_n_workers(n_workers: int | None = None) -> int:
    """Return number of parallel workers (default: cpu_count - 1).

    Leaves one core free for the OS, just like ``_get_ram_budget_gb``
    leaves RAM free.

    Parameters:
        n_workers: Explicit override.  If *None*, uses ``cpu_count - 1``.

    Returns:
        Number of workers (minimum 1).
        MBP M3 -> 7, iMac M4 -> 9.
    """
    if n_workers is not None:
        return max(1, n_workers)
    return max(1, mp.cpu_count() - 1)
