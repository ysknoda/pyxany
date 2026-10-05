"""MPI and Numba availability, kept behind a single small facade.

The whole package runs unchanged with or without mpi4py and numba installed.
Import this module rather than importing mpi4py/numba directly, so that the
"is it available?" logic lives in exactly one place.

Launch examples::

    pyxany run molecule.yaml                  # serial
    mpirun -n 8 pyxany run molecule.yaml      # 8 ranks
"""

from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------
# MPI
# --------------------------------------------------------------------------
try:  # pragma: no cover - depends on the installed environment
    from mpi4py import MPI as _MPI

    COMM = _MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    _HAVE_MPI4PY = True
except ImportError:  # pragma: no cover
    _MPI = None
    COMM = None
    RANK = 0
    SIZE = 1
    _HAVE_MPI4PY = False

ROOT = 0
#: True only when actually running with more than one rank.
PARALLEL = SIZE > 1


def is_root() -> bool:
    """True on the rank that owns console output and file writes."""
    return RANK == ROOT


def mpi_print(*args, **kwargs) -> None:
    """print() that only fires on rank 0 (a no-op elsewhere)."""
    if RANK == ROOT:
        print(*args, **kwargs)


def allreduce_array(arr: np.ndarray) -> np.ndarray:
    """Sum ``arr`` in place across all ranks.  No-op when serial."""
    if PARALLEL:
        COMM.Allreduce(_MPI.IN_PLACE, arr, op=_MPI.SUM)
    return arr


def allreduce_scalar(value: float) -> float:
    """Sum a Python scalar across all ranks.  No-op when serial."""
    if PARALLEL:
        return float(COMM.allreduce(value, op=_MPI.SUM))
    return float(value)


def bcast_array(arr: np.ndarray, root: int = ROOT) -> np.ndarray:
    """Broadcast ``arr`` in place from ``root``.  No-op when serial."""
    if PARALLEL:
        COMM.Bcast(arr, root=root)
    return arr


def bcast_object(obj, root: int = ROOT):
    """Broadcast an arbitrary picklable object.  No-op when serial."""
    if PARALLEL:
        return COMM.bcast(obj, root=root)
    return obj


def barrier() -> None:
    if PARALLEL:
        COMM.Barrier()


def local_slice(n_total: int, rank: int | None = None, size: int | None = None) -> slice:
    """Contiguous block decomposition of ``n_total`` items over the ranks.

    Returns ``slice(0, n_total)`` when running serially, so callers can always
    use the same code path.
    """
    if not PARALLEL:
        return slice(0, n_total)
    r = RANK if rank is None else rank
    s = SIZE if size is None else size
    lo = (r * n_total) // s
    hi = ((r + 1) * n_total) // s
    return slice(lo, hi)


def round_robin(items, rank: int | None = None, size: int | None = None) -> list:
    """This rank's share of ``items`` (round-robin).  All items when serial."""
    if not PARALLEL:
        return list(items)
    r = RANK if rank is None else rank
    s = SIZE if size is None else size
    return list(items)[r::s]


def mpi_version() -> str:
    if not _HAVE_MPI4PY:  # pragma: no cover
        return "not installed"
    major, minor = _MPI.Get_version()
    return f"MPI {major}.{minor} via mpi4py {_MPI.__doc__ and ''}".strip() or f"MPI {major}.{minor}"


# --------------------------------------------------------------------------
# Numba
# --------------------------------------------------------------------------
def _numba_requested() -> bool:
    """Honor PYXANY_DISABLE_NUMBA=1 so tests can compare both code paths."""
    return os.environ.get("PYXANY_DISABLE_NUMBA", "").strip() not in ("1", "true", "TRUE", "yes")


try:  # pragma: no cover - depends on the installed environment
    import numba as _numba  # noqa: F401

    _NUMBA_INSTALLED = True
    NUMBA_VERSION = _numba.__version__
except ImportError:  # pragma: no cover
    _NUMBA_INSTALLED = False
    NUMBA_VERSION = None

#: True when numba is installed *and* not disabled through the environment.
HAVE_NUMBA = _NUMBA_INSTALLED and _numba_requested()


def numba_status() -> str:
    if not _NUMBA_INSTALLED:
        return "not installed"
    if not _numba_requested():
        return f"{NUMBA_VERSION} (disabled via PYXANY_DISABLE_NUMBA)"
    return NUMBA_VERSION
