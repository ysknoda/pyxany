"""Dispatch between the numba and numpy implementations of the hot loops.

``HAVE_NUMBA`` in :mod:`pyxany.parallel` decides which backend is live; setting
``PYXANY_DISABLE_NUMBA=1`` forces the numpy path, which is how the two are
compared in tests.
"""

from __future__ import annotations

from functools import lru_cache

from ..parallel import HAVE_NUMBA
from . import numpy_impl


@lru_cache(maxsize=1)
def _numba():
    if not HAVE_NUMBA:
        return None
    try:  # pragma: no cover - exercised only where numba is installed
        from . import numba_impl

        return numba_impl
    except Exception:  # pragma: no cover - a broken numba should not be fatal
        return None


def backend_name() -> str:
    return "numba" if _numba() is not None else "numpy"


def chi_block_kernel():
    """Fast STO block evaluator, or None to use the generic numpy path."""
    mod = _numba()
    return mod.chi_block if mod is not None else None


def poisson_radial(r, w, rho):
    mod = _numba()
    if mod is not None:
        return mod.poisson_radial(r, w, rho)
    return numpy_impl.poisson_radial(r, w, rho)


def density_from_chi(chi, P, out=None):
    import numpy as np

    mod = _numba()
    if mod is not None:
        if out is None:
            out = np.empty(chi.shape[0], dtype=float)
        mod.density_from_chi(chi, P, out)
        return out
    rho = np.einsum("km,mn,kn->k", chi, P, chi, optimize=True)
    if out is not None:
        out[:] = rho
        return out
    return rho


def numerov_outward(g, h, l, m, out):
    mod = _numba()
    if mod is not None:
        return mod.numerov_outward(g, h, l, m, out)
    return numpy_impl.numerov_outward(g, h, l, m, out)


def numerov_inward(g, h, r, kappa, m, out):
    mod = _numba()
    if mod is not None:
        return mod.numerov_inward(g, h, r, kappa, m, out)
    return numpy_impl.numerov_inward(g, h, r, kappa, m, out)


__all__ = ["backend_name", "chi_block_kernel", "poisson_radial", "density_from_chi",
           "numerov_outward", "numerov_inward"]
