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


def chi_hess_block_kernel():
    """Fast STO Hessian evaluator, or None (no numpy counterpart; callers fall back)."""
    mod = _numba()
    return mod.chi_hess_block if mod is not None else None


def pair_gradient_terms_kernel():
    """Compiled per-point S/T gradient contraction, or None."""
    mod = _numba()
    return mod.pair_gradient_terms if mod is not None else None


def point_charge_field(points, centers, charges, sigma, gradient=False):
    """Compiled Gaussian point-charge potential (and gradient), or None."""
    import numpy as np

    mod = _numba()
    if mod is None:
        return None
    pts = np.ascontiguousarray(points, dtype=float)
    v = np.empty(len(pts))
    g = np.empty((len(pts), 3)) if gradient else np.empty((0, 3))
    mod.point_charge_field(pts, np.ascontiguousarray(centers, dtype=float),
                           np.ascontiguousarray(charges, dtype=float), float(sigma),
                           bool(gradient), v, g)
    return (v, g) if gradient else v


def chi_all_kernel():
    """Single-call STO evaluator for functions on many atoms, or None."""
    mod = _numba()
    return mod.chi_all if mod is not None else None


def chi_grad_block_kernel():
    """Fast STO gradient evaluator, or None to use the generic numpy path."""
    mod = _numba()
    return mod.chi_grad_block if mod is not None else None


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


__all__ = ["backend_name", "chi_block_kernel", "chi_grad_block_kernel", "chi_hess_block_kernel",
           "poisson_radial", "density_from_chi",
           "numerov_outward", "numerov_inward"]


def becke_own_cell(points, owner, centers, a, stiffness):
    """Own-cell Becke weights; compiled when numba is available."""
    import numpy as np

    mod = _numba()
    if mod is None:
        return None
    R = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(R, 1.0)
    out = np.empty(len(points))
    mod.becke_own_cell(np.ascontiguousarray(points, dtype=float), np.asarray(owner, np.int64),
                       np.ascontiguousarray(centers, dtype=float), np.ascontiguousarray(a),
                       1.0 / R, int(stiffness), out)
    return out


def hartree_evaluate(points, centers, radii, v_radial, l_max, gradient: bool = False):
    """Compiled multipole summation, or None to use the numpy path.

    With ``gradient=True`` returns the exact (n_points, 3) gradient of the
    same piecewise potential instead of its values.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    from ..basis.solid_harmonics import normalization_factors

    mod = _numba()
    if mod is None:
        return None
    n_at = len(radii)
    n_lm = (l_max + 1) ** 2
    nmax = max(len(r) for r in radii)
    knots = np.zeros((n_at, nmax))
    n_knots = np.zeros(n_at, dtype=np.int64)
    coefs = np.zeros((n_at, 4, nmax - 1, n_lm))
    v_first = np.zeros((n_at, n_lm))
    v_last = np.zeros((n_at, n_lm))
    for A, (r, v) in enumerate(zip(radii, v_radial, strict=True)):
        cs = CubicSpline(r, v, axis=0, extrapolate=False)
        knots[A, :len(r)] = r
        n_knots[A] = len(r)
        coefs[A, :, :len(r) - 1] = cs.c
        v_first[A] = v[0]
        v_last[A] = v[-1]
    kernel = mod.hartree_evaluate_grad if gradient else mod.hartree_evaluate
    out = np.empty((len(points), 3)) if gradient else np.empty(len(points))
    kernel(np.ascontiguousarray(points, dtype=float),
           np.ascontiguousarray(centers, dtype=float), knots, n_knots, coefs,
           v_first, v_last, int(l_max), normalization_factors(l_max), out)
    return out


def own_cell_weight_gradient(points, owner, coeff, centers, a, stiffness):
    """Compiled Becke-weight gradient contraction, or None for the numpy path."""
    import numpy as np

    mod = _numba()
    if mod is None:
        return None
    owner = np.asarray(owner, np.int64)
    n_at = len(centers)
    order = np.argsort(owner, kind="stable").astype(np.int64)
    starts = np.searchsorted(owner[order], np.arange(n_at + 1)).astype(np.int64)
    out = np.zeros((n_at, n_at, 3))
    mod.own_cell_weight_gradient(np.ascontiguousarray(points, dtype=float), order, starts,
                                 np.ascontiguousarray(coeff, dtype=float),
                                 np.ascontiguousarray(centers, dtype=float),
                                 np.ascontiguousarray(a), int(stiffness), out)
    return out
