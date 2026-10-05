"""One-center overlap and kinetic-energy integrals.

Because every basis type exposes an exact analytic radial function and an exact
analytic radial Laplacian (see :mod:`pyxany.basis.types`), all one-center
integrals reduce to one-dimensional radial quadrature and need no finite
differences.  Slater-Slater pairs additionally have a closed form, which is used
by default and cross-checked against the quadrature in the tests.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..grid.radial import log_squared_map

#: Radial points for the 1-D quadrature fallback.  Cheap: this is a 1-D integral.
RADIAL_POINTS = 400


@lru_cache(maxsize=8)
def _radial_quadrature(n: int = RADIAL_POINTS):
    return log_squared_map(n, 1.0)


def _same_angular(a, b) -> bool:
    return a.l == b.l and a.m == b.m


def _I(k: int, s: float) -> float:
    """int_0^inf r^k exp(-s r) dr = k! / s^(k+1)."""
    return math.factorial(k) / s ** (k + 1) if k >= 0 else 0.0


def overlap_1c(bf_i, bf_j, n_r: int = RADIAL_POINTS) -> float:
    """<chi_i | chi_j> for two functions on the same center."""
    if not _same_angular(bf_i, bf_j):
        return 0.0
    if bf_i.kind == "sto" and bf_j.kind == "sto":
        return bf_i.norm * bf_j.norm * _I(bf_i.n + bf_j.n, bf_i.zeta + bf_j.zeta)
    r, w = _radial_quadrature(n_r)
    return float(np.dot(w, bf_i.radial(r) * bf_j.radial(r) * r * r))


def kinetic_1c(bf_i, bf_j, n_r: int = RADIAL_POINTS) -> float:
    """<chi_i | -1/2 laplacian | chi_j> for two functions on the same center.

    Closed form for Slater pairs, using the exact Laplacian of
    ``r^(n-1) exp(-zeta r) Y_lm``::

        lap = [(n-1)n - l(l+1)] r^(n-3) - 2 n zeta r^(n-2) + zeta^2 r^(n-1)

    all multiplied by ``exp(-zeta r) Y_lm``.
    """
    if not _same_angular(bf_i, bf_j):
        return 0.0
    if bf_i.kind == "sto" and bf_j.kind == "sto":
        ni, zi = bf_i.n, bf_i.zeta
        nj, zj = bf_j.n, bf_j.zeta
        l = bf_i.l
        s = zi + zj
        return -0.5 * bf_i.norm * bf_j.norm * (
            ((nj - 1) * nj - l * (l + 1)) * _I(ni + nj - 2, s)
            - 2.0 * nj * zj * _I(ni + nj - 1, s)
            + zj * zj * _I(ni + nj, s)
        )
    r, w = _radial_quadrature(n_r)
    integrand = bf_i.radial(r) * bf_j.laplacian_radial(r) * r ** bf_j.l * r * r
    return -0.5 * float(np.dot(w, integrand))


def nuclear_attraction_1c(bf_i, bf_j, Z: float) -> float:
    """<chi_i | -Z/r | chi_j> on the same center (used for atomic references)."""
    if not _same_angular(bf_i, bf_j):
        return 0.0
    if bf_i.kind == "sto" and bf_j.kind == "sto":
        return -Z * bf_i.norm * bf_j.norm * _I(bf_i.n + bf_j.n - 1, bf_i.zeta + bf_j.zeta)
    r, w = _radial_quadrature()
    return -Z * float(np.dot(w, bf_i.radial(r) * bf_j.radial(r) * r))
