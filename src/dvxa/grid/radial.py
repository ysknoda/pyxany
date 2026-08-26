"""Radial quadrature for atom-centred grids."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _leggauss(n: int) -> tuple[tuple, tuple]:
    x, w = np.polynomial.legendre.leggauss(n)
    return tuple(x), tuple(w)


def treutler_ahlrichs_m4(n: int, radius: float = 1.0, alpha: float = 0.6):
    """Treutler-Ahlrichs M4 radial mapping, Treutler & Ahlrichs (1995) JCP 102, 346.

    Maps Gauss-Legendre nodes x in [-1, 1] onto r in [0, inf)::

        r(x) = R / ln2 * (1 + x)^alpha * ln(2 / (1 - x))

    The element-specific scale ``R`` (a Bragg-Slater radius) concentrates nodes
    near the nucleus, which is what makes tight core Slater functions integrable
    with a modest number of points.

    Returns
    -------
    (r, w) : arrays such that ``sum(w * f(r))`` approximates ``int_0^inf f(r) dr``.
    Note there is no ``r^2`` factor here -- the volume element is applied by the
    caller that builds the 3-D grid.
    """
    x, w = (np.asarray(a) for a in _leggauss(n))
    xc = np.clip(x, -1.0 + 1e-14, 1.0 - 1e-14)
    ln2 = np.log(2.0)
    r = radius / ln2 * (1.0 + xc) ** alpha * np.log(2.0 / (1.0 - xc))
    drdx = radius / ln2 * (
        alpha * (1.0 + xc) ** (alpha - 1.0) * np.log(2.0 / (1.0 - xc))
        + (1.0 + xc) ** alpha / (1.0 - xc)
    )
    return r, w * drdx


def log_squared_map(n: int, radius: float = 1.0):
    """Rational Gauss-Legendre map r = R (1+x)/(1-x), used for two-centre grids.

    Returns ``(r, w)`` for ``int_0^inf f(r) dr``.
    """
    x, w = (np.asarray(a) for a in _leggauss(n))
    xc = np.clip(x, -1.0 + 1e-14, 1.0 - 1e-14)
    r = radius * (1.0 + xc) / (1.0 - xc)
    drdx = 2.0 * radius / (1.0 - xc) ** 2
    return r, w * drdx
