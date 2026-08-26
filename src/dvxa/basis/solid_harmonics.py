"""Fast real solid harmonics of arbitrary order, by recursion.

The multipole expansion in the Poisson solver needs real spherical harmonics up
to l ~ 6-10 at every grid point of every atom, which is the single hottest loop
in an SCF iteration.  Evaluating them through a general complex special function
costs microseconds per point; the recursions below are pure polynomial
arithmetic in (x, y, z) with no trigonometry and no branches.

Recursions (Helgaker, Jorgensen & Olsen, *Molecular Electronic-Structure
Theory*, sec. 6.4) for the real solid harmonics ``S_l^m = r^l Y_l^m /
sqrt((2l+1)/4pi)``::

    S_0^0    = 1
    S_l^l    = sqrt((2l-1)/(2l)) [ x S_{l-1}^{l-1} - y S_{l-1}^{-(l-1)} ]
    S_l^{-l} = sqrt((2l-1)/(2l)) [ y S_{l-1}^{l-1} + x S_{l-1}^{-(l-1)} ]
    S_l^m    = [ (2l-1) z S_{l-1}^m - sqrt((l-1)^2 - m^2) r^2 S_{l-2}^m ]
               / sqrt(l^2 - m^2)                                  (|m| < l)

The normalised real spherical harmonic is then

    Y_l^m = sqrt((2l+1)/(4 pi)) * (sqrt(2) if m != 0 else 1) * S_l^m / r^l

(the extra sqrt(2) for m != 0 comes from splitting each complex |m| pair into a
cosine and a sine component).  The resulting columns agree with
``scipy.special.sph_harm_y``'s real combinations sign for sign, which
tests/test_harmonics.py pins.

Column order is ``(l, m)`` with ``m`` running -l..l, matching
:func:`dvxa.integrals.coulomb.lm_index_pairs`.
"""

from __future__ import annotations

import math

import numpy as np


def lm_columns(l_max: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]


def _column(l: int, m: int) -> int:
    return l * l + (m + l)


def real_solid_harmonics(l_max: int, x, y, z) -> np.ndarray:
    """(n_points, (l_max+1)^2) array of ``S_l^m(x, y, z)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = x.shape[0]
    r2 = x * x + y * y + z * z
    S = np.zeros((n, (l_max + 1) ** 2))
    S[:, 0] = 1.0
    for l in range(1, l_max + 1):
        f = math.sqrt((2 * l - 1) / (2 * l))
        prev_c = S[:, _column(l - 1, l - 1)]
        prev_s = S[:, _column(l - 1, -(l - 1))] if l > 1 else np.zeros(n)
        S[:, _column(l, l)] = f * (x * prev_c - y * prev_s)
        S[:, _column(l, -l)] = f * (y * prev_c + x * prev_s)
        for m in range(-(l - 1), l):
            num = (2 * l - 1) * z * S[:, _column(l - 1, m)]
            if l >= 2 and abs(m) <= l - 2:
                num = num - math.sqrt((l - 1) ** 2 - m * m) * r2 * S[:, _column(l - 2, m)]
            S[:, _column(l, m)] = num / math.sqrt(l * l - m * m)
    return S


def real_spherical_harmonics(l_max: int, x, y, z, r=None) -> np.ndarray:
    """(n_points, (l_max+1)^2) orthonormal real spherical harmonics ``Y_l^m``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)
    S = real_solid_harmonics(l_max, x, y, z)
    safe = np.maximum(r, 1e-300)
    out = np.empty_like(S)
    for l in range(l_max + 1):
        base = math.sqrt((2 * l + 1) / (4.0 * math.pi))
        lo, hi = l * l, (l + 1) ** 2
        block = S[:, lo:hi] / (safe**l)[:, None] if l else S[:, lo:hi]
        scales = np.array([base * (math.sqrt(2.0) if m else 1.0)
                           for m in range(-l, l + 1)])
        out[:, lo:hi] = block * scales
    return out


def normalisation_factors(l_max: int) -> np.ndarray:
    """Per-column ``sqrt((2l+1)/4pi) * sqrt(2)^[m != 0]`` factors."""
    return np.array([
        math.sqrt((2 * l + 1) / (4.0 * math.pi)) * (math.sqrt(2.0) if m else 1.0)
        for l, m in lm_columns(l_max)
    ])
