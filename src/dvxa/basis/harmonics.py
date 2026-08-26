"""Real spherical and solid harmonics for l = 0..3, and their gradients.

This module is the *single source of truth* for the angular part of every
basis function.  The numba kernels in :mod:`dvxa.kernels` consume the same
coefficient table exported here, so the JIT and pure-numpy code paths cannot
drift apart (they are pinned together by tests/test_harmonics.py).

Convention
----------
Real spherical harmonics ``Y_lm(theta, phi)``, orthonormal on the unit sphere::

    int Y_lm Y_l'm' dOmega = delta_ll' delta_mm'

Each ``Y_lm`` is a homogeneous polynomial of degree ``l`` in (x, y, z) divided
by ``r^l``.  The polynomial itself is the *real solid harmonic*::

    S_lm(x, y, z) = r^l * Y_lm(theta, phi)

which is what basis functions actually multiply, and which has a clean
analytical gradient -- needed for Pulay force terms.

The m-labelling follows the usual convention in which the cosine-type function
carries +|m| and the sine-type function carries -|m|::

    l=2:  m=+2 -> (x^2 - y^2),   m=-2 -> xy
    l=3:  m=+2 -> (x^2 - y^2)z,  m=-2 -> xyz
"""

from __future__ import annotations

import numpy as np

MAX_L = 3

_C0 = 0.28209479177387814   # sqrt(1  / (4  pi))
_C1 = 0.48860251190291992   # sqrt(3  / (4  pi))
_C20 = 0.31539156525252005  # sqrt(5  / (16 pi))
_C21 = 1.09254843059207907  # sqrt(15 / (4  pi))
_C22 = 0.54627421529603953  # sqrt(15 / (16 pi))
_C30 = 0.37317633259011546  # sqrt(7  / (16 pi))
_C31 = 0.45704579946446577  # sqrt(21 / (32 pi))
_C32 = 1.44530572132027735  # sqrt(105/ (16 pi))
_C33 = 0.59004358992664352  # sqrt(35 / (32 pi))
_C3M2 = 2.89061144264055470  # sqrt(105/ (4  pi))

#: (l, m) -> [(coefficient, (px, py, pz)), ...] monomials of S_lm = r^l Y_lm.
SOLID_HARMONIC_TERMS: dict[tuple[int, int], list[tuple[float, tuple[int, int, int]]]] = {
    (0, 0): [(_C0, (0, 0, 0))],

    (1, 0): [(_C1, (0, 0, 1))],                       # z
    (1, 1): [(_C1, (1, 0, 0))],                       # x
    (1, -1): [(_C1, (0, 1, 0))],                      # y

    (2, 0): [(2 * _C20, (0, 0, 2)),                   # 2z^2 - x^2 - y^2
             (-_C20, (2, 0, 0)), (-_C20, (0, 2, 0))],
    (2, 1): [(_C21, (1, 0, 1))],                      # xz
    (2, -1): [(_C21, (0, 1, 1))],                     # yz
    (2, 2): [(_C22, (2, 0, 0)), (-_C22, (0, 2, 0))],  # x^2 - y^2
    (2, -2): [(_C21, (1, 1, 0))],                     # xy

    (3, 0): [(2 * _C30, (0, 0, 3)),                   # z(5z^2 - 3r^2) = 2z^3 - 3zx^2 - 3zy^2
             (-3 * _C30, (2, 0, 1)), (-3 * _C30, (0, 2, 1))],
    (3, 1): [(4 * _C31, (1, 0, 2)),                   # x(5z^2 - r^2) = 4xz^2 - x^3 - xy^2
             (-_C31, (3, 0, 0)), (-_C31, (1, 2, 0))],
    (3, -1): [(4 * _C31, (0, 1, 2)),                  # y(5z^2 - r^2)
              (-_C31, (2, 1, 0)), (-_C31, (0, 3, 0))],
    (3, 2): [(_C32, (2, 0, 1)), (-_C32, (0, 2, 1))],  # (x^2 - y^2) z
    (3, -2): [(_C3M2, (1, 1, 1))],                    # xyz
    (3, 3): [(_C33, (3, 0, 0)), (-3 * _C33, (1, 2, 0))],   # x(x^2 - 3y^2)
    (3, -3): [(3 * _C33, (2, 1, 0)), (-_C33, (0, 3, 0))],  # y(3x^2 - y^2)
}


def lm_pairs(l: int) -> list[int]:
    """m values for a given l, in the canonical storage order 0, +1, -1, +2, ..."""
    order = [0]
    for k in range(1, l + 1):
        order += [k, -k]
    return order


def _check(l: int, m: int) -> None:
    if (l, m) not in SOLID_HARMONIC_TERMS:
        raise ValueError(
            f"real harmonic (l={l}, m={m}) is not implemented "
            f"(supported: l = 0..{MAX_L}). Use a basis without g/h functions."
        )


def solid_harmonic(l: int, m: int, x, y, z):
    """S_lm(x, y, z) = r^l * Y_lm, evaluated on arrays of Cartesian offsets."""
    _check(l, m)
    out = np.zeros(np.shape(x), dtype=float)
    for coeff, (px, py, pz) in SOLID_HARMONIC_TERMS[(l, m)]:
        term = np.full(np.shape(x), coeff, dtype=float)
        if px:
            term = term * x**px
        if py:
            term = term * y**py
        if pz:
            term = term * z**pz
        out += term
    return out


def solid_harmonic_grad(l: int, m: int, x, y, z):
    """grad S_lm, returned as ``(dS/dx, dS/dy, dS/dz)`` arrays."""
    _check(l, m)
    shape = np.shape(x)
    gx = np.zeros(shape); gy = np.zeros(shape); gz = np.zeros(shape)
    for coeff, (px, py, pz) in SOLID_HARMONIC_TERMS[(l, m)]:
        if px:
            t = np.full(shape, coeff * px, dtype=float)
            if px > 1:
                t = t * x ** (px - 1)
            if py:
                t = t * y**py
            if pz:
                t = t * z**pz
            gx += t
        if py:
            t = np.full(shape, coeff * py, dtype=float)
            if px:
                t = t * x**px
            if py > 1:
                t = t * y ** (py - 1)
            if pz:
                t = t * z**pz
            gy += t
        if pz:
            t = np.full(shape, coeff * pz, dtype=float)
            if px:
                t = t * x**px
            if py:
                t = t * y**py
            if pz > 1:
                t = t * z ** (pz - 1)
            gz += t
    return gx, gy, gz


def real_sph_harm(l: int, m: int, x, y, z, r=None):
    """Y_lm(theta, phi) = S_lm / r^l, safe at the origin."""
    _check(l, m)
    if r is None:
        r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2)
    if l == 0:
        return np.full(np.shape(x), _C0, dtype=float)
    safe = np.maximum(r, 1e-14)
    return solid_harmonic(l, m, x, y, z) / safe**l


# --------------------------------------------------------------------------
# Flat, numba-friendly export of the same coefficient table
# --------------------------------------------------------------------------
def flat_coefficient_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack SOLID_HARMONIC_TERMS into flat arrays for the JIT kernels.

    Returns ``(offsets, counts, coeffs, powers)`` where a harmonic identified by
    ``index_of(l, m)`` owns ``coeffs[offsets[i]:offsets[i]+counts[i]]`` and the
    matching rows of ``powers`` (shape (n_terms, 3)).
    """
    keys = [(l, m) for l in range(MAX_L + 1) for m in lm_pairs(l)]
    offsets = np.zeros(len(keys), dtype=np.int64)
    counts = np.zeros(len(keys), dtype=np.int64)
    coeffs: list[float] = []
    powers: list[tuple[int, int, int]] = []
    for i, key in enumerate(keys):
        terms = SOLID_HARMONIC_TERMS[key]
        offsets[i] = len(coeffs)
        counts[i] = len(terms)
        for c, p in terms:
            coeffs.append(c)
            powers.append(p)
    return offsets, counts, np.asarray(coeffs, float), np.asarray(powers, np.int64)


def index_of(l: int, m: int) -> int:
    """Position of (l, m) in the flat coefficient table."""
    _check(l, m)
    idx = 0
    for ll in range(MAX_L + 1):
        for mm in lm_pairs(ll):
            if (ll, mm) == (l, m):
                return idx
            idx += 1
    raise AssertionError  # pragma: no cover
