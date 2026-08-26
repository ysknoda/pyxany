"""Basis functions for pseudo-hydrogen terminators.

A pseudo-hydrogen carries a fractional nuclear charge ``alpha`` and ``alpha``
electrons.  In an all-electron Slater/numerical code that is a *complete*
specification -- there is no pseudopotential to parameterise -- so its basis
function is obtained by simply solving the spherical Kohn-Sham problem for that
charge and fitting a 1s Slater function to the result.

Consistency with the rest of the basis
--------------------------------------
The raw atomic solution is more diffuse than a tabulated molecular basis
function, because X-alpha self-interaction is not cancelled: at ``alpha = 1``
the fit gives ``zeta = 0.882`` while the Clementi-Raimondi hydrogen exponent
used everywhere else in an STO calculation is ``1.24``.  Mixing the two would
make the terminator systematically more diffuse than every other function in the
molecule.

The exponent is therefore rescaled by the single constant that makes ``alpha =
1`` reproduce the basis set's own hydrogen value::

    zeta(alpha) = zeta_atomic(alpha) * zeta_H(basis) / zeta_atomic(1)

That constant is *derived*, not fitted: it is fixed entirely by the requirement
that a pseudo-hydrogen of unit charge be an ordinary hydrogen.  Pass
``calibrate=False`` for the raw atomic exponent.

Two limits make the whole construction checkable, and are asserted in
tests/test_pseudo_h_basis.py:

* with the electron-electron terms switched off the solution is the exact
  hydrogenic 1s and the raw exponent is exactly ``alpha``;
* the calibrated exponent at ``alpha = 1`` equals the tabulated hydrogen value
  by construction.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..constants import DEFAULT_ALPHA
from .atomic_solver import solve_atom


@lru_cache(maxsize=64)
def pseudo_hydrogen_solution(alpha: float, xc_alpha: float = DEFAULT_ALPHA):
    """Self-consistent spherical X-alpha solution for a fractional-charge H."""
    if alpha <= 0:
        raise ValueError(f"pseudo-hydrogen charge must be positive, got {alpha}")
    return solve_atom(Z=float(alpha), occupations={(1, 0): float(alpha)},
                      alpha=xc_alpha, n_points=2000,
                      r_max=max(60.0, 60.0 / max(alpha, 0.2)))


@lru_cache(maxsize=8)
def _hydrogen_reference_zeta() -> float:
    """The basis tables' own hydrogen 1s exponent (Clementi-Raimondi, 1.24)."""
    from ..data.loaders import sto_table

    return float(sto_table("sz")["H"][0][3])


@lru_cache(maxsize=64)
def _calibration(xc_alpha: float) -> float:
    return _hydrogen_reference_zeta() / _raw_zeta(1.0, xc_alpha)


def pseudo_hydrogen_zeta(alpha: float, xc_alpha: float = DEFAULT_ALPHA,
                         calibrate: bool = True) -> float:
    """1s Slater exponent for a pseudo-hydrogen of fractional charge ``alpha``."""
    raw = _raw_zeta(float(alpha), xc_alpha)
    return raw * _calibration(xc_alpha) if calibrate else raw


@lru_cache(maxsize=64)
def _raw_zeta(alpha: float, xc_alpha: float = DEFAULT_ALPHA) -> float:
    """Exponent of the Slater 1s with maximum overlap on the atomic solution.

    For a nodeless function maximising the overlap is equivalent to a
    least-squares fit, and the overlap is unimodal in zeta, so golden-section
    search finds the optimum without derivatives.
    """
    sol = pseudo_hydrogen_solution(alpha, xc_alpha)
    orb = sol.orbitals.get((1, 0))
    if orb is None:  # pragma: no cover - would mean the atomic solve failed
        raise RuntimeError(f"could not solve the pseudo-hydrogen atom for alpha={alpha}")

    r = orb.r
    w = np.log(r[1] / r[0]) * r          # logarithmic-grid weights
    R = orb.radial

    def overlap(zeta: float) -> float:
        slater = 2.0 * zeta**1.5 * np.exp(-zeta * r)
        return float(np.dot(w, R * slater * r * r))

    lo, hi = 0.05, 5.0 * max(alpha, 1.0)
    for _ in range(200):                 # golden-section on a unimodal overlap
        a = lo + 0.381966 * (hi - lo)
        b = hi - 0.381966 * (hi - lo)
        if overlap(a) < overlap(b):
            lo = a
        else:
            hi = b
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def derive_exponents(molecule, xc_alpha: float = DEFAULT_ALPHA) -> dict[str, float]:
    """``{species_label: zeta}`` for every pseudo-hydrogen species in ``molecule``."""
    out: dict[str, float] = {}
    for atom in molecule:
        if atom.is_pseudo and atom.symbol not in out:
            out[atom.symbol] = pseudo_hydrogen_zeta(round(float(atom.Z), 8), xc_alpha)
    return out


def describe(alpha: float, xc_alpha: float = DEFAULT_ALPHA) -> dict:
    """Diagnostics for one pseudo-hydrogen species, used by ``dvxa passivate``."""
    sol = pseudo_hydrogen_solution(alpha, xc_alpha)
    orb = sol.orbitals[(1, 0)]
    r, R = orb.r, orb.radial
    w = np.log(r[1] / r[0]) * r
    return {
        "alpha": float(alpha),
        "zeta": pseudo_hydrogen_zeta(alpha, xc_alpha),
        "zeta_raw": _raw_zeta(float(alpha), xc_alpha),
        "eigenvalue_hartree": float(orb.eigenvalue),
        "mean_radius_bohr": float(np.dot(w, R * R * r**3)),
        "electrons": float(np.dot(w, sol.density * 4.0 * np.pi * r * r)),
        "converged": bool(sol.converged),
    }
