"""Gaussian-broadened density of states."""

from __future__ import annotations

import numpy as np


def density_of_states(eps: np.ndarray, sigma: float = 0.3, n_points: int = 600,
                      e_min: float | None = None, e_max: float | None = None,
                      margin: float = 3.0):
    """Return ``(energies, dos)`` with the input eigenvalues in eV.

    ``sigma`` is the Gaussian broadening in the same units as ``eps``.

    Each eigenvalue contributes unit weight, so the curve integrates to the
    number of **orbitals**, not the number of electronic states: for a
    restricted calculation each spatial orbital holds two electrons, so
    multiply by 2 if you want states.  The axis is labeled "states/eV" by
    convention; this is the convention it follows.
    """
    finite = eps[np.isfinite(eps)]
    if len(finite) == 0:
        return np.zeros(0), np.zeros(0)
    lo = finite.min() - margin if e_min is None else e_min
    hi = finite.max() + margin if e_max is None else e_max
    grid = np.linspace(lo, hi, n_points)
    z = (grid[:, None] - finite[None, :]) / sigma
    dos = np.exp(-0.5 * z**2).sum(axis=1) / (sigma * np.sqrt(2.0 * np.pi))
    return grid, dos


def projected_dos(eps: np.ndarray, weights: np.ndarray, sigma: float = 0.3,
                  n_points: int = 600, **kwargs):
    """DOS weighted by a per-orbital contribution (for a species projection)."""
    finite = np.isfinite(eps)
    e = eps[finite]
    w = np.asarray(weights)[finite]
    if len(e) == 0:
        return np.zeros(0), np.zeros(0)
    # an explicit None means "from the levels", like an omitted key (it raised)
    margin = kwargs.get("margin", 3.0)
    lo = kwargs.get("e_min")
    hi = kwargs.get("e_max")
    lo = e.min() - margin if lo is None else lo
    hi = e.max() + margin if hi is None else hi
    grid = np.linspace(lo, hi, n_points)
    z = (grid[:, None] - e[None, :]) / sigma
    dos = (np.exp(-0.5 * z**2) * w[None, :]).sum(axis=1) / (sigma * np.sqrt(2.0 * np.pi))
    return grid, dos
