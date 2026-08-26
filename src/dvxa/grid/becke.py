"""Becke fuzzy-cell partitioning of a molecular integration grid.

Reference
---------
A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988).

A molecular grid built by stacking atom-centred spherical grids integrates each
atomic sub-grid over *all* space, so a naive superposition counts every region
once per atom -- an ``N_atoms``-fold overcounting.  Becke's construction assigns
each point a set of weights ``w_A(r)`` with ``sum_A w_A(r) = 1`` everywhere, so
that the molecular integral

    int f dV  =  sum_A  sum_k  w_A(r_k^A) * w_k^A * f(r_k^A)

is correct regardless of how many atoms overlap in a region.

The cell function is

    mu_AB  = (r_A - r_B) / R_AB
    nu_AB  = mu_AB + a_AB (1 - mu_AB^2)          atomic-size adjustment
    s(nu)  = 1/2 (1 - f_k(nu)),  f(x) = (3x - x^3)/2 iterated k times
    P_A    = prod_{B != A} s(nu_AB)
    w_A    = P_A / sum_C P_C

with ``a_AB`` derived from the ratio of Bragg-Slater radii and clipped to
|a| <= 1/2 as Becke prescribes.
"""

from __future__ import annotations

import numpy as np

DEFAULT_STIFFNESS = 3


def size_adjustment(radii: np.ndarray) -> np.ndarray:
    """(n_atoms, n_atoms) matrix of Becke's ``a_AB`` atomic-size parameters.

    ``chi = R_A / R_B``, ``u = (chi - 1)/(chi + 1)``, ``a = u / (u^2 - 1)``,
    clipped to [-1/2, 1/2] (Becke 1988, eqs. A2-A6).
    """
    R = np.asarray(radii, dtype=float)
    chi = R[:, None] / R[None, :]
    u = (chi - 1.0) / (chi + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = u / (u * u - 1.0)
    a = np.nan_to_num(a, nan=0.0, posinf=0.5, neginf=-0.5)
    return np.clip(a, -0.5, 0.5)


def _smooth_step(nu: np.ndarray, stiffness: int) -> np.ndarray:
    """s(nu) = 1/2 (1 - f_k(nu)) with f(x) = (3x - x^3)/2 applied k times."""
    f = np.clip(nu, -1.0, 1.0)
    for _ in range(stiffness):
        f = 1.5 * f - 0.5 * f * f * f
    return 0.5 * (1.0 - f)


def becke_weights(
    points: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
    stiffness: int = DEFAULT_STIFFNESS,
) -> np.ndarray:
    """Normalised cell weights ``w[k, A]`` for every point and atom.

    Parameters
    ----------
    points   : (n_points, 3) Cartesian coordinates, Bohr
    centres  : (n_atoms, 3) atomic positions, Bohr
    radii    : (n_atoms,) Bragg-Slater radii, Bohr
    stiffness: number of iterations of Becke's smoothing polynomial

    Returns
    -------
    (n_points, n_atoms) array whose rows sum to 1.
    """
    pts = np.asarray(points, dtype=float)
    C = np.asarray(centres, dtype=float)
    n_atoms = len(C)
    if n_atoms == 1:
        return np.ones((len(pts), 1))

    a = size_adjustment(radii)
    dist = np.linalg.norm(pts[:, None, :] - C[None, :, :], axis=2)     # (npts, natoms)
    R = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)          # (natoms, natoms)
    np.fill_diagonal(R, 1.0)                                            # avoid 0/0; masked below

    P = np.ones((len(pts), n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            mu = (dist[:, i] - dist[:, j]) / R[i, j]
            nu = mu + a[i, j] * (1.0 - mu * mu)
            P[:, i] *= _smooth_step(nu, stiffness)

    total = P.sum(axis=1, keepdims=True)
    # A point so far from every atom that all P underflow: fall back to the
    # nearest atom rather than producing NaN.
    degenerate = (total[:, 0] <= 1e-300)
    if degenerate.any():
        P[degenerate] = 0.0
        P[degenerate, np.argmin(dist[degenerate], axis=1)] = 1.0
        total = P.sum(axis=1, keepdims=True)
    return P / total


def becke_weights_own_cell(
    points: np.ndarray,
    owner: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
    stiffness: int = DEFAULT_STIFFNESS,
) -> np.ndarray:
    """Only the weight of each point's own parent atom -- what a molecular grid needs.

    ``owner[k]`` is the index of the atom whose sub-grid produced ``points[k]``.
    Equivalent to ``becke_weights(...)[k, owner[k]]`` but avoids materialising
    the full (n_points, n_atoms) array.
    """
    w = becke_weights(points, centres, radii, stiffness)
    return w[np.arange(len(points)), np.asarray(owner)]
