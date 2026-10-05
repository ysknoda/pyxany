"""Becke fuzzy-cell partitioning of a molecular integration grid.

Reference
---------
A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988).

A molecular grid built by stacking atom-centerd spherical grids integrates each
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
    centers: np.ndarray,
    radii: np.ndarray,
    stiffness: int = DEFAULT_STIFFNESS,
) -> np.ndarray:
    """Normalized cell weights ``w[k, A]`` for every point and atom.

    Parameters
    ----------
    points   : (n_points, 3) Cartesian coordinates, Bohr
    centers  : (n_atoms, 3) atomic positions, Bohr
    radii    : (n_atoms,) Bragg-Slater radii, Bohr
    stiffness: number of iterations of Becke's smoothing polynomial

    Returns
    -------
    (n_points, n_atoms) array whose rows sum to 1.
    """
    pts = np.asarray(points, dtype=float)
    C = np.asarray(centers, dtype=float)
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
    centers: np.ndarray,
    radii: np.ndarray,
    stiffness: int = DEFAULT_STIFFNESS,
) -> np.ndarray:
    """Only the weight of each point's own parent atom -- what a molecular grid needs.

    ``owner[k]`` is the index of the atom whose sub-grid produced ``points[k]``.
    Equivalent to ``becke_weights(...)[k, owner[k]]`` but avoids materializing
    the full (n_points, n_atoms) array.
    """
    w = becke_weights(points, centers, radii, stiffness)
    return w[np.arange(len(points)), np.asarray(owner)]


def _smooth_step_and_derivative(nu: np.ndarray, stiffness: int):
    """``s(nu)`` and ``ds/dnu`` (chain rule through the iterated polynomial)."""
    f = np.clip(nu, -1.0, 1.0)
    df = np.ones_like(f)
    for _ in range(stiffness):
        df = df * 1.5 * (1.0 - f * f)
        f = 1.5 * f - 0.5 * f * f * f
    return 0.5 * (1.0 - f), -0.5 * df


def own_cell_weight_gradient(
    points: np.ndarray,
    owner: np.ndarray,
    coeff: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    stiffness: int = DEFAULT_STIFFNESS,
) -> np.ndarray:
    """``M[A, B] = sum_{k owned by A} coeff_k * d w_A(r_k) / d R_B`` with the points held fixed.

    ``w_A = P_A / Z`` is the own-cell Becke weight of each point, ``Z = sum_C P_C``.
    With ``P_C = prod_{D != C} s(nu_CD)``, a nuclear displacement enters only
    through the pair factors that contain it, so

        d w_A / d R_B = sum_C (delta_CA - w_A) / Z * P_C * sum_{D != C} dln s_CD / d R_B

    and ``nu_CD`` depends on ``R_C`` and ``R_D`` alone. Returns an
    ``(n_atoms, n_atoms, 3)`` array; memory stays ``O(n_points * n_atoms)``.

    The derivative for ``B = A`` with the points moving *with* their atom (the
    case on an atom-centerd grid) follows from translational invariance,
    ``-sum_{B != A} M[A, B]``; that bookkeeping is left to the caller.
    """
    pts = np.asarray(points, dtype=float)
    C = np.asarray(centers, dtype=float)
    owner = np.asarray(owner)
    n_atoms = len(C)
    out = np.zeros((n_atoms, n_atoms, 3))
    if n_atoms == 1:
        return out
    from ..kernels import own_cell_weight_gradient as compiled

    fast = compiled(pts, owner, coeff, C, size_adjustment(radii), stiffness)
    if fast is not None:
        return fast

    a = size_adjustment(radii)
    diff = pts[:, None, :] - C[None, :, :]                       # r - R_C
    dist = np.linalg.norm(diff, axis=2)
    unit = diff / np.maximum(dist, 1e-300)[:, :, None]

    # P_C at every point, and the per-pair data needed for the derivative
    pair = {}
    P = np.ones((len(pts), n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            Rij_vec = C[i] - C[j]
            Rij = np.linalg.norm(Rij_vec)
            mu = (dist[:, i] - dist[:, j]) / Rij
            nu = mu + a[i, j] * (1.0 - mu * mu)
            s, ds = _smooth_step_and_derivative(nu, stiffness)
            P[:, i] *= s
            # d ln s / d nu, with s -> 0 only where ds -> 0 faster (f'(+-1) = 0)
            dln = np.where(s > 1e-300, ds / np.maximum(s, 1e-300), 0.0) * (1.0 - 2.0 * a[i, j] * mu)
            # d mu / d R_i and d mu / d R_j  (points fixed)
            dmu_di = -unit[:, i, :] / Rij - mu[:, None] * Rij_vec / Rij**2
            dmu_dj = unit[:, j, :] / Rij + mu[:, None] * Rij_vec / Rij**2
            pair[(i, j)] = (dln, dmu_di, dmu_dj)

    Z = P.sum(axis=1)
    Z = np.where(Z > 1e-300, Z, 1.0)
    w_own = P[np.arange(len(pts)), owner] / Z
    for (i, j), (dln, dmu_di, dmu_dj) in pair.items():
        # coefficient of d P_i / d R_B for each point: coeff (delta_{i,owner} - w_own) / Z
        alpha = coeff * ((owner == i) - w_own) / Z * P[:, i] * dln
        for A in range(n_atoms):
            sel = owner == A
            if not sel.any():
                continue
            out[A, i] += alpha[sel] @ dmu_di[sel]
            out[A, j] += alpha[sel] @ dmu_dj[sel]
    return out
