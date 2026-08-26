"""Two-centre overlap and kinetic-energy integrals on a Becke-partitioned grid.

For a pair of atoms (A, B) the integrand has a cusp at *both* centres, so a
single grid centred on A under-resolves the region around B.  This module
therefore builds one atom-centred grid per centre and combines them with a
two-atom Becke partition, exactly as the molecular grid does.

The kinetic energy uses the gradient form

    T_{mu nu} = 1/2 int grad chi_mu . grad chi_nu dV

which is equivalent to ``-1/2 int chi_mu lap chi_nu dV`` for square-integrable
functions (the surface term vanishes) but is far better behaved on a numerical
grid: it is manifestly symmetric and its integrand has no ``1/r`` spike at the
nuclei.
"""

from __future__ import annotations

import math

import numpy as np

from ..basis.evaluate import chi_grad_matrix, chi_matrix
from ..grid.angular import spherical_product
from ..grid.becke import becke_weights
from ..grid.radial import treutler_ahlrichs_m4


def pair_grid(pos_a, pos_b, radius_a: float, radius_b: float, n_r: int, n_a: int):
    """Two-centre Becke-partitioned grid; returns ``(points, weights)``."""
    dirs, w_ang = spherical_product(n_a)
    centres = np.vstack([np.asarray(pos_a, float), np.asarray(pos_b, float)])
    radii = np.array([radius_a, radius_b])

    pts_all, wts_all, owner_all = [], [], []
    for i, (centre, radius) in enumerate(zip(centres, radii)):
        r, w_r = treutler_ahlrichs_m4(n_r, radius)
        pts = (r[:, None, None] * dirs[None, :, :]).reshape(-1, 3) + centre
        wts = (w_r[:, None] * r[:, None] ** 2 * w_ang[None, :]).reshape(-1)
        pts_all.append(pts)
        wts_all.append(wts)
        owner_all.append(np.full(len(pts), i, dtype=np.int32))

    points = np.vstack(pts_all)
    weights = np.concatenate(wts_all)
    owner = np.concatenate(owner_all)
    cell = becke_weights(points, centres, radii)
    return points, weights * cell[np.arange(len(points)), owner]


def overlap_kinetic_2c(basis_a, basis_b, pos_a, pos_b, radius_a, radius_b,
                       n_r: int = 40, n_a: int = 14):
    """Cross-atom (n_a x n_b) blocks of S and T for one atom pair.

    ``basis_a`` / ``basis_b`` are the basis functions centred on A and B.
    Returns ``(S_block, T_block)``.
    """
    points, weights = pair_grid(pos_a, pos_b, radius_a, radius_b, n_r, n_a)
    positions = np.vstack([np.asarray(pos_a, float), np.asarray(pos_b, float)])

    # Re-index the two function lists onto a local 2-atom molecule
    local_a = [_rebind(f, 0) for f in basis_a]
    local_b = [_rebind(f, 1) for f in basis_b]

    chi_a = chi_matrix(local_a, positions, points)
    chi_b = chi_matrix(local_b, positions, points)
    S_block = (chi_a * weights[:, None]).T @ chi_b

    grad_a = chi_grad_matrix(local_a, positions, points)
    grad_b = chi_grad_matrix(local_b, positions, points)
    T_block = 0.5 * np.einsum("kmx,k,knx->mn", grad_a, weights, grad_b, optimize=True)
    return S_block, T_block


def _rebind(bf, atom_idx: int):
    """Shallow copy of a basis function bound to a different atom index."""
    import copy

    g = copy.copy(bf)
    g.atom_idx = atom_idx
    return g


def screening_exponent(bf) -> float:
    """A decay constant used to bound how far a function reaches."""
    return bf.exponent_scale()


def pair_is_negligible(basis_a, basis_b, distance: float, tol: float) -> bool:
    """Cheap upper bound on the largest |S| between two atoms' functions.

    Slater products ``e^(-z_i r_A) e^(-z_j r_B)`` peak at ``e^(-min(z_i, z_j) R)``
    along the internuclear axis; Gaussian products peak at
    ``e^(-alpha beta R^2 / (alpha + beta))``.  Using the *smallest* exponent on
    each atom gives the most generous (safest) bound.
    """
    if not basis_a or not basis_b:
        return True
    za = min(screening_exponent(f) for f in basis_a)
    zb = min(screening_exponent(f) for f in basis_b)
    kinds = {f.kind for f in basis_a} | {f.kind for f in basis_b}
    if kinds == {"gto"}:
        a2, b2 = za * za, zb * zb
        bound = math.exp(-a2 * b2 / (a2 + b2) * distance * distance)
    else:
        bound = math.exp(-min(za, zb) * distance)
    return bound < tol
