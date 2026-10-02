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

from ..basis.evaluate import chi_grad_matrix, chi_hess_matrix, chi_matrix
from ..grid.angular import spherical_product
from ..grid.becke import becke_weights, own_cell_weight_gradient
from ..grid.radial import treutler_ahlrichs_m4


def pair_grid(pos_a, pos_b, radius_a: float, radius_b: float, n_r: int, n_a: int,
              raw: bool = False):
    """Two-centre Becke-partitioned grid; returns ``(points, weights)``.

    With ``raw=True`` returns ``(points, raw_weights, owner, centres, radii)``
    instead, the quadrature weights before the Becke cell factor.
    """
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
    if raw:
        return points, weights, owner, centres, radii
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

    # component-major layout so that each Cartesian slice is contiguous for BLAS
    n = len(points)
    grad_a = np.empty((3, n, len(local_a)))
    grad_b = np.empty((3, n, len(local_b)))
    chi_grad_matrix(local_a, positions, points, out=grad_a.transpose(1, 2, 0))
    chi_grad_matrix(local_b, positions, points, out=grad_b.transpose(1, 2, 0))
    T_block = 0.5 * sum((grad_a[x] * weights[:, None]).T @ grad_b[x] for x in range(3))
    return S_block, T_block


def overlap_kinetic_2c_gradient(basis_a, basis_b, pos_a, pos_b, radius_a, radius_b,
                                Pb, Wb, n_r: int = 40, n_a: int = 14):
    """``d/dR_B [sum(Pb * T_block) - sum(Wb * S_block)]`` for one atom pair.

    The exact derivative of the quadrature :func:`overlap_kinetic_2c` performs,
    with A fixed: B's sub-grid moves with B, A's stays put, and the Becke cell
    factors change with both. Per point ``F = Pb : (1/2 grad a . grad b) -
    Wb : (a b)``; on A's points only the B functions move (``d/dR_B b =
    -grad b``), on B's points only the A functions do (``+grad a``). The
    weight term uses :func:`own_cell_weight_gradient` (points fixed) for A's
    points and translational invariance for B's. ``d/dR_A`` is the negative.
    Returns a 3-vector, or None when the compiled Hessian is unavailable.
    """
    points, raw, owner, centres, radii = pair_grid(
        pos_a, pos_b, radius_a, radius_b, n_r, n_a, raw=True)
    from ..kernels import pair_gradient_terms_kernel

    kernel = pair_gradient_terms_kernel()
    if kernel is None or any(f.kind != "sto" for f in [*basis_a, *basis_b]):
        return None
    local_a = [_rebind(f, 0) for f in basis_a]
    local_b = [_rebind(f, 1) for f in basis_b]
    cell = becke_weights(points, centres, radii)[np.arange(len(points)), owner]
    w = raw * cell
    F = np.empty(len(points))
    dF = np.empty((len(points), 3))
    # A's points: the B functions move (d/dR_B b = -grad b, d grad b = -H_b);
    # B's points move with B, so the A functions do (+grad a, +H_a)
    for sel, fixed, moving, C, D, sign in (
        (owner == 0, local_a, local_b, Pb, Wb, -1.0),
        (owner == 1, local_b, local_a, Pb.T, Wb.T, 1.0),
    ):
        pts = points[sel]
        out_F = np.empty(len(pts))
        out_dF = np.empty((len(pts), 3))
        kernel(
            chi_matrix(fixed, centres, pts),
            chi_grad_matrix(fixed, centres, pts),
            chi_matrix(moving, centres, pts),
            chi_grad_matrix(moving, centres, pts),
            chi_hess_matrix(moving, centres, pts),
            np.ascontiguousarray(C), np.ascontiguousarray(D), sign, out_F, out_dF,
        )
        F[sel] = out_F
        dF[sel] = out_dF
    grad = w @ dF

    M = own_cell_weight_gradient(points, owner, raw * F, centres, radii)
    return grad + M[0, 1] - M[1, 0]


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
