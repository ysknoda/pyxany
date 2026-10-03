"""Hartree potential by Becke multi-centre expansion, and nuclear attraction.

Method
------
The classical Coulomb potential of the molecular density is obtained without any
pair loop or softening parameter:

1. Partition the density with the same Becke cell functions the grid uses::

       rho(r) = sum_A rho_A(r),   rho_A(r) = w_A(r) rho(r)

   Each ``rho_A`` is localised on its own atom and decays in every direction.

2. Project each ``rho_A`` onto real spherical harmonics on the *angular* part of
   atom A's own sub-grid (which is an exact product grid, so this is one
   contraction per radial shell)::

       rho_A^{lm}(r) = int rho_A(r, Omega) Y_lm(Omega) dOmega

3. Solve the radial Poisson equation analytically for each (A, l, m)::

       V^{lm}(r) = 4 pi / (2l+1) [ r^-(l+1) int_0^r s^(l+2) rho^{lm} ds
                                 + r^l      int_r^inf s^(1-l) rho^{lm} ds ]

4. Sum the atomic potentials back onto *every* grid point.

Cost is ``O(N_atoms * N_points * n_lm)`` instead of the
``O(N_atoms^2 * N_points)`` pair loop of a direct sum, and the result is free of
the near-singularity that forces direct-sum implementations to introduce an
ad-hoc softening length.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import sph_harm_y

from ..basis.solid_harmonics import real_spherical_harmonics, real_spherical_harmonics_T
from ..parallel import allreduce_array, round_robin
from ..timing import timer

DEFAULT_L_MAX = 6


def real_sph_harmonics(l_max: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """(n_points, n_lm) real orthonormal spherical harmonics, from angles.

    Reference implementation kept for tests; the hot path uses the Cartesian
    recursion in :mod:`pyxany.basis.solid_harmonics`, which is an order of
    magnitude faster and agrees with this to ~1e-14.
    """
    cols = []
    for l, m in lm_index_pairs(l_max):
        if m == 0:
            cols.append(sph_harm_y(l, 0, theta, phi).real)
        else:
            Y = sph_harm_y(l, abs(m), theta, phi)
            s = np.sqrt(2.0) * (-1.0) ** m
            cols.append(s * (Y.real if m > 0 else Y.imag))
    return np.column_stack(cols)


def lm_index_pairs(l_max: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]


def _angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(vectors, axis=1)
    safe = np.maximum(r, 1e-30)
    theta = np.arccos(np.clip(vectors[:, 2] / safe, -1.0, 1.0))
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    return theta, phi


# Gauss-Legendre order of each radial segment integral in the Poisson solve.
_SEGMENT_ORDER = 8


def _radial_poisson_multipoles(r: np.ndarray, rho_lm: np.ndarray,
                               l_of: np.ndarray) -> np.ndarray:
    """Radial multipole potentials for every ``(l, m)`` column at once.

        V_l(r) = 4pi/(2l+1) [ int_0^r (s/r)^(l+1) s rho_l(s) ds
                            + int_r^inf (r/s)^l   s rho_l(s) ds ]

    This is the Green's-function form of the radial Poisson equation, written so
    that every power that appears is a ratio <= 1. The familiar form
    ``r^-(l+1) int_0^r s^(l+2) rho ds`` is the same quantity but divides one
    tiny number by another near the nucleus; for l >= 8 on a heavy-atom grid
    that amplified spline and projection noise without bound (measured on the
    GaAs4 cluster with off-centre Gaussian test charges: E_H wrong by 1.6e3 Ha at
    l = 8), and the previous remedy -- overwriting rho_l and V_l with r^l power
    laws inside an l-dependent radius that reached 0.2-0.3 Bohr at l = 10 --
    discarded real core structure (the GaAs4 SCF moved by 0.15 Ha between
    l = 8 and l = 10).

    Here both integrals are accumulated node by node,

        I_k = I_(k-1) (r_(k-1)/r_k)^(l+1) + int_(r_(k-1))^(r_k) (s/r_k)^(l+1) s rho ds
        O_k = O_(k+1) (r_k/r_(k+1))^l     + int_(r_k)^(r_(k+1)) (r_k/s)^l   s rho ds

    so noise at small s is *damped* by the ratio rather than amplified, and no
    near-nucleus override is needed. Each segment integral applies an 8-point
    Gauss-Legendre rule to the cubic spline of r^2 rho_l (not partial sums of the
    global radial rule, which are not cumulative integrals at all). Below the
    first node rho_l is continued as ``rho_l(r_0) (s/r_0)^l``, its exact
    small-r form for a regular density.

    ``r`` must be sorted ascending; ``rho_lm`` is (n_r, n_lm); ``l_of`` gives the
    angular momentum of each column.
    """
    r = np.asarray(r, dtype=float)
    lf = np.asarray(l_of, dtype=float)[None, None, :]          # (1, 1, n_lm)
    # Spline r^2 rho_l, not rho_l: it vanishes at the origin and spans far fewer
    # decades through the core, so a cubic follows it between the sparse outer
    # nodes (measured: E_H error 0.10 Ha -> 4e-3 Ha on the GaAs4 test charges).
    spline = CubicSpline(r, r[:, None] ** 2 * rho_lm, axis=0)
    t, wt = np.polynomial.legendre.leggauss(_SEGMENT_ORDER)

    # segments [lo_k, hi_k], k = 0..n_r-1, with lo_0 = 0
    lo = np.concatenate([[0.0], r[:-1]])
    hi = r
    half = 0.5 * (hi - lo)
    s = lo[:, None] + half[:, None] * (t[None, :] + 1.0)        # (n_r, q)
    ws = half[:, None] * wt[None, :]
    rho_s = spline(s) / (s * s)[:, :, None]                     # (n_r, q, n_lm)
    first = s[0][:, None] / r[0]
    rho_s[0] = rho_lm[0][None, :] * np.where(lf[0] > 0, first ** lf[0], 1.0)

    # inner: segment k measured against its upper end r_k
    ratio_in = (s[:, :, None] / hi[:, None, None]) ** (lf + 1.0)
    seg_in = np.einsum("kq,kqj->kj", ws * s, rho_s * ratio_in)
    # outer: segment k+1 measured against its lower end r_k  (lo_(k+1) = r_k)
    seg_out = np.einsum("kq,kqj->kj", ws * s,
                        rho_s * (lo[:, None, None] / np.maximum(s[:, :, None], 1e-300)) ** lf)

    n_r = len(r)
    inner = np.empty_like(seg_in)
    outer = np.empty_like(seg_in)
    inner[0] = seg_in[0]
    for k in range(1, n_r):
        inner[k] = inner[k - 1] * (r[k - 1] / r[k]) ** (lf[0, 0] + 1.0) + seg_in[k]
    outer[-1] = 0.0          # beyond the last node (tens of Bohr) rho is negligible
    for k in range(n_r - 2, -1, -1):
        outer[k] = outer[k + 1] * (r[k] / r[k + 1]) ** lf[0, 0] + seg_out[k + 1]
    return (4.0 * np.pi / (2.0 * lf[0, 0] + 1.0)) * (inner + outer)


def hartree_potential(grid, rho: np.ndarray, positions: np.ndarray,
                      l_max: int = DEFAULT_L_MAX,
                      eval_points: np.ndarray | None = None,
                      gradient: bool = False) -> np.ndarray | None:
    """V_H from a density sampled on ``grid``, evaluated at ``eval_points``.

    Parameters
    ----------
    grid        : :class:`pyxany.grid.Grid` built with Becke partitioning
    rho         : (n_points,) total electron density on the **full** grid --
                  the multipole projection needs complete angular shells, so
                  this cannot be a rank-local slice
    positions   : (n_atoms, 3) nuclear positions, Bohr
    l_max       : multipole truncation of the atomic expansions
    eval_points : where to evaluate; defaults to the full grid.  Passing
                  ``grid.local_points`` keeps the reconstruction step
                  distributed under MPI.
    gradient    : return the exact (n_points, 3) gradient of the potential
                  instead; needs the compiled kernel, None without it

    Under MPI the projection loop is split over atoms and the (small) radial
    multipole arrays are summed, so only the reconstruction cost scales with the
    number of evaluation points held by each rank.
    """
    with timer("coulomb.hartree"):
        n_atoms = grid.n_atoms
        lm = lm_index_pairs(l_max)
        l_of = np.array([l for l, _ in lm])

        # Projection kernels w_ang * Y_lm, one per angular grid in use. A pruned
        # shell (n_a' < n_a) resolves only l <= n_a' - 1; its higher multipoles
        # are left at zero (ADR 0009) -- the inner shells it applies to see an
        # almost spherical density.
        kernels: dict[int, np.ndarray] = {}

        def kernel(na: int) -> np.ndarray:
            if na not in kernels:
                from ..grid.angular import angular_grid

                d0, w0 = angular_grid(grid.angular, na)
                n_lm = (min(l_max, na - 1) + 1) ** 2
                Y = real_spherical_harmonics(min(l_max, na - 1), d0[:, 0], d0[:, 1],
                                             d0[:, 2], np.ones(len(d0)))
                kernels[na] = w0[:, None] * Y[:, :n_lm]
            return kernels[na]

        # -- steps 1-3: one set of radial multipole potentials per atom -----
        v_radial = [np.zeros((int(grid.n_r_atom[ai]), len(lm))) for ai in range(n_atoms)]
        rho_cell = grid.cell_weights * rho
        for ai in round_robin(range(n_atoms)):
            rho_lm = np.zeros((int(grid.n_r_atom[ai]), len(lm)))          # (n_r, n_lm)
            for lo, hi, na, rows in grid.shell_blocks(ai):
                wY = kernel(na)
                rho_lm[lo:hi, :wY.shape[1]] = rho_cell[rows].reshape(hi - lo, -1) @ wY
            r = grid.radii[ai]
            v_radial[ai] = _radial_poisson_multipoles(r, rho_lm, l_of)
        # atoms may carry different radial sizes: reduce one flat buffer
        flat = allreduce_array(np.concatenate([v.ravel() for v in v_radial]))
        sizes = np.cumsum([0] + [v.size for v in v_radial])
        v_radial = [flat[sizes[i]:sizes[i + 1]].reshape(v.shape)
                    for i, v in enumerate(v_radial)]

        # -- step 4: sum every atom's potential onto the evaluation points ---
        pts = grid.points if eval_points is None else np.asarray(eval_points, float)
        from ..kernels import hartree_evaluate

        if gradient:
            return hartree_evaluate(pts, positions, grid.radii, v_radial, l_max, gradient=True)
        V = hartree_evaluate(pts, positions, grid.radii, v_radial, l_max)
        if V is None:
            V = _sum_multipoles(pts, positions, grid.radii, v_radial, l_of, l_max)
        return V


def _sum_multipoles(pts, positions, radii, v_radial, l_of, l_max) -> np.ndarray:
    """numpy reference for the multipole summation (kernels.hartree_evaluate)."""
    V = np.zeros(len(pts))
    for ai in range(len(radii)):
        d = pts - positions[ai]
        rr = np.sqrt(np.einsum("ij,ij->i", d, d))
        Yt = real_spherical_harmonics_T(l_max, d[:, 0], d[:, 1], d[:, 2], rr)
        v_at_r = _evaluate_multipoles(radii[ai], v_radial[ai], l_of, rr)
        V += np.einsum("kj,jk->k", v_at_r, Yt, optimize=True)
    return V


def _evaluate_multipoles(r_nodes: np.ndarray, v_nodes: np.ndarray,
                         l_of: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Interpolate every radial multipole potential at once, with analytic tails.

    Outside the charge distribution the exact behaviour is ``V ~ C / r^(l+1)``;
    near the origin it is ``V ~ r^l``.  Extrapolating with those forms rather
    than clamping is what keeps the potential correct beyond the outermost
    radial node -- a plain ``right=0`` there throws away the entire monopole
    tail, which is by far the largest term.

    ``v_nodes`` is (n_r, n_lm) and one CubicSpline handles all columns.
    """
    out = np.empty((len(r), v_nodes.shape[1]))
    inner = r < r_nodes[0]
    outer = r > r_nodes[-1]
    mid = ~(inner | outer)
    if mid.any():
        out[mid] = CubicSpline(r_nodes, v_nodes, axis=0, extrapolate=False)(r[mid])
    if inner.any():
        ratio = (r[inner] / r_nodes[0])[:, None] ** l_of[None, :]
        out[inner] = v_nodes[0][None, :] * ratio
    if outer.any():
        moment = v_nodes[-1] * r_nodes[-1] ** (l_of + 1.0)
        out[outer] = moment[None, :] / r[outer][:, None] ** (l_of + 1.0)[None, :]
    return out


def nuclear_potential(points: np.ndarray, positions: np.ndarray,
                      charges: np.ndarray) -> np.ndarray:
    """``sum_A -Z_A / |r - R_A|`` at each point (Hartree)."""
    v = np.zeros(len(points))
    for pos, Z in zip(positions, charges):
        r = np.linalg.norm(points - pos, axis=1)
        v -= Z / np.maximum(r, 1e-12)
    return v


def build_potential_matrix(chi: np.ndarray, weights: np.ndarray, v: np.ndarray,
                           reduce: bool = True) -> np.ndarray:
    """``<mu| v |nu>`` from a chi block, its weights and a potential sample."""
    M = (chi * (weights * v)[:, None]).T @ chi
    return allreduce_array(M) if reduce else M
