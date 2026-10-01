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

from ..basis.solid_harmonics import real_spherical_harmonics
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


# Below this radius the only physically meaningful multipole of a regular
# density is the spherical one: rho_lm(r) -> c_lm r^l as r -> 0, so every l >= 1
# component is already vanishing there and what the angular projection actually
# returns is quadrature noise.  The outer integrand carries a factor r^(1-l),
# which turns that noise into a spike -- measured at 1.0e9 for the O atom of
# water at l = 6, against 4.8e1 for the rest of the range.  Imposing the exact
# r^l behaviour below this radius removes the spike; the region it covers
# contributes O(r^2) ~ 1e-4 to the outer integral, far below the quadrature
# error of the rest.
INNER_MULTIPOLE_RADIUS = 1e-2   # Bohr

# The inner multipole integral is divided by r^(l+1).  Below the radius where
# that factor falls to this value, the division amplifies double-precision
# round-off past usefulness and the exact r^l small-radius form is used instead.
_AMPLIFICATION_FLOOR = 1e-6


def _cumulative_integral(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``int_{r[0]}^{r} f ds`` at every node, for every column of ``f``.

    Integrates the cubic spline of the integrand, which is exact for a cubic and
    converges as O(n_r^-4).  One CubicSpline call covers all (l, m) columns.
    """
    anti = CubicSpline(r, f, axis=0).antiderivative()
    return anti(r) - anti(r[0])[None, :]


def _radial_poisson_multipoles(r: np.ndarray, rho_lm: np.ndarray,
                               l_of: np.ndarray) -> np.ndarray:
    """Radial multipole potentials for every ``(l, m)`` column at once.

        V_l(r) = 4pi/(2l+1) [ r^-(l+1) int_0^r s^(l+2) rho_l(s) ds
                            + r^l      int_r^inf s^(1-l) rho_l(s) ds ]

    Both integrals are *cumulative*, which is why they cannot be formed from
    partial sums of the Gauss-Legendre radial rule.  A Gauss rule is exact only
    over its whole interval; its partial sums are not approximations to
    ``int_0^{r_k}`` at all and converge as O(1/n_r).  Measured on a unit
    Gaussian at n_r = 30 the partial sums were 0.107 electrons out of 1.0 wrong
    while the full sum was exact to ten digits -- which is why this went
    unnoticed.  Integrating the spline instead restores O(n_r^-4): the error in
    V against the closed form ``erf(sqrt(a) r)/r`` falls from 5.1e-3 to 1.2e-4
    at n_r = 30, and from 8.4e-5 to 2.4e-8 at n_r = 240.

    Every ``l >= 1`` component also needs protection near the nucleus, where
    both integrands carry a large negative power of r.  Physically
    ``rho_l(r) -> c r^l`` and ``V_l(r) -> c' r^l`` as ``r -> 0`` -- exactly, for
    any regular density -- so inside the radius where those powers stop being
    numerically trustworthy the asymptotic form is imposed instead of the ratio
    being evaluated.  Both radii follow from the same rule: the r at which the
    amplifying power drops to ``_AMPLIFICATION_FLOOR``.

    ``r`` must be sorted ascending.  ``rho_lm`` is (n_r, n_lm) and ``l_of`` is
    the angular momentum of each column.
    """
    safe = np.maximum(r, 1e-14)
    rho = rho_lm.copy()
    l_values = np.unique(l_of.astype(int))

    # -- 1. impose rho_l ~ r^l where r^(1-l) would amplify projection noise ---
    # Measured without this: the l = 6 outer integrand for the O atom of water
    # reached 1.0e9 at the innermost node against 4.8e1 over the rest of the
    # range, and a cubic spline through that spike rings across the whole grid.
    for l in l_values:
        if l < 1:
            continue
        cut = INNER_MULTIPOLE_RADIUS if l == 1 else max(
            INNER_MULTIPOLE_RADIUS, _AMPLIFICATION_FLOOR ** (1.0 / (l - 1)))
        below = safe < cut
        if not below.any() or below.all():
            continue
        anchor = int(np.argmax(~below))
        cols = np.flatnonzero(l_of == l)
        rows = np.flatnonzero(below)
        shape = (safe[below][:, None] / safe[anchor]) ** float(l)
        rho[np.ix_(rows, cols)] = rho[anchor, cols][None, :] * shape

    # -- 2. the two cumulative integrals -------------------------------------
    inner = _cumulative_integral(r, safe[:, None] ** (l_of + 2.0)[None, :] * rho)
    # The spline starts at r[0]; add the analytic [0, r[0]] shell, over which
    # rho_l is constant to O(r[0]).
    inner = inner + (rho[0] * safe[0] ** (l_of + 3.0) / (l_of + 3.0))[None, :]

    outward = _cumulative_integral(r, safe[:, None] ** (1.0 - l_of)[None, :] * rho)
    outer = outward[-1][None, :] - outward

    v = (4.0 * np.pi / (2.0 * l_of + 1.0))[None, :] * (
        inner / safe[:, None] ** (l_of + 1.0)[None, :]
        + safe[:, None] ** l_of[None, :] * outer
    )

    # -- 3. impose V_l ~ r^l where r^-(l+1) would amplify spline round-off ----
    # At r = 1.3e-3 and l = 12 that factor is ~1e38, which produced |V_l| ~ 1e22.
    # l = 0 is exempt and needs no help: its numerator vanishes as r^3, so
    # inner/r stays finite and carries the real nuclear-region potential.
    for l in l_values:
        if l < 1:
            continue
        cut = _AMPLIFICATION_FLOOR ** (1.0 / (l + 1))
        below = safe < cut
        if not below.any() or below.all():
            continue
        anchor = int(np.argmax(~below))
        cols = np.flatnonzero(l_of == l)
        rows = np.flatnonzero(below)
        shape = (safe[below][:, None] / safe[anchor]) ** float(l)
        v[np.ix_(rows, cols)] = v[anchor, cols][None, :] * shape
    return v


def hartree_potential(grid, rho: np.ndarray, positions: np.ndarray,
                      l_max: int = DEFAULT_L_MAX,
                      eval_points: np.ndarray | None = None) -> np.ndarray:
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

    Under MPI the projection loop is split over atoms and the (small) radial
    multipole arrays are summed, so only the reconstruction cost scales with the
    number of evaluation points held by each rank.
    """
    with timer("coulomb.hartree"):
        n_atoms = grid.n_atoms
        n_ang = grid.n_angular
        lm = lm_index_pairs(l_max)

        # Angular harmonics on the (shared) direction set of every sub-grid
        d0 = grid.directions
        Y_ang = real_spherical_harmonics(l_max, d0[:, 0], d0[:, 1], d0[:, 2],
                                         np.ones(len(d0)))     # (n_ang, n_lm)
        wY = grid.angular_weights[:, None] * Y_ang             # projection kernel
        l_of = np.array([l for l, _ in lm])

        # -- steps 1-3: one set of radial multipole potentials per atom -----
        v_radial = np.zeros((n_atoms, grid.n_r, len(lm)))
        rho_cell = grid.cell_weights * rho
        for ai in round_robin(range(n_atoms)):
            rho_a = grid.as_radial_angular(rho_cell, ai)                   # (n_r, n_ang)
            rho_lm = rho_a @ wY                                            # (n_r, n_lm)
            r = grid.radii[ai]
            v_radial[ai] = _radial_poisson_multipoles(r, rho_lm, l_of)
        allreduce_array(v_radial)

        # -- step 4: sum every atom's potential onto the evaluation points ---
        pts = grid.points if eval_points is None else np.asarray(eval_points, float)
        V = np.zeros(len(pts))
        for ai in range(n_atoms):
            d = pts - positions[ai]
            rr = np.sqrt(np.einsum("ij,ij->i", d, d))
            Y = real_spherical_harmonics(l_max, d[:, 0], d[:, 1], d[:, 2], rr)
            v_at_r = _evaluate_multipoles(grid.radii[ai], v_radial[ai], l_of, rr)
            V += np.einsum("kj,kj->k", v_at_r, Y, optimize=True)
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
