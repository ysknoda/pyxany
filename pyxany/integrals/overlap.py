"""Overlap (S) and kinetic-energy (T) matrices.

Same-atom blocks come from the exact one-centre routines; cross-atom blocks use
a two-centre Becke-partitioned grid.  Atom pairs whose functions cannot overlap
above ``screening_tol`` are skipped entirely, which is what makes a few-hundred
atom cluster tractable: the number of surviving pairs grows linearly with the
number of atoms rather than quadratically.

Under MPI the surviving cross-atom pairs are distributed round-robin and the
partial matrices are summed with a single Allreduce.  Same-atom blocks are
cheap and analytic, so they are computed redundantly on every rank and written
after the reduction rather than being double counted.
"""

from __future__ import annotations

import numpy as np

from ..parallel import allreduce_array, round_robin
from ..timing import timer
from .onecentre import RADIAL_POINTS, kinetic_1c, overlap_1c
from .twocentre import overlap_kinetic_2c, overlap_kinetic_2c_gradient, pair_is_negligible

DEFAULT_SCREENING_TOL = 1e-10


def surviving_pairs(molecule, basis, screening_tol: float = DEFAULT_SCREENING_TOL):
    """Cross-atom pairs whose overlap can exceed ``screening_tol``."""
    groups = _groups(molecule, basis)
    D = molecule.distance_matrix()
    pairs = []
    for ai in range(len(molecule)):
        for aj in range(ai + 1, len(molecule)):
            fa = [basis[mu] for mu in groups[ai]]
            fb = [basis[mu] for mu in groups[aj]]
            if not pair_is_negligible(fa, fb, float(D[ai, aj]), screening_tol):
                pairs.append((ai, aj))
    return pairs


def _groups(molecule, basis) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(len(molecule))]
    for mu, f in enumerate(basis):
        groups[f.atom_idx].append(mu)
    return groups


def build_overlap_kinetic(molecule, basis, n_r: int = 40, n_a: int = 14,
                          screening_tol: float = DEFAULT_SCREENING_TOL,
                          n_r_1c: int = RADIAL_POINTS):
    """Return ``(S, T)`` for the whole molecule.

    ``n_r`` / ``n_a`` size the two-centre grids (``grid.pair``); ``n_r_1c`` is
    the radial quadrature of one-centre integrals of non-Slater functions
    (``grid.one_centre``; Slater pairs use the closed form).
    """
    with timer("integrals.S_T"):
        n = len(basis)
        S = np.zeros((n, n))
        T = np.zeros((n, n))
        groups = _groups(molecule, basis)

        # -- cross-atom blocks (distributed, then reduced) -----------------
        pairs = surviving_pairs(molecule, basis, screening_tol)
        radii = [a.grid_radius() for a in molecule]
        for ai, aj in round_robin(pairs):
            mu_idx, nu_idx = groups[ai], groups[aj]
            if not mu_idx or not nu_idx:
                continue
            S_blk, T_blk = overlap_kinetic_2c(
                [basis[m] for m in mu_idx], [basis[n_] for n_ in nu_idx],
                molecule[ai].pos, molecule[aj].pos, radii[ai], radii[aj],
                n_r=n_r, n_a=n_a,
            )
            ix = np.ix_(mu_idx, nu_idx)
            S[ix] = S_blk
            T[ix] = T_blk
        allreduce_array(S)
        allreduce_array(T)
        S += S.T
        T += T.T

        # -- same-atom blocks (analytic, written after the reduction) ------
        for idx in groups:
            for a, mu in enumerate(idx):
                for nu in idx[a:]:
                    s = overlap_1c(basis[mu], basis[nu], n_r_1c)
                    t = kinetic_1c(basis[mu], basis[nu], n_r_1c)
                    S[mu, nu] = S[nu, mu] = s
                    T[mu, nu] = T[nu, mu] = t
        # T from the gradient form is symmetric by construction; symmetrise the
        # one-centre closed form too so that S and T are exactly symmetric.
        T = 0.5 * (T + T.T)
        return S, T


def condition_number(S: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(S)
    return float(ev.max() / max(ev.min(), 1e-30))


def overlap_kinetic_gradient(molecule, basis, P: np.ndarray, W: np.ndarray,
                             n_r: int = 40, n_a: int = 14,
                             screening_tol: float = DEFAULT_SCREENING_TOL,
                             step: float = 1e-4) -> np.ndarray:
    """``d/dR [tr(P T) - tr(W S)]`` for the S and T that the SCF actually used.

    The Pulay terms involving S and T must differentiate *these* matrices --
    built on two-centre grids and one-centre closed forms -- not a DV-grid
    quadrature of the same integrals, or the two quadrature errors do not
    cancel (measured on CO/DZ at n_r = 40: 1.3e-3 and 1.7e-3 Ha/Bohr).

    One-centre blocks do not depend on the geometry. A two-centre block
    depends only on ``R_B - R_A``, so ``d/dR_A = -d/dR_B``. With the compiled
    kernels the derivative is exact (:func:`overlap_kinetic_2c_gradient`,
    one pass with basis Hessians and Becke-weight derivatives); otherwise one
    central difference per Cartesian direction of ``R_B`` (6 block
    evaluations per pair). The two agree to ~1e-9. Returns ``(n_atoms, 3)``;
    rank-local pairs are summed with one Allreduce.
    """
    with timer("forces.S_T_gradient"):
        groups = _groups(molecule, basis)
        radii = [a.grid_radius() for a in molecule]
        grad = np.zeros((len(molecule), 3))
        for ai, aj in round_robin(surviving_pairs(molecule, basis, screening_tol)):
            mu_idx, nu_idx = groups[ai], groups[aj]
            if not mu_idx or not nu_idx:
                continue
            fa = [basis[m] for m in mu_idx]
            fb = [basis[n] for n in nu_idx]
            ix = np.ix_(mu_idx, nu_idx)
            # factor 2: the (B, A) block is the transpose and contributes equally
            Pb, Wb = 2.0 * P[ix], 2.0 * W[ix]
            exact = overlap_kinetic_2c_gradient(fa, fb, molecule[ai].pos, molecule[aj].pos,
                                                radii[ai], radii[aj], Pb, Wb, n_r=n_r, n_a=n_a)
            if exact is not None:
                grad[aj] += exact
                grad[ai] -= exact
                continue
            for x in range(3):
                e = np.zeros(3)
                e[x] = step
                Sp, Tp = overlap_kinetic_2c(fa, fb, molecule[ai].pos, molecule[aj].pos + e,
                                            radii[ai], radii[aj], n_r=n_r, n_a=n_a)
                Sm, Tm = overlap_kinetic_2c(fa, fb, molecule[ai].pos, molecule[aj].pos - e,
                                            radii[ai], radii[aj], n_r=n_r, n_a=n_a)
                d = (np.sum(Pb * (Tp - Tm)) - np.sum(Wb * (Sp - Sm))) / (2.0 * step)
                grad[aj, x] += d
                grad[ai, x] -= d
        allreduce_array(grad)
        return grad
