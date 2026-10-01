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
from .onecentre import kinetic_1c, overlap_1c
from .twocentre import overlap_kinetic_2c, pair_is_negligible

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
                          screening_tol: float = DEFAULT_SCREENING_TOL):
    """Return ``(S, T)`` for the whole molecule."""
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
                    s = overlap_1c(basis[mu], basis[nu])
                    t = kinetic_1c(basis[mu], basis[nu])
                    S[mu, nu] = S[nu, mu] = s
                    T[mu, nu] = T[nu, mu] = t
        # T from the gradient form is symmetric by construction; symmetrise the
        # one-centre closed form too so that S and T are exactly symmetric.
        T = 0.5 * (T + T.T)
        return S, T


def condition_number(S: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(S)
    return float(ev.max() / max(ev.min(), 1e-30))
