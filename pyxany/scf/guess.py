"""Initial density matrices from a superposition of atomic occupations."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..data import elements as elem
from ..data.elements import _subshell_l


def atomic_superposition(molecule, basis) -> np.ndarray:
    """Diagonal P whose atomic blocks carry the free-atom electron counts.

    Occupations of a given angular momentum on an atom are shared equally among
    all basis functions of that (atom, l), so a double-zeta pair each get half.
    A pseudo-hydrogen carries its fractional charge in its single 1s function.
    """
    n = len(basis)
    P = np.zeros((n, n))
    share = Counter((f.atom_idx, f.l) for f in basis)

    for mu, f in enumerate(basis):
        atom = molecule[f.atom_idx]
        if atom.is_pseudo:
            P[mu, mu] = atom.Z / share[(f.atom_idx, f.l)]
            continue
        occ = elem.occupations(atom.element)
        total = sum(v for k, v in occ.items() if _subshell_l(k) == f.l)
        count = share[(f.atom_idx, f.l)]
        P[mu, mu] = total / count if count else 0.0
    return P


def rescale_to_electron_count(P: np.ndarray, S: np.ndarray, n_electrons: float) -> np.ndarray:
    """Scale P so that ``Tr(P S)`` is exactly the requested electron count."""
    trace = float(np.einsum("mn,nm->", P, S))
    if trace <= 1e-12:
        return P
    return P * (n_electrons / trace)


def spin_split(P: np.ndarray, molecule, basis, n_alpha: float, n_beta: float):
    """Split a closed-shell guess into (P_alpha, P_beta).

    The excess alpha electrons are placed on the valence functions of the
    heaviest atom, which is enough to break the spin symmetry and let the SCF
    find the open-shell solution.
    """
    total = n_alpha + n_beta
    P_a = P * (n_alpha / total)
    P_b = P * (n_beta / total)
    extra = n_alpha - n_beta
    if extra <= 0:
        return P_a, P_b

    heavy = max(range(len(molecule)), key=lambda i: molecule[i].Z)
    valence = [mu for mu, f in enumerate(basis)
               if f.atom_idx == heavy and getattr(f, "n", 1) >= 2]
    if not valence:
        valence = [mu for mu, f in enumerate(basis) if f.atom_idx == heavy]
    if valence:
        per = extra / len(valence)
        for mu in valence:
            P_a[mu, mu] += per
    return P_a, P_b
