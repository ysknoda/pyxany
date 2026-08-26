"""Mulliken populations and Wiberg bond orders (vectorised)."""

from __future__ import annotations

import numpy as np


def mulliken(P: np.ndarray, S: np.ndarray, basis, molecule) -> dict:
    """Mulliken gross populations and net atomic charges.

    ``gross_A = sum_{mu in A} (P S)_{mu mu}``; the net charge is the nuclear
    charge minus that, which stays correct for a fractional-charge
    pseudo-hydrogen because ``Molecule.charges`` is float.
    """
    PS = P @ S
    atom_of = np.array([f.atom_idx for f in basis])
    gross = np.zeros(len(molecule))
    np.add.at(gross, atom_of, np.diag(PS))
    return {
        "gross": gross,
        "charges": molecule.charges - gross,
        "atom_of_basis": atom_of,
    }


def mulliken_orbital_populations(C: np.ndarray, S: np.ndarray, occ: np.ndarray,
                                 basis, molecule) -> np.ndarray:
    """(n_occupied, n_atoms) per-MO atomic populations.

    ``pop_{i A} = occ_i sum_{mu in A} sum_nu C_{mu i} C_{nu i} S_{mu nu}``
    """
    active = np.where(occ > 1e-8)[0]
    if len(active) == 0:
        return np.zeros((0, len(molecule)))
    Ca = C[:, active]
    contrib = Ca * (S @ Ca) * occ[active]        # (n_basis, n_active)
    atom_of = np.array([f.atom_idx for f in basis])
    pops = np.zeros((len(molecule), len(active)))
    np.add.at(pops, atom_of, contrib)
    return pops.T


def wiberg_bond_orders(P: np.ndarray, S: np.ndarray, basis, molecule) -> np.ndarray:
    """``B_AB = sum_{mu in A, nu in B} (P S)_{mu nu} (P S)_{nu mu}``."""
    PS = P @ S
    M = PS * PS.T
    atom_of = np.array([f.atom_idx for f in basis])
    n = len(molecule)
    B = np.zeros((n, n))
    np.add.at(B, (atom_of[:, None], atom_of[None, :]), M)
    return B


def basis_function_weights(C: np.ndarray, occ: np.ndarray, S: np.ndarray,
                           selector) -> np.ndarray:
    """Fraction of each MO carried by the basis functions ``selector`` picks.

    Used to check that pseudo-hydrogen terminators contribute no states inside
    the gap: pass a mask selecting the terminator functions and look for
    orbitals with a large weight near the frontier.
    """
    mask = np.asarray(selector, dtype=bool)
    contrib = C * (S @ C)
    total = contrib.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(total) > 1e-12, contrib[mask].sum(axis=0) / total, 0.0)
