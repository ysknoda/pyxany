"""Mulliken populations and Wiberg bond orders (vectorized)."""

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


def mayer_bond_orders(P: np.ndarray, S: np.ndarray, basis, molecule,
                      P_alpha: np.ndarray | None = None,
                      P_beta: np.ndarray | None = None) -> np.ndarray:
    """``B_AB = sum_{mu in A, nu in B} (P S)_{mu nu} (P S)_{nu mu}``.

    This is the **Mayer** bond order, which is what the expression above is.
    The *Wiberg* index is ``sum P^2_{mu nu}`` and coincides with it only in an
    orthonormal basis (Loewdin-orthogonalized, or a ZDO/CNDO scheme).  This
    code uses a decidedly non-orthogonal STO/GTO basis -- ``pyxany.scf.linalg``
    exists precisely to handle ``S`` -- so the two are not interchangeable
    here, and the name matters.

    For an open-shell system the correct expression is
    ``2 sum_sigma (P_sigma S)_{mu nu} (P_sigma S)_{nu mu}``, which reduces to
    the closed-shell form only when ``P_alpha == P_beta``.  Pass the two spin
    density matrices to get it; with only the total ``P`` the closed-shell
    formula is used (CH3: 0.94439 against a correct 0.94233 for C-H).
    """
    atom_of = np.array([f.atom_idx for f in basis])
    n = len(molecule)
    if P_alpha is not None and P_beta is not None:
        M = sum((Ps * Ps.T) for Ps in (P_alpha @ S, P_beta @ S))
        M = 2.0 * M
    else:
        PS = P @ S
        M = PS * PS.T
    B = np.zeros((n, n))
    np.add.at(B, (atom_of[:, None], atom_of[None, :]), M)
    return B


def wiberg_bond_orders(P: np.ndarray, S: np.ndarray, basis, molecule,
                       P_alpha: np.ndarray | None = None,
                       P_beta: np.ndarray | None = None) -> np.ndarray:
    """Deprecated alias for :func:`mayer_bond_orders` -- see its docstring for
    why the Mayer name is the correct one in a non-orthogonal basis."""
    return mayer_bond_orders(P, S, basis, molecule, P_alpha, P_beta)


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


def _sqrt_overlap(S: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(S)
    return (U * np.sqrt(np.maximum(w, 0.0))) @ U.T


def loewdin(P: np.ndarray, S: np.ndarray, basis, molecule) -> dict:
    """Loewdin gross populations and charges: ``diag(S^1/2 P S^1/2)`` summed per atom.

    Unlike Mulliken, each orbital's share stays within [0, 1]: Mulliken gave a
    pseudo-hydrogen-terminated MgO6 cluster's LUMO a terminator weight of 1.55,
    Loewdin 0.72.
    """
    Sh = _sqrt_overlap(S)
    atom_of = np.array([f.atom_idx for f in basis])
    gross = np.zeros(len(molecule))
    np.add.at(gross, atom_of, np.diag(Sh @ P @ Sh))
    return {"gross": gross, "charges": molecule.charges - gross, "atom_of_basis": atom_of}


def loewdin_orbital_weights(C: np.ndarray, S: np.ndarray, selector) -> np.ndarray:
    """Fraction of each MO on the basis functions ``selector`` picks, in [0, 1].

    ``sum_{mu selected} (S^1/2 C)_{mu i}^2`` -- the orbital expressed in the
    symmetrically orthogonalized basis, so the weights of all functions add up
    to the orbital's norm, 1, and none can be negative.
    """
    mask = np.asarray(selector, dtype=bool)
    LC = _sqrt_overlap(S) @ C
    return np.sum(LC[mask] ** 2, axis=0)


def core_gap(eps: np.ndarray, occ: np.ndarray, weights: np.ndarray,
             max_weight: float = 0.5) -> dict:
    """Gap between the frontier orbitals that are *not* terminator states.

    ``weights`` is each orbital's terminator fraction (Loewdin). The core HOMO
    (LUMO) is the highest occupied (lowest empty) orbital with weight below
    ``max_weight``. In small passivated clusters a terminator-dominated empty
    state can sit below the cluster's own conduction states, so the plain
    HOMO-LUMO gap is not the gap of the cluster interior.
    """
    finite = np.isfinite(eps)
    occupied = finite & (occ > 1e-8)
    empty = finite & ~occupied
    core = weights < max_weight
    homo_i = np.where(occupied)[0][np.argmax(eps[occupied])] if occupied.any() else None
    lumo_i = np.where(empty)[0][np.argmin(eps[empty])] if empty.any() else None
    core_occ = np.where(occupied & core)[0]
    core_emp = np.where(empty & core)[0]
    c_homo = core_occ[np.argmax(eps[core_occ])] if len(core_occ) else None
    c_lumo = core_emp[np.argmin(eps[core_emp])] if len(core_emp) else None
    return {
        "homo": homo_i, "lumo": lumo_i, "core_homo": c_homo, "core_lumo": c_lumo,
        "gap": (eps[lumo_i] - eps[homo_i]) if homo_i is not None and lumo_i is not None else np.nan,
        "core_gap": (eps[c_lumo] - eps[c_homo]) if c_homo is not None and c_lumo is not None
        else np.nan,
        "n_terminator_states_below_core_lumo": int(np.sum(empty & ~core & (eps < eps[c_lumo])))
        if c_lumo is not None else 0,
    }


def overlap_population(P: np.ndarray, S: np.ndarray, basis, molecule) -> np.ndarray:
    """Mulliken bond overlap population ``n_AB = sum_{mu in A, nu in B} 2 P_mu,nu S_mu,nu``.

    Symmetric ``(n_atoms, n_atoms)``; the diagonal holds the net population
    ``sum_{mu, nu in A} P S`` so that ``trace + sum_{A<B} n_AB`` is the electron
    count -- the sum rule the tests check. Positive off-diagonal entries are
    bonding, negative antibonding.
    """
    atom_of = np.array([f.atom_idx for f in basis])
    M = P * S
    out = np.zeros((len(molecule), len(molecule)))
    np.add.at(out, (atom_of[:, None], atom_of[None, :]), M)
    off = ~np.eye(len(molecule), dtype=bool)
    out[off] *= 2.0
    return out


def mo_overlap_population(C: np.ndarray, S: np.ndarray, basis, pairs) -> np.ndarray:
    """Overlap population of each MO for each atom pair, per electron: ``(n_mo, n_pairs)``.

    ``OP_i(A, B) = 2 sum_{mu in A, nu in B} C_mu,i C_nu,i S_mu,nu``. Weighted by
    the occupations and summed over orbitals it reproduces
    :func:`overlap_population` -- the input for COOP-type plots.
    """
    atom_of = np.array([f.atom_idx for f in basis])
    SC = S @ C
    out = np.zeros((C.shape[1], len(pairs)))
    for k, (a, b) in enumerate(pairs):
        ma, mb = atom_of == a, atom_of == b
        # sum_{mu in A, nu in B} C_mu S_mu,nu C_nu  ==  sum_{mu in A} C_mu (S_{:,B} C_B)_mu
        out[:, k] = 2.0 * np.einsum("mi,mi->i", C[ma], S[np.ix_(ma, mb)] @ C[mb])
    return out


def bonded_pairs(bond_orders: np.ndarray, min_bond_order: float = 0.2) -> list[tuple[int, int]]:
    """Atom pairs ``(A, B)``, ``A < B``, whose (Mayer) bond order reaches the threshold."""
    n = len(bond_orders)
    return [(a, b) for a in range(n) for b in range(a + 1, n)
            if bond_orders[a, b] >= min_bond_order]


def loewdin_orbital_groups(C: np.ndarray, S: np.ndarray, labels) -> dict:
    """Loewdin weight of every MO on each group of basis functions: ``{label: (n_mo,)}``.

    ``labels`` gives one group label per basis function (e.g. ``"Ga d"``); the
    returned weights of all groups add up to 1 for every orbital.
    """
    LC = _sqrt_overlap(S) @ C
    sq = LC**2
    labels = np.asarray(labels)
    return {lab: sq[labels == lab].sum(axis=0) for lab in dict.fromkeys(labels)}
