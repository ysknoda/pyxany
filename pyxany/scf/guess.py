"""Initial density matrices from a superposition of atomic occupations."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..data import elements as elem


def atomic_superposition(molecule, basis) -> np.ndarray:
    """Diagonal P whose atomic blocks carry the free-atom shell occupations.

    Each subshell's electrons go to the basis functions of that shell (n, l),
    shared equally (a double-zeta pair each get half), when the functions carry
    a principal quantum number (STO). Sharing an atom's electrons over *every*
    function of the same l instead -- what the guess used to do, and still does
    for Gaussian sets, which have no n -- puts half of Si's 1s electrons into
    the diffuse 3s functions: 2700 Ha too high for Si47H60, from which the SCF
    never recovered. Electrons of a shell the basis lacks fall back to the
    functions of the same l. A pseudo-hydrogen spreads its fractional charge
    over its functions.
    """
    n = len(basis)
    P = np.zeros((n, n))
    by_atom: dict[int, list[int]] = {}
    for mu, f in enumerate(basis):
        by_atom.setdefault(f.atom_idx, []).append(mu)

    for ai, idx in by_atom.items():
        atom = molecule[ai]
        if atom.is_pseudo:
            for mu in idx:
                P[mu, mu] = atom.Z / len(idx)
            continue
        totals = elem._subshell_totals(atom.element)
        shells = {}
        has_n = all(getattr(basis[mu], "n", None) is not None for mu in idx)
        if has_n:
            for mu in idx:
                shells.setdefault((basis[mu].n, basis[mu].l), []).append(mu)
        leftover: Counter = Counter()
        for (nn, l), ne in totals.items():
            members = shells.get((nn, l))
            if members:
                for mu in members:
                    P[mu, mu] += ne / len(members)
            else:
                leftover[l] += ne
        for l, ne in leftover.items():
            same_l = [mu for mu in idx if basis[mu].l == l]
            for mu in same_l:
                P[mu, mu] += ne / len(same_l)
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


def sad_density(molecule, basis, xc_alpha=0.7, n_points: int = 2000,
                grid_size: tuple[int, int] = (80, 14)) -> np.ndarray:
    """Superposition of atomic densities, each projected onto the molecular basis.

    For every species the spherical X-alpha atom is solved on its logarithmic
    grid (pseudo-hydrogens with their fractional charge), each occupied atomic
    orbital ``R_nl Y_lm`` is projected onto that atom's basis functions,
    ``c = S_AA^-1 <chi|phi>``, and the block ``sum occ/(2l+1) c c^T`` (a
    spherical average over m) is placed on every atom of the species. Unlike a
    diagonal occupation guess this carries the cross terms of contracted and
    multiple-zeta functions: the first SCF energy of H2O/DZ is within a few
    hartree of the converged one instead of 20 Ha above it. The overlaps are
    quadratures on a fine one-atom DV grid (``grid_size`` = n_r, n_a; input
    ``grid.guess``), so no normalisation convention of the basis or the
    harmonics enters. ``n_points`` is the atom solver's grid (``grid.atom``).
    ``xc_alpha`` is a number or a function of the element symbol
    (``scf.alpha_for``); pseudo-hydrogens use hydrogen's value.
    """
    import copy

    from scipy.interpolate import splev

    from ..basis.evaluate import chi_matrix
    from ..basis.nao import _atom_solution
    from ..basis.pseudo_h import pseudo_hydrogen_solution
    from ..basis.solid_harmonics import real_spherical_harmonics
    from ..grid import build_molecular_grid
    from ..molecule import Molecule

    n = len(basis)
    P = np.zeros((n, n))
    by_atom: dict[int, list[int]] = {}
    for mu, f in enumerate(basis):
        by_atom.setdefault(f.atom_idx, []).append(mu)
    blocks: dict[str, np.ndarray] = {}

    for ai, idx in by_atom.items():
        atom = molecule[ai]
        if atom.symbol not in blocks:
            local = []
            for mu in idx:
                g = copy.copy(basis[mu])
                g.atom_idx = 0
                local.append(g)
            lone = copy.copy(atom)
            lone.pos = np.zeros(3)
            grid = build_molecular_grid(Molecule([lone], name=atom.symbol), *grid_size,
                                        radial_by_element=False)
            chi = chi_matrix(local, np.zeros((1, 3)), grid.points)
            wchi = chi * grid.weights[:, None]
            S_loc = chi.T @ wchi
            r = np.linalg.norm(grid.points, axis=1)
            a_x = xc_alpha("H" if atom.is_pseudo else atom.element) if callable(xc_alpha) \
                else xc_alpha
            sol = (pseudo_hydrogen_solution(round(float(atom.Z), 8), a_x, n_points)
                   if atom.is_pseudo else _atom_solution(atom.element, n_points, a_x))
            Pa = np.zeros((len(idx), len(idx)))
            for (_, l), orb in sol.orbitals.items():
                if orb.occupation <= 0 or orb.spline is None:
                    continue
                R = np.where(r <= orb.r[-1], splev(np.maximum(r, orb.r[0]), orb.spline), 0.0)
                Y = real_spherical_harmonics(l, grid.points[:, 0], grid.points[:, 1],
                                             grid.points[:, 2], r)[:, l * l:(l + 1) ** 2]
                b = wchi.T @ (R[:, None] * Y)                     # (n_A, 2l+1)
                c = np.linalg.lstsq(S_loc, b, rcond=1e-10)[0]
                Pa += orb.occupation / (2 * l + 1) * (c @ c.T)
            blocks[atom.symbol] = Pa
        P[np.ix_(idx, idx)] = blocks[atom.symbol]
    return P
