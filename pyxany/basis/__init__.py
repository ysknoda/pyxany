"""Basis-set construction and evaluation."""

from __future__ import annotations

from .evaluate import chi_grad_matrix, chi_matrix
from .gto import GTO_KEYS, build_gto_basis
from .harmonics import real_sph_harm, solid_harmonic, solid_harmonic_grad
from .sto import STO_KEYS, build_sto_basis
from .types import GTO, NAO, STO, BasisSet


def build_basis(molecule, config, pseudo_zeta=None) -> BasisSet:
    """Build the basis for ``molecule`` according to ``config.basis.set``.

    ``pseudo_zeta`` maps pseudo-hydrogen species labels to the 1s exponent
    derived for their fractional charge.  When it is omitted and the molecule
    contains pseudo-hydrogens, it is derived here rather than left empty: the
    exponents depend only on the fractional charges, so every caller would
    otherwise have to remember to pass them, and the ones that forgot
    (``forces.optimise`` and ``forces.finite_difference``) made
    ``pyxany opt --freeze-core`` -- the advertised way to relax a passivated
    cluster -- fail with a KeyError on every passivated example.
    """
    key = config.basis.set.lower().strip()
    if pseudo_zeta is None and any(atom.is_pseudo for atom in molecule):
        from .pseudo_h import derive_exponents

        pseudo_zeta = derive_exponents(molecule, config.scf.alpha,
                                       config.grid.atom["n_points"])
    if key in STO_KEYS:
        functions = build_sto_basis(molecule, key, pseudo_zeta)
    elif key in GTO_KEYS:
        functions = build_gto_basis(molecule, key, pseudo_zeta)
    elif key == "nao":
        from .nao import build_nao_basis

        functions = build_nao_basis(molecule, config, pseudo_zeta)
    else:  # pragma: no cover - config.validate rejects this first
        raise ValueError(f"unknown basis set '{config.basis.set}'")

    basis = BasisSet(functions, n_atoms=len(molecule))
    for ai, atom in enumerate(molecule):
        atom.basis = [f for f in basis if f.atom_idx == ai]
    _check_saturation(molecule, basis)
    return basis


def _check_saturation(molecule, basis) -> None:
    n_occ = molecule.n_electrons / 2.0
    if len(basis) < n_occ:
        raise ValueError(
            f"basis too small: {len(basis)} functions cannot hold "
            f"{molecule.n_electrons:g} electrons. Use a larger basis set."
        )


__all__ = ["build_basis", "BasisSet", "STO", "GTO", "NAO",
           "build_sto_basis", "build_gto_basis", "STO_KEYS", "GTO_KEYS",
           "chi_matrix", "chi_grad_matrix",
           "solid_harmonic", "solid_harmonic_grad", "real_sph_harm"]
