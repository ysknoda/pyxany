"""Numerical atomic orbital (NAO) basis: the atomic solution used directly."""

from __future__ import annotations

from functools import lru_cache

from ..parallel import mpi_print
from .atomic_solver import solve_atom
from .harmonics import lm_pairs
from .pseudo_h import pseudo_hydrogen_solution
from .types import NAO


@lru_cache(maxsize=128)
def _atom_solution(element: str, n_points: int, xc_alpha: float):
    return solve_atom(element, alpha=xc_alpha, n_points=n_points)


def build_nao_basis(molecule, config, pseudo_zeta=None) -> list:
    """One NAO per (n, l, m) of each atom's self-consistent atomic solution."""
    functions: list = []
    xc_alpha = config.scf.alpha
    n_points = max(int(config.basis.nao_n_r), 500)

    for ai, atom in enumerate(molecule):
        if atom.is_pseudo:
            sol = pseudo_hydrogen_solution(round(float(atom.Z), 8), xc_alpha)
        else:
            sol = _atom_solution(atom.element, n_points, xc_alpha)
            if config.runtime.verbosity > 1:
                mpi_print(f"  [NAO] {atom.element}: {len(sol.orbitals)} shells, "
                          f"{'converged' if sol.converged else 'NOT converged'}")
        for (n, l), orb in sorted(sol.orbitals.items()):
            if orb.spline is None:
                continue
            for m in lm_pairs(l):
                functions.append(NAO(
                    l=l, m=m, r_grid=orb.r, radial_grid=orb.radial,
                    spline_tck=orb.spline, atom_idx=ai,
                    label=f"{atom.symbol}{ai + 1}_{n}{'spdf'[l]}{m:+d}_nao",
                ))
    return functions
