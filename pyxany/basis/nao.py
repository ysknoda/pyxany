"""Numerical atomic orbital (NAO) basis: the atomic solution used directly.

Two properties of the generating atoms are worth knowing before reading
anything physical into an NAO or its eigenvalue.

**The atomic solver is spin-restricted.**  Orbitals come from the
spherically-averaged, spin-averaged atom, so for an open-shell element (C, N,
O, the transition metals) they are not that atom's open-shell orbitals.  That
is a deliberate and normal choice for a *basis generator* -- the molecular SCF
is free to be unrestricted -- but it means an NAO is not a physical atomic
orbital of the free atom.

**Slater X-alpha carries its full self-interaction error.**  An electron is
repelled by the whole density including itself, and the X-alpha exchange does
not cancel that.  Hydrogen 1s comes out at about -0.204 Ha instead of the exact
-0.5, so H NAOs are markedly more diffuse than the true hydrogen 1s.  This is
expected X-alpha behaviour, not a defect in the solver -- which reproduces the
hydrogenic spectrum to 1e-11 when given the bare -Z/r potential -- but NAO
eigenvalues must not be read as ionisation potentials or orbital energies.
"""

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
    alpha_for = config.scf.alpha_for
    n_points = int(config.grid.atom["n_points"])

    for ai, atom in enumerate(molecule):
        if atom.is_pseudo:
            sol = pseudo_hydrogen_solution(round(float(atom.Z), 8), alpha_for("H"), n_points)
        else:
            sol = _atom_solution(atom.element, n_points, alpha_for(atom.element))
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
