"""``pyxany gridcheck`` -- how accurately the grid integrates the density."""

from __future__ import annotations

import time

import numpy as np

from ..basis.evaluate import chi_matrix
from ..grid import build_molecular_grid
from ..parallel import mpi_print
from ..scf.guess import atomic_superposition, rescale_to_electron_count
from ..integrals.overlap import build_overlap_kinetic
from ..timing import REGISTRY, timer
from . import EXIT_OK
from ._shared import build_molecule_basis, prepare, report_timing


def execute(args) -> int:
    REGISTRY.reset()
    with timer("total"):
        cfg, code = _gridcheck(args)
    report_timing(cfg)
    return code


def _gridcheck(args):
    """Returns ``(cfg, exit_code)`` so the caller can report timings."""
    cfg, molecule = prepare(args)
    basis = build_molecule_basis(cfg, molecule)
    levels = _parse_levels(args.levels)

    mpi_print("  GRID QUALITY")
    mpi_print("  " + "-" * 74)
    mpi_print(f"  {molecule.summary()}")
    mpi_print(f"  target: int rho dV = {molecule.n_electrons:g} electrons")
    mpi_print("")
    mpi_print(f"  {'n_r x n_a':>12} {'points':>10} {'int rho dV':>14} "
              f"{'error':>12} {'rel error':>11} {'chi MB':>9} {'time s':>9}")
    mpi_print("  " + "-" * 84)

    with timer("overlap"):
        S, _ = build_overlap_kinetic(molecule, basis,
                                     screening_tol=cfg.grid.screening_tol)
    P = rescale_to_electron_count(atomic_superposition(molecule, basis), S,
                                  molecule.n_electrons)
    for n_r, n_a in levels:
        label = f"{n_r} x {n_a}"
        elapsed = time.perf_counter()
        with timer(f"grid {label}"):
            grid = build_molecular_grid(molecule, n_r, n_a,
                                        partition=cfg.grid.partition,
                                        radial_by_element=cfg.grid.radial_by_element)
        with timer(f"chi {label}"):
            chi = chi_matrix(basis, molecule.positions, grid.local_points)
        with timer(f"density {label}"):
            rho = np.einsum("km,mn,kn->k", chi, P, chi, optimize=True)
            total = float(np.dot(grid.local_weights, rho))
        elapsed = time.perf_counter() - elapsed
        err = total - molecule.n_electrons
        mpi_print(f"  {label:>12} {len(grid):>10,} {total:>14.8f} "
                  f"{err:>12.2e} {abs(err) / molecule.n_electrons:>11.2e} "
                  f"{grid.chi_memory_mb(len(basis)):>9.1f} {elapsed:>9.2f}")
    mpi_print("  " + "-" * 84)
    mpi_print("  (measured with the atomic-superposition density, so this isolates the")
    mpi_print("   quadrature error from the SCF; a converged density behaves the same.)")
    return cfg, EXIT_OK


def _parse_levels(spec: str):
    out = []
    for token in spec.split(","):
        n_r, _, n_a = token.strip().lower().partition("x")
        out.append((int(n_r), int(n_a)))
    return out
