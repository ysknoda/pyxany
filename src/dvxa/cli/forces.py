"""``dvxa forces`` -- forces only, with an optional analytic/FD comparison."""

from __future__ import annotations

import numpy as np

from ..parallel import mpi_print
from ..scf import run_scf
from ..timing import REGISTRY, timer
from . import EXIT_OK, EXIT_SCF
from ._shared import build_molecule_basis, prepare, print_forces, report_timing


def execute(args) -> int:
    REGISTRY.reset()
    with timer("total"):
        cfg, molecule = prepare(args)
        if args.method:
            cfg.forces.method = args.method
        if args.fd_step:
            cfg.forces.fd_step = args.fd_step
        basis = build_molecule_basis(cfg, molecule)
        result = run_scf(molecule, basis, cfg, verbose=cfg.runtime.verbosity)

        if args.compare:
            from ..forces import compare_with_fd

            mpi_print("\n  comparing analytic forces against central finite differences "
                      f"(step {cfg.forces.fd_step} Bohr)")
            cmp = compare_with_fd(result, cfg, step=cfg.forces.fd_step)
            _print_comparison(molecule, cmp)
            forces = cmp["analytic"]
        else:
            from ..forces import compute_forces, compute_forces_fd

            forces = (compute_forces_fd(molecule, cfg, step=cfg.forces.fd_step,
                                        verbose=cfg.runtime.verbosity)
                      if cfg.forces.method == "finite_difference"
                      else compute_forces(result, cfg))
            print_forces(molecule, forces)

    report_timing(cfg)
    return EXIT_OK if result.converged else EXIT_SCF


def _print_comparison(molecule, cmp) -> None:
    a, f = cmp["analytic"], cmp["finite_difference"]
    mpi_print("")
    mpi_print(f"  {'atom':>8} {'axis':>5} {'analytic':>14} {'finite diff':>14} {'diff':>12}")
    mpi_print("  " + "-" * 58)
    for i in cmp["atoms"]:
        for ax, name in enumerate("xyz"):
            mpi_print(f"  {molecule[i].symbol + str(i + 1):>8} {name:>5} "
                      f"{a[i, ax]:>14.8f} {f[i, ax]:>14.8f} {a[i, ax] - f[i, ax]:>12.2e}")
    mpi_print("  " + "-" * 58)
    mpi_print(f"  max |analytic - fd|            {cmp['max_abs_diff']:.3e} Ha/Bohr")
    mpi_print(f"  rms |analytic - fd|            {cmp['rms_diff']:.3e} Ha/Bohr")
    mpi_print(f"  max |F| (scale for comparison) {cmp['max_abs_force']:.3e} Ha/Bohr")
    mpi_print(f"  |sum F| analytic / fd          {cmp['translation_residual_analytic']:.2e}"
              f" / {cmp['translation_residual_fd']:.2e}")
    mpi_print("")
    mpi_print("  The residual is dominated by the neglected derivative of the Becke cell")
    mpi_print("  weights and by grid quadrature error; both shrink as grid.n_r / grid.n_a")
    mpi_print("  are increased.")
