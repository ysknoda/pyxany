"""``pyxany forces`` -- forces only, with an optional analytic/FD comparison."""

from __future__ import annotations

from ..parallel import mpi_print
from ..scf import run_scf
from ..timing import REGISTRY, timer
from . import EXIT_OK, EXIT_SCF
from ._shared import build_molecule_basis, out_path, prepare, print_forces, report_timing


def execute(args) -> int:
    REGISTRY.reset()
    with timer("total"):
        cfg, molecule = prepare(args)
        if args.method:
            cfg.forces.method = args.method
        if args.fd_step:
            cfg.forces.fd_step = args.fd_step
        basis = build_molecule_basis(cfg, molecule)
        result = run_scf(molecule, basis, cfg, verbose=cfg.runtime.verbosity,
                         restart=args.restart, checkpoint_path=out_path(cfg, "_scf.npz"))

        if args.compare:
            from ..forces import compare_with_fd

            mpi_print("\n  comparing analytic forces against central finite differences "
                      f"(step {cfg.forces.fd_step} Bohr)")
            cmp = compare_with_fd(result, cfg, step=cfg.forces.fd_step)
            _print_comparison(molecule, cmp)
            forces = cmp["analytic"]
            _print_radial_estimate(molecule, cfg, result, forces)
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
    mpi_print("  The analytic forces include the grid-motion terms, so this difference only")
    mpi_print("  checks the implementation (finite-difference step and SCF tolerance). It")
    mpi_print("  does not measure the grid's own error; the radial estimate below does.")


#: radial points added for the grid-error estimate of `forces --compare`
RADIAL_STEP = 15


def _print_radial_estimate(molecule, cfg, result, forces) -> None:
    """Change of the analytic forces when grid.n_r grows by RADIAL_STEP.

    The force error of a grid is dominated by its radial quadrature: NH2 (UKS)
    at n_r 30 / n_a 12 is 6.5e-3 Ha/Bohr off an (80, 26) reference, 1.5e-4 at
    n_r 45, and hardly changes with n_a (P0 re-review, ADR 0012), while the
    analytic-FD difference on one grid is 2.7e-4. This compares two radial
    grids, which is what an optimization's f_max has to be judged against.
    """
    import copy

    import numpy as np

    from ..forces import compute_forces

    finer = copy.deepcopy(cfg)
    finer.grid.n_r = cfg.grid.n_r + RADIAL_STEP
    mol2 = molecule.copy()
    basis2 = build_molecule_basis(finer, mol2)
    res2 = run_scf(mol2, basis2, finer, verbose=0,
                   initial={"P": result.P, "P_a": result.P_a, "P_b": result.P_b})
    f2 = compute_forces(res2, finer)
    change = float(np.abs(f2 - forces).max())
    mpi_print(f"  grid error estimate: max |F(n_r {finer.grid.n_r}) - F(n_r {cfg.grid.n_r})| "
              f"= {change:.3e} Ha/Bohr")
    if cfg.optimize.f_max < change:
        mpi_print(f"  [warning] optimize.f_max ({cfg.optimize.f_max:g}) is below this "
                  f"estimate: an optimization would chase grid noise. Raise grid.n_r or "
                  f"f_max.")
    mpi_print("")
