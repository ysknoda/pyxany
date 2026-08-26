"""``dvxa run`` -- single-point SCF with energy, forces and properties."""

from __future__ import annotations

import json

from ..io.report import format_results
from ..io.results import result_dict, write_json, write_npz, write_text
from ..io.xyz import write_xyz
from ..parallel import is_root, mpi_print
from ..properties import dipole_moment
from ..scf import run_scf
from ..timing import REGISTRY, timer
from . import EXIT_OK, EXIT_SCF
from ._shared import (
    build_molecule_basis,
    out_path,
    prepare,
    print_forces,
    report_timing,
    resource_estimate,
)


def execute(args) -> int:
    REGISTRY.reset()
    with timer("total"):
        cfg, molecule = prepare(args)
        basis = build_molecule_basis(cfg, molecule)

        if args.dry_run:
            mpi_print(resource_estimate(cfg, molecule, basis))
            return EXIT_OK

        verbose = cfg.runtime.verbosity
        result = run_scf(molecule, basis, cfg, verbose=verbose)

        forces = None
        if cfg.forces.enabled:
            forces = _forces(result, cfg)
            if verbose:
                print_forces(molecule, forces)

        dipole = None
        if cfg.properties.dipole:
            dipole = dipole_moment(molecule, result.grid, result.terms.density(result.P))

        text = format_results(result, cfg, forces=forces, dipole=dipole)
        if verbose:
            mpi_print("")
            mpi_print(text)

        if is_root():
            write_text(out_path(cfg, "_results.txt"), text)
            write_xyz(out_path(cfg, "_final.xyz"), molecule,
                      f"{molecule.name}  E = {result.total_energy:.9f} Ha")
            payload = result_dict(result, cfg, forces=forces, dipole=dipole)
            if cfg.output.json:
                write_json(out_path(cfg, "_results.json"), payload)
            if cfg.output.npz:
                write_npz(out_path(cfg, "_matrices.npz"), result)
            if cfg.output.figure:
                _figure(result, cfg, forces)
            if args.json:
                print(json.dumps(payload, indent=2))

    report_timing(cfg)
    if is_root() and cfg.output.timing:
        write_json(out_path(cfg, "_timing.json"), REGISTRY.to_dict())
    if is_root() and cfg.runtime.verbosity:
        mpi_print(f"\n  output written to {out_path(cfg, '')}*\n")
    return EXIT_OK if result.converged else EXIT_SCF


def _forces(result, cfg):
    from ..forces import compute_forces, compute_forces_fd

    if cfg.forces.method == "finite_difference":
        return compute_forces_fd(result.molecule, cfg, step=cfg.forces.fd_step,
                                 verbose=cfg.runtime.verbosity)
    return compute_forces(result, cfg)


def _figure(result, cfg, forces) -> None:
    try:
        from ..plotting import summary_figure
    except ImportError:
        mpi_print("  [note] matplotlib is not installed; skipping the figure")
        return
    summary_figure(result, cfg, out_path(cfg, "_results.png"), forces=forces)
