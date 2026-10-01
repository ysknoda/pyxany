"""``pyxany opt`` -- geometry optimisation."""

from __future__ import annotations

from ..forces import optimise_geometry
from ..io.report import format_results
from ..io.results import result_dict, write_json, write_text
from ..io.xyz import write_xyz
from ..parallel import is_root, mpi_print
from ..timing import REGISTRY, timer
from . import EXIT_OK, EXIT_OPT
from ._shared import (
    build_molecule_basis,
    out_path,
    prepare,
    report_timing,
    resource_estimate,
)


def execute(args) -> int:
    REGISTRY.reset()
    with timer("total"):
        cfg, molecule = prepare(args)
        cfg.optimise.enabled = True
        if args.freeze_core:
            cfg.optimise.freeze = "core"
        if args.max_steps:
            cfg.optimise.max_steps = args.max_steps
        cfg.validate("<cli>")

        if args.dry_run:
            mpi_print(resource_estimate(cfg, molecule, build_molecule_basis(cfg, molecule)))
            return EXIT_OK

        result = optimise_geometry(molecule, cfg, verbose=cfg.runtime.verbosity)
        scf = result.scf

        if is_root():
            write_xyz(out_path(cfg, "_final.xyz"), result.molecule,
                      f"{result.molecule.name}  optimised  E = {result.energy:.9f} Ha")
            text = format_results(scf, cfg, forces=result.forces)
            write_text(out_path(cfg, "_results.txt"), text)
            if cfg.output.json:
                payload = result_dict(scf, cfg, forces=result.forces, extra={
                    "optimisation": {
                        "converged": bool(result.converged),
                        "n_evaluations": result.n_steps,
                        "energies": [float(t["E"]) for t in result.trajectory],
                        "max_force": [float(t["f_max"]) for t in result.trajectory],
                    }})
                write_json(out_path(cfg, "_results.json"), payload)

    report_timing(cfg)
    if is_root() and cfg.runtime.verbosity:
        mpi_print(f"\n  output written to {out_path(cfg, '')}*\n")
    return EXIT_OK if result.converged else EXIT_OPT
