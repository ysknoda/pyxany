"""``pyxany validate`` -- parse and check an input file without running anything."""

from __future__ import annotations

from ..io.yaml_input import dump_config
from ..parallel import mpi_print
from . import EXIT_OK
from ._shared import build_molecule_basis, prepare, resource_estimate


def execute(args) -> int:
    cfg, molecule = prepare(args)
    basis = build_molecule_basis(cfg, molecule)
    mpi_print("  input is valid\n")
    mpi_print(f"  {molecule.summary()}")
    mpi_print(f"  structure file : {cfg.structure_path()}")
    mpi_print(f"  basis          : {cfg.basis.set.upper()}  "
              f"({len(basis)} functions, max l = {max(f.l for f in basis)})")
    pseudo = [a for a in molecule if a.is_pseudo]
    if pseudo:
        zetas = getattr(cfg, "_pseudo_zeta", {})
        mpi_print(f"  pseudo-hydrogen: {len(pseudo)} atoms")
        for label, zeta in sorted(zetas.items()):
            alpha = next(a.Z for a in pseudo if a.symbol == label)
            mpi_print(f"      {label:<8} alpha = {alpha:.6f}   derived zeta = {zeta:.5f}")
    mpi_print("")
    mpi_print(resource_estimate(cfg, molecule, basis))
    mpi_print("")
    mpi_print("  resolved settings")
    mpi_print("  " + "-" * 62)
    for line in dump_config(cfg).splitlines():
        mpi_print("  " + line)
    return EXIT_OK
