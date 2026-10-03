"""Helpers shared by the CLI subcommands."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import Config
from ..io.yaml_input import load_config, load_molecule
from ..parallel import is_root, mpi_print
from ..timing import REGISTRY
from .overrides import parse_overrides


def prepare(args) -> tuple[Config, object]:
    """Load the config and molecule, applying every command-line override."""
    overrides = parse_overrides(getattr(args, "overrides", None))
    if getattr(args, "basis", None):
        overrides["basis.set"] = args.basis
    if getattr(args, "numba", None) is False:
        os.environ["PYXANY_DISABLE_NUMBA"] = "1"
    cfg = load_config(args.input, overrides)
    if getattr(args, "outdir", None):
        cfg.output.dir = args.outdir
    if getattr(args, "no_figure", False):
        cfg.output.figure = False
    if getattr(args, "timing", None) is not None:
        cfg.output.timing = args.timing
    REGISTRY.enabled = cfg.output.timing
    cfg.runtime.verbosity = verbosity(args)

    molecule = load_molecule(cfg)
    _apply_passivation_basis(cfg, molecule)
    return cfg, molecule


def _apply_passivation_basis(cfg, molecule) -> None:
    """Derive 1s exponents for any pseudo-hydrogen species present."""
    if not any(a.is_pseudo for a in molecule):
        return
    from ..basis.pseudo_h import derive_exponents

    cfg._pseudo_zeta = derive_exponents(molecule, cfg.scf.alpha_for("H"), cfg.grid.atom["n_points"])


def build_molecule_basis(cfg, molecule):
    from ..basis import build_basis

    return build_basis(molecule, cfg, getattr(cfg, "_pseudo_zeta", None))


def verbosity(args) -> int:
    if getattr(args, "quiet", False):
        return 0
    return 1 + int(getattr(args, "verbose", 0))


def output_dir(cfg) -> Path:
    path = Path(cfg.output.dir)
    if is_root():
        path.mkdir(parents=True, exist_ok=True)
    return path


def out_path(cfg, suffix: str) -> Path:
    return output_dir(cfg) / f"{cfg.output.prefix}{suffix}"


def resource_estimate(cfg, molecule, basis) -> str:
    """What ``--dry-run`` prints: every grid, memory and a rough cost estimate."""
    from ..grid.molecular import grid_from_config
    from ..grid.summary import grid_summary, summary_lines
    from ..scf.grid_terms import SparseBlocks, choose_strategy

    grid = grid_from_config(molecule, cfg.grid)
    n_pts = len(grid)
    nb = len(basis)
    chi_mb = n_pts * nb * 8 / 1024**2
    n_lm = (cfg.grid.l_max + 1) ** 2
    hartree_ops = len(molecule) * n_pts * n_lm
    lines = [
        "  RESOURCE ESTIMATE",
        "  " + "-" * 62,
        f"  atoms                {len(molecule)}",
        f"  basis functions      {nb}",
        f"  electrons            {molecule.n_electrons:g}",
        f"  grid points          {n_pts:,}",
        f"  chi matrix           {chi_mb:,.1f} MB  (dense, one rank)",
        f"  S/T pairs            {_n_pairs(molecule, basis, cfg)} of "
        f"{len(molecule) * (len(molecule) - 1) // 2} after screening",
        f"  Hartree work         {hartree_ops:,.0f} harmonic evaluations per iteration"
        f"  (l_max = {cfg.grid.l_max})",
        f"  nb x nb matrices     {nb * nb * 8 / 1024**2:.1f} MB each",
        "",
    ]
    budget = cfg.grid.memory_budget_mb
    sparse = None
    if cfg.grid.basis_cutoff > 0:
        # layout only (budget 0 keeps it from evaluating chi)
        sparse = SparseBlocks(basis, molecule, grid.local_points, cfg.grid.basis_cutoff, 0.0)
        cached = sparse.size_mb <= budget / 4.0
        lines.append(
            f"  -> block-sparse chi: {sparse.stats()['kept_fraction']:.1%} kept, "
            f"{sparse.size_mb:,.1f} MB/rank; "
            + ("cached" if cached else f"exceeds a quarter of the {budget:.0f} MB budget, "
               "rebuilt on every pass (one extra chi evaluation per SCF iteration)")
        )
    else:
        chunk = choose_strategy(grid, nb, cfg.grid.chunk_size, budget)
        if chunk is None:
            lines.append(
                f"  -> chi ({grid.chi_memory_mb(nb):.1f} MB/rank) fits the "
                f"{budget:.0f} MB budget and will be cached"
            )
        else:
            lines.append(
                f"  -> chi exceeds the {budget:.0f} MB budget; the grid will be "
                f"walked in chunks of {chunk:,} points "
                f"({grid.chi_memory_mb(nb, chunk):.1f} MB/rank), at the cost of "
                f"one extra chi evaluation per SCF iteration"
            )
    lines.append("     (forces walk the grid in their own chunks within the same budget)")
    lines.append("")
    lines.extend(summary_lines(grid_summary(cfg, grid, molecule, _Stats(sparse))))
    from ..parallel import SIZE

    if SIZE > 1:
        lines.append(
            f"  -> running on {SIZE} ranks: the grid is split, so each rank "
            f"holds about 1/{SIZE} of the points above"
        )
    return "\n".join(lines)


class _Stats:
    """Adapter so grid_summary can report a layout-only SparseBlocks."""

    def __init__(self, sparse):
        self.sparsity = None if sparse is None else sparse.stats()


def _n_pairs(cfg_molecule, basis, cfg) -> int:
    from ..integrals.overlap import surviving_pairs

    return len(surviving_pairs(cfg_molecule, basis, cfg.grid.screening_tol))


def report_timing(cfg) -> None:
    from ..parallel import PARALLEL

    if not cfg.output.timing:
        return
    if PARALLEL:
        REGISTRY.report_mpi(printer=mpi_print)
    else:
        REGISTRY.report(printer=mpi_print, min_seconds=0.001)


def print_forces(molecule, forces) -> None:
    mpi_print("")
    mpi_print("  NUCLEAR FORCES  [Hartree/Bohr]")
    mpi_print("  " + "-" * 62)
    mpi_print(f"  {'atom':>8}  {'Fx':>13}  {'Fy':>13}  {'Fz':>13}  {'|F|':>11}")
    for i, atom in enumerate(molecule):
        fx, fy, fz = forces[i]
        mpi_print(
            f"  {atom.symbol + str(i + 1):>8}  {fx:>13.8f}  {fy:>13.8f}  "
            f"{fz:>13.8f}  {np.linalg.norm(forces[i]):>11.8f}"
        )
    mpi_print(
        f"  max |F| = {np.abs(forces).max():.8f} Ha/Bohr   "
        f"|sum F| = {np.linalg.norm(forces.sum(axis=0)):.2e}"
    )
    from ..forces.analytic import compute_forces

    raw = getattr(compute_forces, "last_translation_residual", None)
    if raw is not None:
        mpi_print(f"  raw |sum F| before projection = {raw:.2e} Ha/Bohr (grid-derivative error)")
