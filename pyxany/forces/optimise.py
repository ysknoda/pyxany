"""Geometry optimisation driven by analytic forces.

Uses scipy's L-BFGS-B on the (energy, gradient) pair.  Frozen atoms are removed
from the optimisation vector entirely rather than having their gradient zeroed,
so the optimiser never spends steps on coordinates that cannot move -- this is
what makes ``--freeze-core`` (relax only the pseudo-hydrogen terminators of a
cluster) cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..basis import build_basis
from ..constants import BOHR_TO_ANG
from ..io.xyz import append_frame
from ..parallel import is_root, mpi_print
from ..scf import run_scf
from ..timing import timer
from .analytic import compute_forces, project_out_translation
from .finite_difference import compute_forces_fd

#: config name -> the scipy solver actually used.  `cg` is named for what it is:
#: the option used to be called `steepest_descent` while running conjugate
#: gradient, so the banner reported a method that was never run.
SCIPY_METHOD = {"lbfgs": "L-BFGS-B", "cg": "CG"}


class _Converged(Exception):
    """Raised from the objective to stop scipy once max|F| < f_max."""


@dataclass
class OptimisationResult:
    converged: bool
    molecule: object
    energy: float
    forces: np.ndarray
    n_steps: int
    trajectory: list = field(default_factory=list)
    scf: object = None


def freeze_mask(molecule, freeze) -> np.ndarray:
    """Boolean mask of atoms held fixed.

    ``"none"``  nothing frozen
    ``"core"``  everything except the pseudo-hydrogen terminators
    list        the given atom indices
    """
    n = len(molecule)
    mask = np.zeros(n, dtype=bool)
    if freeze in (None, "none"):
        return mask
    if freeze == "core":
        for i, atom in enumerate(molecule):
            mask[i] = not atom.is_pseudo
        if mask.all():
            raise ValueError(
                "optimise.freeze='core' would freeze every atom: the structure has "
                "no pseudo-hydrogen terminators to relax"
            )
        return mask
    for i in freeze:
        if not 0 <= i < n:
            raise ValueError(f"optimise.freeze: atom index {i} out of range 0..{n - 1}")
        mask[i] = True
    return mask


def optimise_geometry(molecule, config, verbose: int = 1, restart=None,
                      checkpoint_path=None) -> OptimisationResult:
    """Relax the geometry until ``max|F| < config.optimise.f_max``."""
    opt_cfg = config.optimise
    frozen = freeze_mask(molecule, opt_cfg.freeze)
    moving = np.where(~frozen)[0]
    if len(moving) == 0:
        raise ValueError("optimise: every atom is frozen, nothing to do")

    current = molecule.copy()
    state = {"n": 0, "best": None, "traj": [], "scf": None}
    traj_path = None
    if opt_cfg.trajectory and is_root():
        # Same directory as every other output file, not the working directory.
        out_dir = Path(config.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = out_dir / f"{config.output.prefix}_traj.xyz"
        open(traj_path, "w").close()

    if verbose:
        mpi_print("=" * 78)
        mpi_print(f"  Geometry optimisation ({SCIPY_METHOD[opt_cfg.method]})   "
                  f"f_max = {opt_cfg.f_max:g} Ha/Bohr   "
                  f"{len(moving)}/{len(molecule)} atoms free")
        mpi_print("=" * 78)
        mpi_print(f"  {'step':>4}  {'E (Ha)':>17}  {'max|F|':>11}  {'rms F':>11}")
        mpi_print("  " + "-" * 52)

    # Removing the net translation is legitimate only while the whole molecule
    # is free; with anything frozen it injects a uniform spurious force into the
    # moving atoms (1.8e-3 Ha/Bohr for water on the default grid, larger than
    # the default f_max of 1e-3), so scipy would be handed an energy and a
    # gradient that do not correspond.  Applying it to the finite-difference
    # branch as well keeps the two forces.method settings consistent.
    anything_frozen = bool(frozen.any())
    # fixed embedding charges break translation invariance the same way
    embedded = bool(config.embedding.charges)
    project = (not anything_frozen) and (not embedded) and getattr(
        config.forces, "project_translation", True)

    def energy_and_gradient(x: np.ndarray):
        pos = current.positions
        pos[moving] = x.reshape(-1, 3)
        current.positions = pos
        basis = build_basis(current, config)
        # start from the previous evaluation's density (ADR 0004); the first may
        # come from a checkpoint
        result = run_scf(current, basis, config, verbose=0, initial=state.get("guess"),
                         restart=restart if state["n"] == 0 else None,
                         checkpoint_path=checkpoint_path)
        state["guess"] = {"P": result.P, "P_a": result.P_a, "P_b": result.P_b}
        if config.forces.method == "finite_difference":
            forces = compute_forces_fd(current, config, step=config.forces.fd_step,
                                       atoms=list(moving), verbose=0)
            if project:
                forces = project_out_translation(forces)
        else:
            forces = compute_forces(result, config, project=project)
        state["n"] += 1
        f_move = forces[moving]
        f_max = float(np.abs(f_move).max())
        if verbose:
            mpi_print(f"  {state['n']:>4}  {result.total_energy:>17.9f}  {f_max:>11.3e}  "
                      f"{float(np.sqrt(np.mean(f_move**2))):>11.3e}")
        state["traj"].append({"step": state["n"], "E": result.total_energy,
                              "f_max": f_max,
                              "positions": current.positions.copy()})
        if traj_path:
            append_frame(traj_path, current,
                         f"step {state['n']}  E = {result.total_energy:.9f} Ha  "
                         f"max|F| = {f_max:.3e} Ha/Bohr")
        record = (result.total_energy, forces.copy(), current.positions.copy(), result)
        # Keep the evaluation the result will report: the lowest energy (which is
        # where L-BFGS-B ends, res.x) -- not simply the last one, which is often a
        # rejected line-search trial whose geometry, energy and forces do not
        # belong to the minimiser's answer.
        if state["best"] is None or result.total_energy < state["best"][0]:
            state["best"] = record
        if f_max < opt_cfg.f_max:
            # Converged by the stated criterion. On a finite grid E and the
            # analytic forces are not exactly consistent, so the point where F
            # vanishes need not be the energy minimum; L-BFGS-B would reject it
            # and stall at the energy minimum with |F| above f_max.
            state["best"] = record
            raise _Converged
        # scipy minimises E and wants dE/dx = -F
        return result.total_energy, (-f_move).ravel()

    with timer("optimise"):
        x0 = current.positions[moving].ravel()
        # `ftol` is an L-BFGS-B option; CG does not take it and scipy warns and
        # ignores it, so the energy tolerance was silently dropped on that
        # branch.  `maxiter` also counts differently between the two.
        method = SCIPY_METHOD[opt_cfg.method]
        options = {"maxiter": opt_cfg.max_steps, "gtol": opt_cfg.f_max}
        if method == "L-BFGS-B":
            options["ftol"] = 1e-12
        try:
            minimize(energy_and_gradient, x0, jac=True, method=method, options=options)
        except _Converged:
            pass

    energy, forces, positions, state["scf"] = state["best"]
    current.positions = positions
    f_max = float(np.abs(forces[moving]).max())
    converged = f_max < opt_cfg.f_max
    if verbose:
        mpi_print("  " + "-" * 52)
        status = "CONVERGED" if converged else "NOT CONVERGED"
        mpi_print(f"  [{status}] max|F| = {f_max:.3e} Ha/Bohr after {state['n']} "
                  f"energy/gradient evaluations")
        if traj_path:
            mpi_print(f"  trajectory: {traj_path}")
        _report_geometry(current, verbose)

    return OptimisationResult(converged=converged, molecule=current, energy=energy,
                              forces=forces, n_steps=state["n"],
                              trajectory=state["traj"], scf=state["scf"])


def _report_geometry(molecule, verbose: int) -> None:
    """Print the bonded distances (and angles for a triatomic) after relaxing."""
    n = len(molecule)
    if n < 2:
        return
    D = molecule.distance_matrix() * BOHR_TO_ANG
    mpi_print("")
    mpi_print("  Final geometry (Angstrom)")
    for i, atom in enumerate(molecule):
        x, y, z = atom.pos_angstrom
        mpi_print(f"    {atom.symbol:<6s} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] < 3.0]
    if pairs:
        mpi_print("  Bond lengths")
        for i, j in pairs:
            mpi_print(f"    {molecule[i].symbol}{i + 1}-{molecule[j].symbol}{j + 1}: "
                      f"{D[i, j]:.4f} A")
    if n == 3:
        v1 = molecule[0].pos - molecule[1].pos
        v2 = molecule[2].pos - molecule[1].pos
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        mpi_print(f"  Angle {molecule[0].symbol}1-{molecule[1].symbol}2-"
                  f"{molecule[2].symbol}3: {np.degrees(np.arccos(np.clip(cos, -1, 1))):.2f} deg")
