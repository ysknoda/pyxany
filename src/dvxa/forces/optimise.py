"""Geometry optimisation driven by analytic forces.

Uses scipy's L-BFGS-B on the (energy, gradient) pair.  Frozen atoms are removed
from the optimisation vector entirely rather than having their gradient zeroed,
so the optimiser never spends steps on coordinates that cannot move -- this is
what makes ``--freeze-core`` (relax only the pseudo-hydrogen terminators of a
cluster) cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..basis import build_basis
from ..constants import BOHR_TO_ANG
from ..io.xyz import append_frame
from ..parallel import is_root, mpi_print
from ..scf import run_scf
from ..timing import timer
from .analytic import compute_forces
from .finite_difference import compute_forces_fd


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


def optimise_geometry(molecule, config, verbose: int = 1) -> OptimisationResult:
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
        traj_path = f"{config.output.prefix}_traj.xyz"
        open(traj_path, "w").close()

    if verbose:
        mpi_print("=" * 78)
        mpi_print(f"  Geometry optimisation ({opt_cfg.method.upper()})   "
                  f"f_max = {opt_cfg.f_max:g} Ha/Bohr   "
                  f"{len(moving)}/{len(molecule)} atoms free")
        mpi_print("=" * 78)
        mpi_print(f"  {'step':>4}  {'E (Ha)':>17}  {'max|F|':>11}  {'rms F':>11}")
        mpi_print("  " + "-" * 52)

    def energy_and_gradient(x: np.ndarray):
        pos = current.positions
        pos[moving] = x.reshape(-1, 3)
        current.positions = pos
        basis = build_basis(current, config)
        result = run_scf(current, basis, config, verbose=0)
        if config.forces.method == "finite_difference":
            forces = compute_forces_fd(current, config, step=config.forces.fd_step,
                                       atoms=list(moving), verbose=0)
        else:
            forces = compute_forces(result, config)
        state["n"] += 1
        state["scf"] = result
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
        state["best"] = (result.total_energy, forces.copy())
        # scipy minimises E and wants dE/dx = -F
        return result.total_energy, (-f_move).ravel()

    with timer("optimise"):
        x0 = current.positions[moving].ravel()
        method = "L-BFGS-B" if opt_cfg.method == "lbfgs" else "CG"
        res = minimize(energy_and_gradient, x0, jac=True, method=method,
                       options={"maxiter": opt_cfg.max_steps,
                                "gtol": opt_cfg.f_max, "ftol": 1e-12})

    energy, forces = state["best"]
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
