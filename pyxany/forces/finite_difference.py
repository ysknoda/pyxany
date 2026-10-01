"""Finite-difference forces: the reference implementation.

Central differences of the total energy, which is slow (``6 N`` SCF runs per
geometry) but has no derivation to get wrong.  This is what
:mod:`pyxany.forces.analytic` is validated against.
"""

from __future__ import annotations

import numpy as np

from ..basis import build_basis
from ..parallel import mpi_print
from ..scf import run_scf
from ..timing import timer


def _energy_at(molecule, config, positions) -> float:
    trial = molecule.copy()
    trial.positions = positions
    basis = build_basis(trial, config)
    return run_scf(trial, basis, config, verbose=0).total_energy


def compute_forces_fd(molecule, config, step: float = 0.02,
                      atoms: list[int] | None = None, verbose: int = 1) -> np.ndarray:
    """Central-difference forces (Hartree/Bohr), shape (n_atoms, 3).

    ``step`` is in Bohr.  ``atoms`` restricts the calculation to a subset, which
    is what makes validating one atom of a large cluster affordable.
    """
    with timer("forces.finite_difference"):
        which = list(range(len(molecule))) if atoms is None else list(atoms)
        base = molecule.positions
        forces = np.zeros((len(molecule), 3))
        total = 2 * 3 * len(which)
        done = 0
        for ai in which:
            for ax in range(3):
                energies = []
                for sign in (+1, -1):
                    pos = base.copy()
                    pos[ai, ax] += sign * step
                    energies.append(_energy_at(molecule, config, pos))
                    done += 1
                    if verbose > 1:
                        mpi_print(f"    fd {done}/{total}  atom {ai} axis {ax} "
                                  f"{sign:+d}: E = {energies[-1]:.9f}")
                forces[ai, ax] = -(energies[0] - energies[1]) / (2.0 * step)
        return forces


def compare_with_fd(result, config, step: float = 0.02,
                    atoms: list[int] | None = None) -> dict:
    """Analytic vs finite-difference forces, with the differences summarised.

    Returns a dict carrying both force arrays, the maximum absolute deviation
    and the translational-invariance residual of each -- the two numbers that
    decide whether the analytic gradient can be trusted for optimisation.
    """
    from .analytic import compute_forces, translation_residual

    analytic = compute_forces(result, config)
    fd = compute_forces_fd(result.molecule, config, step=step, atoms=atoms, verbose=0)
    which = list(range(len(result.molecule))) if atoms is None else list(atoms)
    diff = analytic[which] - fd[which]
    return {
        "analytic": analytic,
        "finite_difference": fd,
        "atoms": which,
        "step": step,
        "max_abs_diff": float(np.abs(diff).max()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "max_abs_force": float(np.abs(fd[which]).max()),
        "translation_residual_analytic": translation_residual(analytic),
        "translation_residual_fd": translation_residual(fd),
    }
