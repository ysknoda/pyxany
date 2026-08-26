"""Analytic forces against finite differences and against exact sum rules."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from dvxa.basis import build_basis
from dvxa.forces import (
    compute_forces,
    compute_forces_fd,
    nuclear_repulsion_gradient,
    translation_residual,
)
from dvxa.scf import run_scf

FORCE_CFG = dict(basis={"set": "dz"}, grid={"n_r": 40, "n_a": 14, "poisson_l_max": 4},
                 scf={"max_iter": 80, "e_tol": 1e-9, "d_tol": 1e-8},
                 forces={"enabled": True})


def h2_at(distance_ang):
    return make_molecule("h2", [("H", 0, 0, 0), ("H", 0, 0, distance_ang)])


def converged(molecule, **extra):
    cfg = make_config(**{**FORCE_CFG, **extra})
    return run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0), cfg


def test_translational_invariance_is_exact(h2o):
    """sum_A F_A = 0 must hold to machine precision after projection."""
    result, cfg = converged(h2o)
    assert translation_residual(compute_forces(result, cfg)) < 1e-12


def test_nuclear_repulsion_gradient_matches_finite_differences(h2o):
    step = 1e-5
    analytic = nuclear_repulsion_gradient(h2o)
    base = h2o.positions
    for i in range(len(h2o)):
        for ax in range(3):
            up, dn = base.copy(), base.copy()
            up[i, ax] += step
            dn[i, ax] -= step
            trial = h2o.copy()
            trial.positions = up
            e_up = trial.nuclear_repulsion()
            trial.positions = dn
            e_dn = trial.nuclear_repulsion()
            assert analytic[i, ax] == pytest.approx((e_up - e_dn) / (2 * step), abs=1e-7)


@pytest.mark.parametrize("distance", [0.65, 0.90])
def test_analytic_forces_match_finite_differences_h2(distance):
    """The decisive check on the gradient derivation."""
    molecule = h2_at(distance)
    result, cfg = converged(molecule)
    analytic = compute_forces(result, cfg)
    fd = compute_forces_fd(molecule, cfg, step=0.02, verbose=0)
    assert np.abs(analytic - fd).max() < 5e-4


def test_force_direction_follows_the_energy(h2o):
    """Compressed bond pushes out, stretched bond pulls in; both signs checked."""
    short, _ = converged(h2_at(0.60))
    long_, cfg = converged(h2_at(1.00))
    f_short = compute_forces(short, cfg)
    f_long = compute_forces(long_, cfg)
    assert f_short[1, 2] > 0.0, "a too-short bond must push the atoms apart"
    assert f_long[1, 2] < 0.0, "a too-long bond must pull the atoms together"


def test_force_vanishes_near_the_minimum():
    """Between the two signs there is a geometry with essentially zero force."""
    energies = {}
    forces = {}
    for d in (0.74, 0.78, 0.82):
        result, cfg = converged(h2_at(d))
        energies[d] = result.total_energy
        forces[d] = compute_forces(result, cfg)[1, 2]
    best = min(energies, key=energies.get)
    assert abs(forces[best]) < max(abs(f) for f in forces.values())


@pytest.mark.slow
def test_force_error_shrinks_with_the_grid():
    """The analytic-vs-FD residual is a quadrature effect, so refining helps."""
    molecule = h2_at(0.90)
    errors = []
    for n_r, n_a in ((30, 12), (50, 16)):
        result, cfg = converged(molecule, grid={"n_r": n_r, "n_a": n_a,
                                                "poisson_l_max": 6})
        analytic = compute_forces(result, cfg)
        fd = compute_forces_fd(molecule, cfg, step=0.02, verbose=0)
        errors.append(float(np.abs(analytic - fd).max()))
    assert errors[-1] < errors[0]
