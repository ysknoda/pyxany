"""Analytic forces against finite differences and against exact sum rules."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.forces import (
    compute_forces,
    compute_forces_fd,
    nuclear_repulsion_gradient,
    translation_residual,
)
from pyxany.scf import run_scf

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
def test_analytic_forces_match_finite_differences_on_coarse_and_fine_grids():
    """With the grid-motion and S/T terms, analytic forces are the derivative of
    the computed energy, not of the continuum functional, so they agree with
    finite differences even on a coarse grid (measured 2.5e-5 / 5.9e-5 Ha/Bohr,
    against ~1e-3 before those terms; the FD step itself limits the comparison).
    """
    molecule = h2_at(0.90)
    for n_r, n_a in ((30, 12), (50, 16)):
        result, cfg = converged(molecule, grid={"n_r": n_r, "n_a": n_a,
                                                "poisson_l_max": 6})
        analytic = compute_forces(result, cfg)
        fd = compute_forces_fd(molecule, cfg, step=0.02, verbose=0)
        assert np.abs(analytic - fd).max() < 1e-4

# A C1 water: unequal O-H lengths, bent out of every mirror plane.  H2 is a poor
# gradient test on its own -- one axis, two equivalent atoms, and the
# translation projection makes F_2 = -F_1 whatever the derivation says, so a
# wrong per-atom or per-component term cancels instead of showing up.  Here all
# nine components are independent and non-zero.
ASYMMETRIC_WATER = [("O", 0.000, 0.000, 0.000),
                    ("H", 0.950, 0.000, 0.000),
                    ("H", -0.250, 0.880, 0.140)]


def test_analytic_forces_match_finite_differences_on_an_asymmetric_molecule():
    """Every Cartesian component on every inequivalent atom, against FD.

    Measured residual at this grid is 4.5e-3 Ha/Bohr on a 1.3e-1 Ha/Bohr force
    (3.5%), and it falls to 1.0e-3 (1.7%) at 50x18 / l_max 6 -- see
    ``test_asymmetric_force_error_shrinks_with_the_grid`` and docs/forces.md.
    The residual is the derivative of the quadrature error (the Becke weights
    and the grid itself move with the nuclei), not an error in the derivation,
    so the tolerance here is set from that measurement rather than from what the
    derivation "should" give.  A dropped Pulay or nuclear-attraction term would
    miss by ~1e-1, twenty times this bound.
    """
    molecule = make_molecule("h2o_c1", ASYMMETRIC_WATER)
    cfg = make_config(basis={"set": "sz"},
                      grid={"n_r": 30, "n_a": 12, "poisson_l_max": 4},
                      scf={"max_iter": 80, "e_tol": 1e-9, "d_tol": 1e-8},
                      forces={"enabled": True})
    result = run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)
    assert result.converged

    analytic = compute_forces(result, cfg)
    fd = compute_forces_fd(molecule, cfg, step=0.02, verbose=0)

    assert np.abs(analytic - fd).max() < 8e-3

    # A max-norm alone lets one bad component hide behind a large good one, so
    # check each atom and axis, and check the signs of the components that carry
    # real magnitude.
    for i in range(len(molecule)):
        for ax in range(3):
            assert analytic[i, ax] == pytest.approx(fd[i, ax], abs=8e-3), (
                f"atom {i} axis {'xyz'[ax]}: {analytic[i, ax]} vs FD {fd[i, ax]}"
            )
    significant = np.abs(analytic) > 1e-3
    assert np.array_equal(np.sign(analytic[significant]), np.sign(fd[significant])), (
        "analytic and finite-difference forces disagree in sign"
    )


@pytest.mark.slow
def test_asymmetric_force_error_shrinks_with_the_grid():
    """Refining the quadrature must close the analytic-FD gap on a C1 geometry.

    This is what distinguishes "the residual is quadrature error" from "the
    gradient is wrong": a wrong derivation does not converge to the FD result.
    """
    errors = []
    for n_r, n_a, l_max in ((40, 14, 4), (50, 18, 6)):
        molecule = make_molecule("h2o_c1", ASYMMETRIC_WATER)
        cfg = make_config(basis={"set": "dz"},
                          grid={"n_r": n_r, "n_a": n_a, "poisson_l_max": l_max},
                          scf={"max_iter": 80, "e_tol": 1e-9, "d_tol": 1e-8},
                          forces={"enabled": True})
        result = run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)
        analytic = compute_forces(result, cfg)
        fd = compute_forces_fd(molecule, cfg, step=0.02, verbose=0)
        errors.append(float(np.abs(analytic - fd).max()))
    assert errors[-1] < 0.5 * errors[0], f"residual did not fall with the grid: {errors}"


def _pseudo_terminated_molecule():
    """Be with two pseudo-hydrogens of different fractional charge."""
    from pyxany.constants import ANG_TO_BOHR
    from pyxany.molecule import Atom, Molecule
    return Molecule([
        Atom.from_element("Be", np.array([0.0, 0.0, 0.0])),
        Atom.pseudo_hydrogen("HBe", 0.50, np.array([1.30, 0.0, 0.0]) * ANG_TO_BOHR),
        Atom.pseudo_hydrogen("HBe2", 1.25, np.array([-0.35, 1.22, 0.0]) * ANG_TO_BOHR),
    ], name="be_ph")


def _pseudo_config():
    # The shared 25x10 test grid is too coarse for a fractional-charge center
    # (27.6% analytic-vs-FD residual); 30x12 brings it to 6.9%.
    return make_config(basis={"set": "sz"},
                       grid={"n_r": 30, "n_a": 12, "poisson_l_max": 4},
                       scf={"max_iter": 60}, forces={"enabled": True})


def test_forces_work_on_a_pseudo_hydrogen_terminated_molecule():
    """The passivated-cluster force path must not need a pre-derived exponent.

    ``optimize_geometry`` and ``compute_forces_fd`` used to call ``build_basis``
    without the pseudo-hydrogen exponents, which only the CLI derived, so
    ``pyxany opt --freeze-core`` -- the documented way to relax a passivated
    cluster -- raised KeyError on every passivated example.  ``build_basis``
    now derives them itself when they are needed and absent.
    """
    molecule = _pseudo_terminated_molecule()
    cfg = _pseudo_config()
    result = run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)
    analytic = compute_forces(result, cfg)
    fd = compute_forces_fd(molecule, cfg, step=0.02, verbose=0)
    assert np.all(np.isfinite(analytic)) and np.all(np.isfinite(fd))
    significant = np.abs(analytic) > 2e-3
    assert np.array_equal(np.sign(analytic[significant]), np.sign(fd[significant]))
    # Measured 5.4e-3 on a 7.9e-2 force at this grid.  This test exists to show
    # the fractional-Z path runs and is structurally right, not to pin accuracy.
    assert np.abs(analytic - fd).max() < 1.2e-2


def test_optimization_runs_with_pseudo_hydrogens_and_a_freeze_mask(tmp_path, monkeypatch):
    """The `--freeze-core` workflow end to end: relax only the terminators."""
    monkeypatch.chdir(tmp_path)
    from pyxany.forces.optimize import optimize_geometry
    molecule = _pseudo_terminated_molecule()
    cfg = make_config(basis={"set": "sz"},
                      grid={"n_r": 25, "n_a": 10, "poisson_l_max": 4},
                      scf={"max_iter": 60}, forces={"enabled": True},
                      optimize={"enabled": True, "max_steps": 2, "freeze": "core"},
                      output={"dir": str(tmp_path / "out"), "prefix": "t",
                              "figure": False, "json": False, "timing": False})
    result = optimize_geometry(molecule, cfg, verbose=0)
    assert np.all(np.isfinite(result.molecule.positions))
    # The trajectory belongs in output.dir, not in the working directory.
    assert (tmp_path / "out" / "t_traj.xyz").is_file()
    assert not list(tmp_path.glob("*_traj.xyz"))
    # The frozen core must not have moved.
    assert np.allclose(result.molecule.positions[0], molecule.positions[0])


def test_translation_projection_is_off_when_atoms_are_frozen():
    """Projecting the all-atom mean is invalid once part of the molecule is fixed.

    The energy is not invariant under translating a subset, so subtracting the
    all-atom mean injects a uniform spurious force into the moving atoms. With
    the grid-motion terms the unprojected forces are the derivative of the
    computed energy: measured 4.5e-5 Ha/Bohr from finite differences on the
    hydrogens, against 3.0e-4 after projection. l_max = 9 keeps the multipole
    truncation (6.7e-3 residual at l_max = 4) from swamping the comparison.
    """
    molecule = make_molecule("h2o_c1", ASYMMETRIC_WATER)
    cfg = make_config(basis={"set": "sz"},
                      grid={"n_r": 30, "n_a": 12, "poisson_l_max": 9},
                      scf={"max_iter": 80, "e_tol": 1e-9, "d_tol": 1e-8},
                      forces={"enabled": True})
    result = run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)
    projected = compute_forces(result, cfg, project=True)
    raw = compute_forces(result, cfg, project=False)
    fd = compute_forces_fd(molecule, cfg, step=0.02, verbose=0)

    shift = raw.mean(axis=0)
    assert np.linalg.norm(shift) > 1e-4, "no shift to speak of; test is not exercising anything"
    assert np.allclose(projected, raw - shift)

    # With O frozen, the unprojected gradient is the one that matches dE/dR.
    moving = [1, 2]
    assert (np.abs(raw[moving] - fd[moving]).max()
            < np.abs(projected[moving] - fd[moving]).max())


def test_raw_translation_residual_is_recorded():
    """The reported |sum F| is ~0 by construction; the raw residual is the real
    grid-quality signal and must be kept."""
    molecule = make_molecule("h2o_c1", ASYMMETRIC_WATER)
    cfg = make_config(basis={"set": "sz"},
                      grid={"n_r": 30, "n_a": 12, "poisson_l_max": 4},
                      scf={"max_iter": 80}, forces={"enabled": True})
    result = run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)
    forces = compute_forces(result, cfg, project=True)
    assert translation_residual(forces) < 1e-12          # by construction
    raw = compute_forces.last_translation_residual        # the informative one
    assert raw > 1e-6, "the raw residual should be a real, non-trivial number"


def _fake_optimizer_backend(monkeypatch, energy_min_bohr, force_zero_bohr, k=0.5):
    """Replace SCF and forces with a harmonic H2 whose E-minimum and F-zero differ.

    On a finite DV grid the energy and the analytic forces are not exactly
    consistent, so the bond length where E is lowest and the one where F = 0
    differ slightly. This stand-in reproduces that in microseconds.
    """
    from types import SimpleNamespace

    import pyxany.forces.optimize as opt

    def fake_scf(mol, basis, config, verbose=0, **kwargs):
        r = float(np.linalg.norm(mol.positions[1] - mol.positions[0]))
        return SimpleNamespace(total_energy=k * (r - energy_min_bohr) ** 2, molecule=mol,
                               converged=True,
                               P=None, P_a=None, P_b=None)

    def fake_forces(result, config, project=True):
        pos = result.molecule.positions
        d = pos[1] - pos[0]
        r = float(np.linalg.norm(d))
        f1 = -2.0 * k * (r - force_zero_bohr) * d / r
        return np.array([-f1, f1])

    monkeypatch.setattr(opt, "build_basis", lambda mol, cfg: None)
    monkeypatch.setattr(opt, "run_scf", fake_scf)
    monkeypatch.setattr(opt, "compute_forces", fake_forces)


def _h2(r_bohr):
    from pyxany.molecule import Atom, Molecule

    return Molecule([Atom.from_element("H", np.zeros(3)),
                     Atom.from_element("H", np.array([0.0, 0.0, r_bohr]))], name="h2")


def test_optimizer_returns_the_geometry_its_energy_and_forces_belong_to(monkeypatch, tmp_path):
    """The result must not be the last line-search trial point but a coherent one."""
    from pyxany.forces.optimize import optimize_geometry

    _fake_optimizer_backend(monkeypatch, 1.40, 1.40)
    cfg = make_config(optimize={"enabled": True, "f_max": 1e-4, "max_steps": 50},
                      output={"dir": str(tmp_path), "prefix": "t"})
    res = optimize_geometry(_h2(1.70), cfg, verbose=0)
    r = float(np.linalg.norm(res.molecule.positions[1] - res.molecule.positions[0]))
    assert res.converged
    assert res.energy == pytest.approx(0.5 * (r - 1.40) ** 2, abs=1e-14)
    assert np.abs(res.forces).max() < 1e-4


def test_optimizer_accepts_a_force_converged_point_despite_grid_inconsistency(
        monkeypatch, tmp_path):
    """Measured on H2O/dz at n_r=40: max|F| hit 5.8e-6 on one evaluation, but L-BFGS
    settled on the energy minimum where max|F| = 3.6e-4 and ran out of steps."""
    from pyxany.forces.optimize import optimize_geometry

    # F = 0 sits 1e-3 Bohr from the energy minimum: |F| = 1e-3 Ha/Bohr at E_min
    _fake_optimizer_backend(monkeypatch, 1.400, 1.401)
    cfg = make_config(optimize={"enabled": True, "f_max": 3e-4, "max_steps": 50},
                      output={"dir": str(tmp_path), "prefix": "t"})
    res = optimize_geometry(_h2(1.70), cfg, verbose=0)
    assert res.converged
    assert np.abs(res.forces).max() < 3e-4


def test_chunked_force_evaluation_matches_a_single_chunk():
    """Forces are built chunk by chunk to bound memory; the split must not matter."""
    import pyxany.forces.analytic as analytic

    molecule = make_molecule("h2o_c1", ASYMMETRIC_WATER)
    cfg = make_config(basis={"set": "dz"}, scf={"max_iter": 80, "e_tol": 1e-9, "d_tol": 1e-8},
                      forces={"enabled": True})
    result = run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)
    whole = compute_forces(result, cfg, project=False)
    n_points = len(result.grid.local_points)
    saved = analytic._force_chunk
    try:
        analytic._force_chunk = lambda *a, **k: 997           # many uneven chunks
        split = compute_forces(result, cfg, project=False)
    finally:
        analytic._force_chunk = saved
    assert n_points > 5 * 997
    assert np.abs(split - whole).max() < 1e-12


def test_uks_forces_equal_rks_forces_for_a_closed_shell():
    """Regression: the UKS grid terms used twice the exchange-energy density
    (0.46 Ha/Bohr off on H2O) from 4f4746c until this test."""
    from conftest import make_config, make_molecule
    from pyxany.basis import build_basis
    from pyxany.forces.analytic import compute_forces
    from pyxany.scf import run_scf

    w = [("O", 0.05, 0.02, 0.1173), ("H", 0.03, 0.7572, -0.4692), ("H", -0.02, -0.7572, -0.4492)]
    mol = make_molecule("w", w)
    forces = []
    for unrestricted in (False, True):
        cfg = make_config(basis={"set": "dz"}, grid={"n_r": 30, "n_a": 12},
                          scf={"e_tol": 1e-10, "d_tol": 1e-8, "unrestricted": unrestricted})
        res = run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)
        forces.append(compute_forces(res, cfg, project=False))
    assert np.abs(forces[0] - forces[1]).max() < 1e-7


@pytest.mark.slow
def test_open_shell_forces_match_finite_differences():
    """CH3 (doublet, UKS) on an asymmetric geometry, full multipole order."""
    from conftest import make_config, make_molecule
    from pyxany.basis import build_basis
    from pyxany.forces.analytic import compute_forces
    from pyxany.scf import run_scf

    geom = [("C", 0.02, -0.01, 0.03), ("H", 1.08, 0.02, -0.02),
            ("H", -0.55, 0.95, 0.04), ("H", -0.52, -0.91, -0.05)]
    cfg = make_config(basis={"set": "dz"}, grid={"n_r": 40, "n_a": 14, "poisson_l_max": "auto"},
                      scf={"e_tol": 1e-11, "d_tol": 1e-9, "max_iter": 150})

    def run(g):
        mol = make_molecule("ch3", g, multiplicity=2)
        return run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)

    F = compute_forces(run(geom), cfg, project=False)
    h = 1e-3 * 0.529177210903
    fd = np.zeros_like(F)
    for a in range(len(geom)):
        for x in range(3):
            p = [list(t) for t in geom]
            m = [list(t) for t in geom]
            p[a][1 + x] += h
            m[a][1 + x] -= h
            fd[a, x] = -(run(p).total_energy - run(m).total_energy) / 2e-3
    assert np.abs(F - fd).max() < 5e-4
