"""Element-specific X-alpha parameters (ADR 0011)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.config import ConfigError
from pyxany.scf import run_scf
from pyxany.scf.alpha_field import alpha_on_points, alpha_point_gradient, atom_alphas

WATER = [("O", 0.05, 0.02, 0.1173), ("H", 0.03, 0.7572, -0.4692), ("H", -0.02, -0.7572, -0.4492)]


def test_settings_resolve_per_element():
    scf = make_config(scf={"alpha": {"default": 0.71, "O": 0.74}}).scf
    assert (scf.alpha_for("O"), scf.alpha_for("Si")) == (0.74, 0.71)
    assert scf.uniform_alpha is None
    assert make_config(scf={"alpha": 0.7}).scf.uniform_alpha == 0.7
    assert make_config(scf={"alpha": {"default": 0.72}}).scf.uniform_alpha == 0.72


@pytest.mark.parametrize("bad", ["schwarz", {"Xx": 0.7}, {"O": 3.0}, -0.1, True])
def test_bad_alpha_settings_are_rejected(bad):
    with pytest.raises(ConfigError, match="alpha"):
        make_config(scf={"alpha": bad})


def test_uniform_alpha_stays_a_scalar():
    mol = make_molecule("w", WATER)
    assert alpha_on_points(np.zeros((5, 3)), mol, make_config().scf) == 0.7
    same = make_config(scf={"alpha": {"O": 0.7, "H": 0.7}}).scf
    assert alpha_on_points(np.zeros((5, 3)), mol, same) == 0.7


def test_mixed_alpha_takes_each_atoms_value_inside_its_cell():
    mol = make_molecule("w", WATER)
    scf = make_config(scf={"alpha": {"O": 0.74, "H": 0.97}}).scf
    near = mol.positions + 1e-3
    a = alpha_on_points(near, mol, scf)
    assert a == pytest.approx([0.74, 0.97, 0.97], abs=1e-6)
    mid = 0.5 * (mol.positions[0] + mol.positions[1])
    assert 0.74 < alpha_on_points(mid[None], mol, scf)[0] < 0.97


def test_alpha_point_gradient_matches_finite_differences():
    mol = make_molecule("w", WATER)
    scf = make_config(scf={"alpha": {"O": 0.74, "H": 0.97}}).scf
    alphas = atom_alphas(mol, scf)
    pts = np.random.default_rng(0).normal(scale=1.2, size=(200, 3))
    g = alpha_point_gradient(pts, mol, alphas)
    h = 1e-6
    base = mol.positions.copy()
    for B in range(3):
        for x in range(3):
            shift = np.zeros_like(base)
            shift[B, x] = h
            mol.positions = base + shift
            ap = alpha_on_points(pts, mol, scf)
            mol.positions = base - shift
            am = alpha_on_points(pts, mol, scf)
            mol.positions = base
            assert np.abs(g[:, B, x] - (ap - am) / (2 * h)).max() < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("unrestricted", [False, True])
def test_forces_with_element_alpha_match_finite_differences(unrestricted):
    from pyxany.forces.analytic import compute_forces

    cfg = make_config(basis={"set": "dz"}, grid={"n_r": 40, "n_a": 14, "poisson_l_max": "auto"},
                      scf={"e_tol": 1e-11, "d_tol": 1e-9, "max_iter": 100,
                           "alpha": {"O": 0.74, "H": 0.97}, "unrestricted": unrestricted})

    def run(g):
        mol = make_molecule("w", g)
        return run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)

    F = compute_forces(run(WATER), cfg, project=False)
    h = 1e-3 * 0.529177210903
    fd = np.zeros_like(F)
    for a in range(3):
        for x in range(3):
            p = [list(t) for t in WATER]
            m = [list(t) for t in WATER]
            p[a][1 + x] += h
            m[a][1 + x] -= h
            fd[a, x] = -(run(p).total_energy - run(m).total_energy) / 2e-3
    assert np.abs(F - fd).max() < 2.5e-4          # 1.3e-4 measured; 1.0e-4 at uniform alpha


def test_polarised_atom_reproduces_exchange_only_lda():
    """alpha = 2/3 is LDA exchange; spin-polarised H and closed-shell He.

    References: PySCF 2.14 exchange-only LDA ('lda,'), uncontracted ANO-RCC,
    grid level 9 (UKS for H): -0.457065 and -2.723592 Ha.
    """
    from pyxany.basis.atomic_solver import solve_atom_polarised

    assert solve_atom_polarised("H", alpha=2 / 3).energy == pytest.approx(-0.457065, abs=5e-5)
    assert solve_atom_polarised("He", alpha=2 / 3).energy == pytest.approx(-2.723592, abs=5e-5)
