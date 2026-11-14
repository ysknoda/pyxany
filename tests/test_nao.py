"""Numerical atomic orbitals in a real SCF (ADR 0012, F10).

Until F10 no test ran an SCF with the NAO basis, and a constant spline tail
(splev ext=3) made every NAO result wrong while the SCF reported convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.scf import run_scf

SIH4 = [("Si", 0, 0, 0), ("H", 0.8544, 0.8544, 0.8544), ("H", -0.8544, -0.8544, 0.8544),
        ("H", -0.8544, 0.8544, -0.8544), ("H", 0.8544, -0.8544, -0.8544)]


def _run(geom, basis):
    mol = make_molecule("m", geom)
    cfg = make_config(basis={"set": basis}, grid={"n_r": 30, "n_a": 12},
                      scf={"max_iter": 100, "e_tol": 1e-6, "d_tol": 1e-5})
    return run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)


@pytest.mark.parametrize("element", ["H", "C", "O", "Si", "Ga", "Se"])
def test_every_nao_is_normalized(element):
    """R(r) is normalized on its own grid; S_mumu must be 1 (was 4.09 for Si 3p)."""
    mol = make_molecule("a", [(element, 0, 0, 0)])
    cfg = make_config(basis={"set": "nao"})
    basis = build_basis(mol, cfg)
    from pyxany.integrals.overlap import build_overlap_kinetic

    S, T = build_overlap_kinetic(mol, basis)
    assert np.abs(np.diag(S) - 1.0).max() < 1e-6
    assert np.all(np.isfinite(T)) and np.diag(T).max() < 1e4   # Ga 1s: T ~ Z^2/2


def test_nao_beyond_its_grid_is_zero():
    mol = make_molecule("a", [("Si", 0, 0, 0)])
    basis = build_basis(mol, make_config(basis={"set": "nao"}))
    for f in basis:
        r_end = f.r_grid[-1]
        assert f.radial(np.array([r_end * 1.5, 1e3, 1e5])).tolist() == [0.0, 0.0, 0.0]


def test_nao_sih4_is_close_to_dz_and_integrates_consistently():
    """Was -38.40 Ha (DZ -289.47) with the constant tail."""
    nao, dz = _run(SIH4, "nao"), _run(SIH4, "dz")
    assert nao.converged and dz.converged
    assert abs(nao.total_energy - dz.total_energy) < 2.0
    e = nao.energies
    assert abs(e["n_electrons_integrated"] - e["trace_PS"]) < 1e-3


def test_pair_screening_keeps_pairs_that_overlap():
    """The old exp(-min zeta R) bound dropped NAO Si-Si at 5.45 A, where |S|
    reaches 6.9e-2 (ADR 0012, F12); far-apart pairs must still be dropped."""
    from pyxany.integrals.overlap import build_overlap_kinetic, surviving_pairs

    mol = make_molecule("si2", [("Si", 0, 0, 0), ("Si", 0, 0, 5.45)])
    basis = build_basis(mol, make_config(basis={"set": "nao"}))
    assert surviving_pairs(mol, basis) == [(0, 1)]
    S, _ = build_overlap_kinetic(mol, basis)
    S0, _ = build_overlap_kinetic(mol, basis, screening_tol=0.0)
    assert np.abs(S - S0).max() == 0.0

    far = make_molecule("si2", [("Si", 0, 0, 0), ("Si", 0, 0, 60.0)])
    assert surviving_pairs(far, build_basis(far, make_config(basis={"set": "dz"}))) == []
