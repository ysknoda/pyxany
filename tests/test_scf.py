"""End-to-end SCF: conservation laws first, then reference energies."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from dvxa.basis import build_basis
from dvxa.scf import run_scf


def run(molecule, **cfg_overrides):
    cfg = make_config(**cfg_overrides)
    basis = build_basis(molecule, cfg)
    return run_scf(molecule, basis, cfg, verbose=0), cfg


#: Relative tolerance on int rho dV at the small grid the suite uses (25 x 10).
#: Tr(PS) is algebraically exact and is held to machine precision separately.
GRID_RTOL = 5e-4


def test_h2_conserves_electrons_and_converges(h2):
    result, _ = run(h2, basis={"set": "dz"})
    assert result.converged
    assert result.energies["trace_PS"] == pytest.approx(2.0, abs=1e-8)
    assert result.energies["n_electrons_integrated"] == pytest.approx(2.0, rel=GRID_RTOL)


def test_water_conserves_electrons(h2o):
    result, _ = run(h2o, basis={"set": "dz"})
    assert result.converged
    assert result.energies["trace_PS"] == pytest.approx(10.0, abs=1e-8)
    assert result.energies["n_electrons_integrated"] == pytest.approx(10.0, rel=GRID_RTOL)


def test_open_shell_methyl_has_nine_electrons(ch3):
    """The whole point of the UKS fix: an odd-electron molecule stays odd."""
    result, _ = run(ch3, basis={"set": "sz"}, scf={"max_iter": 80})
    assert result.unrestricted
    assert (result.n_alpha, result.n_beta) == (5.0, 4.0)
    assert result.occ_a.sum() == pytest.approx(5.0, abs=1e-8)
    assert result.occ_b.sum() == pytest.approx(4.0, abs=1e-8)
    assert result.energies["trace_PS"] == pytest.approx(9.0, abs=1e-7)
    assert result.energies["n_electrons_integrated"] == pytest.approx(9.0, rel=GRID_RTOL)


def test_grid_error_shrinks_when_the_grid_is_refined(h2o):
    """int rho dV converges to the exact electron count as the grid is refined."""
    errors = []
    for n_r, n_a in ((20, 8), (30, 12), (45, 16)):
        result, _ = run(h2o, basis={"set": "sz"},
                        grid={"n_r": n_r, "n_a": n_a})
        errors.append(abs(result.energies["n_electrons_integrated"] - 10.0))
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-4


def test_energies_sit_just_above_hartree_fock():
    """X-alpha without correlation lands within a few percent of HF.

    Loose bounds on purpose: this is a sanity band that catches a broken
    functional or grid, not a claim about the accuracy of X-alpha.
    """
    cases = [
        ("he", [("He", 0, 0, 0)], -2.862),
        ("ne", [("Ne", 0, 0, 0)], -128.547),
        ("h2", [("H", 0, 0, 0), ("H", 0, 0, 0.741)], -1.134),
        ("co", [("C", 0, 0, 0), ("O", 0, 0, 1.128)], -112.79),
    ]
    for name, geom, hf in cases:
        result, _ = run(make_molecule(name, geom), basis={"set": "dz"})
        ratio = result.total_energy / hf
        assert 0.90 < ratio < 1.01, f"{name}: E = {result.total_energy}, HF = {hf}"


def test_total_energy_terms_add_up(h2o):
    result, _ = run(h2o, basis={"set": "sz"})
    e = result.energies
    total = (e["E_kinetic"] + e["E_nuclear_attraction"] + e["E_hartree"]
             + e["E_xc"] + e["E_nn"])
    assert total == pytest.approx(e["E_total"], abs=1e-12)


def test_nuclear_repulsion_matches_the_direct_sum(h2o):
    z = h2o.charges
    pos = h2o.positions
    expected = sum(z[a] * z[b] / np.linalg.norm(pos[a] - pos[b])
                   for a in range(len(z)) for b in range(a + 1, len(z)))
    assert h2o.nuclear_repulsion() == pytest.approx(expected)


def test_overlap_matrix_has_unit_diagonal(h2o):
    result, _ = run(h2o, basis={"set": "dz"})
    assert np.abs(np.diag(result.S) - 1.0).max() < 1e-9
    assert np.abs(result.S - result.S.T).max() < 1e-12
    assert np.abs(result.T - result.T.T).max() < 1e-12
