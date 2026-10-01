"""Chunked grid evaluation must give exactly the same numbers as the cached path.

The chunked path exists so that a large cluster fits in memory; it is only
useful if it is numerically indistinguishable from the fast path.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config
from pyxany.basis import build_basis
from pyxany.forces import compute_forces
from pyxany.grid import build_molecular_grid
from pyxany.scf import run_scf
from pyxany.scf.grid_terms import GridTerms, choose_strategy


def test_choose_strategy_respects_the_budget(h2o):
    cfg = make_config(basis={"set": "dz"})
    basis = build_basis(h2o, cfg)
    grid = build_molecular_grid(h2o, 25, 10)
    assert choose_strategy(grid, len(basis), "auto", 4096.0) is None
    tiny = choose_strategy(grid, len(basis), "auto", 1.0)
    assert tiny is not None and 0 < tiny < grid.n_local
    assert choose_strategy(grid, len(basis), 500, 4096.0) == 500
    assert choose_strategy(grid, len(basis), 10**9, 4096.0) is None


def test_density_and_potential_matrices_are_chunk_independent(h2o):
    cfg = make_config(basis={"set": "dz"})
    basis = build_basis(h2o, cfg)
    grid = build_molecular_grid(h2o, 25, 10)
    rng = np.random.default_rng(0)
    P = rng.normal(size=(len(basis), len(basis)))
    P = P + P.T
    v = rng.normal(size=grid.n_local)

    cached = GridTerms(basis, h2o, grid, chunk_size=None)
    assert cached.cached
    for chunk in (997, 5000):
        chunked = GridTerms(basis, h2o, grid, chunk_size=chunk)
        assert not chunked.cached
        assert np.abs(cached.density(P) - chunked.density(P)).max() < 1e-12
        assert np.abs(cached.potential_matrix(v)
                      - chunked.potential_matrix(v)).max() < 1e-11


def test_potential_matrices_batch_matches_one_at_a_time(h2o):
    cfg = make_config(basis={"set": "sz"})
    basis = build_basis(h2o, cfg)
    grid = build_molecular_grid(h2o, 20, 8)
    terms = GridTerms(basis, h2o, grid, chunk_size=333)
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=grid.n_local), rng.normal(size=grid.n_local)
    batch = terms.potential_matrices({"a": a, "b": b})
    assert np.abs(batch["a"] - terms.potential_matrix(a)).max() < 1e-12
    assert np.abs(batch["b"] - terms.potential_matrix(b)).max() < 1e-12


@pytest.mark.parametrize("molecule_name", ["h2o"])
def test_scf_energy_is_chunk_independent(request, molecule_name):
    """A whole SCF run, cached versus chunked."""
    molecule = request.getfixturevalue(molecule_name)
    results = {}
    for budget, label in ((4096.0, "cached"), (0.05, "chunked")):
        cfg = make_config(basis={"set": "dz"},
                          grid={"n_r": 25, "n_a": 10, "poisson_l_max": 4,
                                "memory_budget_mb": budget},
                          scf={"max_iter": 60, "e_tol": 1e-9, "d_tol": 1e-8},
                          forces={"enabled": True})
        basis = build_basis(molecule.copy(), cfg)
        mol = molecule.copy()
        res = run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)
        results[label] = (res.total_energy, compute_forces(res, cfg),
                          res.energies["n_electrons_integrated"], res.terms.cached)

    assert results["cached"][3] is True
    assert results["chunked"][3] is False
    assert abs(results["cached"][0] - results["chunked"][0]) < 1e-9
    assert abs(results["cached"][2] - results["chunked"][2]) < 1e-9
    assert np.abs(results["cached"][1] - results["chunked"][1]).max() < 1e-9
