"""Every grid is set from the input file; pruning and block sparsity (ADR 0009)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.config import ConfigError
from pyxany.grid.molecular import grid_from_config
from pyxany.integrals.coulomb import hartree_potential
from pyxany.scf import run_scf
from pyxany.scf.grid_terms import GridTerms

# -- settings ---------------------------------------------------------------


def test_every_grid_has_a_default_and_partial_blocks_are_completed():
    g = make_config(grid={"n_r": 25, "n_a": 10, "pair": {"n_r": 50}}).grid
    assert g.pair == {"n_r": 50, "n_a": 14}
    assert g.atom == {"n_points": 2000}
    assert g.guess == {"n_r": 80, "n_a": 14}
    assert g.one_center == {"n_r": 400}
    assert g.pruning == {"enabled": True, "regions": "auto"}
    assert g.basis_cutoff == 1e-6


@pytest.mark.parametrize(
    "old, new",
    [
        ({"grid": {"pair_n_r": 40}}, "pair"),
        ({"grid": {"pair_n_a": 14}}, "pair"),
        ({"basis": {"nao_n_r": 2000}}, "atom"),
    ],
)
def test_moved_keys_say_where_they_went(old, new):
    with pytest.raises(ConfigError, match=new):
        make_config(**old)


@pytest.mark.parametrize(
    "grid",
    [
        {"pair": {"n_rr": 40}},
        {"atom": {"n_points": 100}},
        {"guess": {"n_a": 2}},
        {"n_r_by_element": {"Xx": 40}},
        {"n_r_by_element": {"Si": 2}},
        {"pruning": {"regions": [[0.5, 4], [0.25, 6]]}},
        {"pruning": {"regions": [[0.25, 40]]}},
        {"pruning": {"enabeld": True}},
        {"basis_cutoff": -1.0},
    ],
)
def test_bad_grid_settings_are_rejected(grid):
    with pytest.raises(ConfigError):
        make_config(grid={"n_r": 25, "n_a": 10, **grid})


def test_nested_overrides_from_the_command_line():
    cfg = make_config()
    cfg.apply_overrides({"grid.pair.n_r": "55", "grid.pruning.enabled": "false"})
    assert cfg.grid.pair == {"n_r": 55, "n_a": 14}
    assert cfg.grid.pruning["enabled"] is False


def test_radial_points_by_element_override_the_table(h2o):
    cfg = make_config(grid={"n_r": 25, "n_a": 10, "n_r_by_element": {"O": 33}})
    grid = grid_from_config(h2o, cfg.grid)
    assert list(grid.n_r_atom) == [33, *grid_from_config(h2o, make_config().grid).n_r_atom[1:]]


# -- angular pruning --------------------------------------------------------


def _pruning(on: bool):
    return make_config(
        grid={
            "n_r": 30,
            "n_a": 12,
            "poisson_l_max": 11,
            "pruning": {"enabled": on},
            "basis_cutoff": 0.0,
        }
    )


def test_pruned_grid_layout(h2o):
    grid = grid_from_config(h2o, _pruning(True).grid)
    full = grid_from_config(h2o, _pruning(False).grid)
    assert grid.pruned and not full.pruned
    assert len(grid) < 0.75 * len(full)
    for ai in range(len(h2o)):
        r = grid.radii[ai]
        R = h2o[ai].grid_radius()
        covered = 0
        for lo, hi, na, rows in grid.shell_blocks(ai):
            assert rows.stop - rows.start == (hi - lo) * 2 * na * na
            expect = 4 if r[lo] < 0.25 * R else (8 if r[lo] < 0.5 * R else 12)
            assert na == expect
            covered += hi - lo
        assert covered == len(r)
    with pytest.raises(ValueError):
        grid.as_radial_angular(np.zeros(len(grid)), 0)


def test_pruning_changes_energy_and_forces_far_less_than_the_grid_error(h2o):
    """The criterion of ADR 0009: < 10 % of the n_a 12 -> 16 difference."""
    from pyxany.forces.analytic import compute_forces

    def run(cfg):
        res = run_scf(h2o, build_basis(h2o, cfg), cfg, verbose=0)
        return res.total_energy, compute_forces(res, cfg, project=False)

    e_p, f_p = run(_pruning(True))
    e_u, f_u = run(_pruning(False))
    fine = make_config(
        grid={
            "n_r": 30,
            "n_a": 16,
            "poisson_l_max": 11,
            "pruning": {"enabled": False},
            "basis_cutoff": 0.0,
        }
    )
    e_f, f_f = run(fine)
    assert abs(e_p - e_u) < 0.1 * abs(e_u - e_f) + 1e-8
    assert np.abs(f_p - f_u).max() < 0.1 * np.abs(f_u - f_f).max()


def test_pruned_hartree_matches_unpruned_for_the_same_density(h2o):
    cfg_p, cfg_u = _pruning(True), _pruning(False)
    res = run_scf(h2o, build_basis(h2o, cfg_u), cfg_u, verbose=0)
    probe = np.array([[0.0, 0.0, 3.0], [2.0, -1.0, 0.5], [0.0, 1.5, -1.0]])
    gp = grid_from_config(h2o, cfg_p.grid)
    from pyxany.basis.evaluate import chi_matrix, density_on_grid

    rho_p = density_on_grid(chi_matrix(res.basis, h2o.positions, gp.points), res.P)
    v_p = hartree_potential(gp, rho_p, h2o.positions, l_max=11, eval_points=probe)
    v_u = hartree_potential(
        res.grid, res.terms.density(res.P), h2o.positions, l_max=11, eval_points=probe
    )
    assert np.abs(v_p - v_u).max() < 1e-4


# -- block-sparse basis -----------------------------------------------------


@pytest.fixture
def two_waters():
    """Two H2O 12 A apart: far enough for blocks to drop the other molecule."""
    w = [("O", 0, 0, 0.1173), ("H", 0, 0.7572, -0.4692), ("H", 0, -0.7572, -0.4692)]
    return make_molecule("2h2o", w + [(s, x + 12.0, y, z) for s, x, y, z in w])


def test_sparse_terms_match_dense_ones(two_waters):
    cfg = make_config(grid={"n_r": 25, "n_a": 10, "basis_cutoff": 1e-6})
    basis = build_basis(two_waters, cfg)
    grid = grid_from_config(two_waters, cfg.grid)
    dense = GridTerms(basis, two_waters, grid)
    sparse = GridTerms(basis, two_waters, grid, cutoff=1e-6)
    assert sparse.sparsity["kept_fraction"] < 0.8
    rng = np.random.default_rng(0)
    A = rng.normal(size=(len(basis), len(basis)))
    P = A @ A.T / len(basis)
    rho_d, rho_s = dense.density(P), sparse.density(P)
    assert np.abs(rho_d - rho_s).max() < 1e-5 * np.abs(rho_d).max()
    v = rng.normal(size=grid.n_local)
    M_d, M_s = dense.potential_matrix(v), sparse.potential_matrix(v)
    assert np.abs(M_d - M_s).max() < 1e-5 * np.abs(M_d).max()


def test_sparse_scf_reproduces_the_dense_energy(two_waters):
    def energy(cut):
        cfg = make_config(grid={"n_r": 25, "n_a": 10, "basis_cutoff": cut})
        return run_scf(two_waters, build_basis(two_waters, cfg), cfg, verbose=0).total_energy

    assert abs(energy(1e-10) - energy(0.0)) < 1e-8


# -- reporting --------------------------------------------------------------


def test_every_grid_is_reported(h2o):
    from pyxany.grid.summary import grid_summary, summary_lines

    cfg = make_config()
    res = run_scf(h2o, build_basis(h2o, cfg), cfg, verbose=0)
    d = grid_summary(cfg, res.grid, h2o, res.terms)
    assert set(d) >= {"molecular", "pair", "atom", "guess", "one_center", "basis_cutoff"}
    assert d["molecular"]["points"] == len(res.grid)
    assert list(d["molecular"]["radial_points_by_species"]) == ["H", "O"]  # by Z
    text = "\n".join(summary_lines(d))
    for word in ("molecular", "pair", "atom", "guess", "one-center", "basis cutoff"):
        assert word in text


# -- Lebedev angular grids --------------------------------------------------


@pytest.mark.parametrize("n_a", [4, 8, 12, 15])
def test_lebedev_grid_matches_the_product_grid_degree(n_a):
    """Both rules must integrate Y_lm Y_l'm' exactly for l, l' <= n_a - 1."""
    from pyxany.basis.solid_harmonics import real_spherical_harmonics
    from pyxany.grid.angular import angular_grid

    d, w = angular_grid("lebedev", n_a)
    assert len(d) < len(angular_grid("product", n_a)[0])
    assert w.sum() == pytest.approx(4 * np.pi, rel=1e-13)
    Y = real_spherical_harmonics(n_a - 1, d[:, 0], d[:, 1], d[:, 2], np.ones(len(d)))
    gram = (Y * w[:, None]).T @ Y
    assert np.abs(gram - np.eye(len(gram))).max() < 1e-12


def test_lebedev_scf_is_close_to_the_product_grid(h2o):
    def run(angular):
        cfg = make_config(grid={"n_r": 30, "n_a": 14, "angular": angular})
        res = run_scf(h2o, build_basis(h2o, cfg), cfg, verbose=0)
        return res

    leb, prod = run("lebedev"), run("product")
    assert len(leb.grid) < 0.8 * len(prod.grid)
    assert leb.energies["n_electrons_integrated"] == pytest.approx(10.0, abs=1e-3)
    assert abs(leb.total_energy - prod.total_energy) < 1e-4


def test_unknown_angular_rule_is_rejected():
    with pytest.raises(ConfigError, match="angular"):
        make_config(grid={"n_r": 25, "n_a": 10, "angular": "lebedov"})


@pytest.mark.parametrize("raw, new", [
    ({"optimise": {"f_max": 1e-3}}, "optimize"),
    ({"grid": {"one_centre": {"n_r": 400}}}, "one_center"),
    ({"scf": {"alpha": "alpha_hf_polarised"}}, "alpha_hf_polarized"),
])
def test_british_spellings_name_their_american_replacement(raw, new):
    with pytest.raises(ConfigError, match=new):
        make_config(**raw)
