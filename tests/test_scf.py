"""End-to-end SCF: conservation laws first, then reference energies."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.constants import HARTREE_TO_EV
from pyxany.scf import run_scf


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


def test_non_convergence_is_diagnosed():
    """A stalled SCF must say why, not just that it failed."""
    from pyxany.config import SCFConfig
    from pyxany.scf.driver import diagnose_non_convergence

    history = [{"E": -10.0 + 1e-6 * (i % 2), "dE": 1e-6, "error": 1e-3,
                "d_rms": 1e-2, "method": "DIIS", "iter": i} for i in range(20)]
    eps = np.array([-1.0, -0.5, 0.1000, 0.1001, 0.5])
    occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0])
    lines = "\n".join(diagnose_non_convergence(history, eps, occ, SCFConfig()))
    assert "Near-degenerate" in lines
    assert "scf.smearing" in lines


def test_smearing_resolves_a_degenerate_frontier():
    """The suggested remedy has to actually work."""
    from pyxany.config import SCFConfig
    from pyxany.scf.driver import diagnose_non_convergence

    history = [{"E": -10.0, "dE": 1.0, "error": 1.0, "d_rms": 1.0,
                "method": "EDIIS", "iter": i} for i in range(5)]
    eps = np.array([-1.0, -0.5, 0.10, 0.40, 0.9])
    occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0])
    # a wide gap must not be blamed on degeneracy
    lines = "\n".join(diagnose_non_convergence(history, eps, occ, SCFConfig()))
    assert "Near-degenerate" not in lines


# Total energies at the shared coarse test grid (sz, n_r=25, n_a=10, l_max=4).
# These are pinned so that a numerical regression anywhere in the chain -- grid,
# Poisson solver, Xalpha, occupations -- shows up as a failing number rather
# than as a plausible-looking energy.  The only prior guard was a +/-10% band
# against Hartree-Fock, which is 50x too loose to see a real defect: the radial
# Poisson solver was integrating partial sums of a Gauss-Legendre rule as though
# they were cumulative integrals, which moved H2O by 0.15 Ha (4 eV) and passed.
#
# Before changing a value here, establish that the new number is *better* --
# against a closed form, or by grid convergence -- and say so in the commit.
# The physics these rest on is checked independently: V_H against
# erf(sqrt(a)r)/r at 4th order (tests/test_coulomb.py), Xalpha against Dirac
# exchange (tests/test_xc.py), and the hydrogenic spectrum
# (tests/test_pseudo_h_basis.py).
# Re-pinned 2026-10-01 for the ratio-damped Poisson solver (+0.01 to +18 mHa on this
# coarse grid) and again for per-element radial sizes (-0.08 to +4.5 mHa;
# H gets 30 and C 21 radial points at n_r = 25); the new solver is ~3x more accurate per radial
# grid size, see pyxany/integrals/coulomb.py).
REFERENCE_ENERGIES = {
    "h2": -1.063266,
    "h2o": -75.207356,
    "ch3": -39.040999,
    "co": -111.578590,
}

REFERENCE_GEOMETRIES = {
    "h2": ([("H", 0, 0, 0), ("H", 0, 0, 0.741)], 1),
    "h2o": ([("O", 0, 0, 0.1173), ("H", 0, 0.7572, -0.4692),
             ("H", 0, -0.7572, -0.4692)], 1),
    "ch3": ([("C", 0, 0, 0), ("H", 1.079, 0, 0), ("H", -0.5395, 0.9345, 0),
             ("H", -0.5395, -0.9345, 0)], 2),
    "co": ([("C", 0, 0, 0), ("O", 0, 0, 1.128)], 1),
}


@pytest.mark.parametrize("name", sorted(REFERENCE_ENERGIES))
def test_total_energy_regression(name):
    geometry, multiplicity = REFERENCE_GEOMETRIES[name]
    molecule = make_molecule(name, geometry, multiplicity=multiplicity)
    cfg = make_config()
    result = run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)
    assert result.converged
    # 1e-5 Ha is far below any physical effect but far above BLAS-level jitter,
    # so this fails on a real numerical change and not on the platform.
    assert result.total_energy == pytest.approx(REFERENCE_ENERGIES[name], abs=1e-5)


def test_multiplicity_must_match_the_electron_count():
    """A radical left at the default mult=1 must fail, not converge a half state.

    CH3 has 9 electrons; at mult=1 the alpha/beta split is 4.5/4.5, which is not
    any real spin state, and the SCF used to converge it silently 26 mHa above
    the true doublet.
    """
    molecule = make_molecule("ch3", REFERENCE_GEOMETRIES["ch3"][0], multiplicity=1)
    cfg = make_config()
    with pytest.raises(ValueError, match="impossible for 9 electrons"):
        run_scf(molecule, build_basis(molecule, cfg), cfg, verbose=0)


def test_unrestricted_gap_spans_both_spin_channels(ch3):
    """The reported gap must not be the alpha channel's.

    For CH3 the beta LUMO lies *below* the alpha LUMO, so taking the alpha
    channel alone does not merely omit information -- it names the wrong
    orbital as the LUMO and reports a gap several times too large.
    """
    cfg = make_config()
    result = run_scf(ch3, build_basis(ch3, cfg), cfg, verbose=0)
    assert result.unrestricted

    by_spin = result.homo_lumo_by_spin()
    (homo_a, lumo_a), (homo_b, lumo_b) = by_spin["alpha"], by_spin["beta"]
    assert lumo_b < lumo_a, "test premise: the beta LUMO should be the lower one"

    homo, lumo = result.homo_lumo()
    assert homo == pytest.approx(max(homo_a, homo_b))
    assert lumo == pytest.approx(min(lumo_a, lumo_b))
    # and therefore strictly smaller than either channel's own gap
    assert result.gap_ev() < (lumo_a - homo_a) * HARTREE_TO_EV
    assert result.gap_ev() < (lumo_b - homo_b) * HARTREE_TO_EV


def test_restricted_frontier_is_unchanged(h2o):
    """The closed-shell path must not be disturbed by the unrestricted fix."""
    cfg = make_config()
    result = run_scf(h2o, build_basis(h2o, cfg), cfg, verbose=0)
    assert not result.unrestricted
    homo, lumo = result.homo_lumo()
    occ, eps = result.occupations, result.eps
    assert homo == pytest.approx(float(eps[occ > 1e-8].max()))
    assert list(result.homo_lumo_by_spin()) == ["restricted"]


def test_loewdin_weights_are_bounded_and_complete(h2o):
    """Loewdin orbital weights lie in [0, 1] and sum to 1; charges sum to the net charge."""
    from pyxany.properties import loewdin, loewdin_orbital_weights

    cfg = make_config(basis={"set": "dz"})
    basis = build_basis(h2o, cfg)
    res = run_scf(h2o, basis, cfg, verbose=0)
    atom_of = np.array([f.atom_idx for f in basis])
    per_atom = np.array([loewdin_orbital_weights(res.C, res.S, atom_of == a)
                         for a in range(len(h2o))])
    assert per_atom.min() >= -1e-12 and per_atom.max() <= 1 + 1e-12
    assert np.allclose(per_atom.sum(axis=0), 1.0, atol=1e-10)
    assert loewdin(res.P, res.S, basis, h2o)["charges"].sum() == pytest.approx(0.0, abs=1e-8)


def test_loewdin_equals_mulliken_in_an_orthonormal_basis():
    from types import SimpleNamespace

    from pyxany.properties import loewdin, mulliken

    rng = np.random.default_rng(0)
    P = rng.normal(size=(4, 4))
    P = P @ P.T
    basis = [SimpleNamespace(atom_idx=i // 2) for i in range(4)]
    mol = type("M", (), {"charges": np.array([1.0, 1.0]), "__len__": lambda self: 2})()
    a = loewdin(P, np.eye(4), basis, mol)["charges"]
    b = mulliken(P, np.eye(4), basis, mol)["charges"]
    assert np.allclose(a, b)


def test_core_gap_skips_terminator_states():
    from pyxany.properties import core_gap

    eps = np.array([-0.30, -0.20, -0.05, 0.02, 0.10])
    occ = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    weights = np.array([0.1, 0.2, 0.8, 0.3, 0.1])   # LUMO is a terminator state
    g = core_gap(eps, occ, weights, max_weight=0.5)
    assert g["gap"] == pytest.approx(0.15)
    assert g["core_gap"] == pytest.approx(0.22)
    assert g["n_terminator_states_below_core_lumo"] == 1


def test_overlap_population_sum_rule_and_mo_decomposition(h2o):
    """Net + bond overlap populations add to N; the MO decomposition sums back."""
    from pyxany.properties import mo_overlap_population, overlap_population

    cfg = make_config(basis={"set": "dz"})
    basis = build_basis(h2o, cfg)
    res = run_scf(h2o, basis, cfg, verbose=0)
    op = overlap_population(res.P, res.S, basis, h2o)
    assert np.allclose(op, op.T)
    assert np.trace(op) + op[np.triu_indices(3, 1)].sum() == pytest.approx(10.0, abs=1e-8)
    assert op[0, 1] > 0.0, "O-H must be bonding"
    pairs = [(0, 1), (0, 2), (1, 2)]
    per_mo = mo_overlap_population(res.C, res.S, basis, pairs)
    assert np.allclose(per_mo.T @ res.occupations, [op[a, b] for a, b in pairs], atol=1e-10)


def test_loewdin_orbital_groups_partition_every_orbital(h2o):
    from pyxany.properties import loewdin_orbital_groups

    cfg = make_config(basis={"set": "dzp"})
    basis = build_basis(h2o, cfg)
    res = run_scf(h2o, basis, cfg, verbose=0)
    labels = [f"{h2o[f.atom_idx].element} {'spdf'[f.l]}" for f in basis]
    groups = loewdin_orbital_groups(res.C, res.S, labels)
    assert set(groups) == set(labels) and "O d" in groups
    assert np.allclose(sum(groups.values()), 1.0, atol=1e-10)


def test_sad_guess_starts_close_to_the_converged_energy(h2o):
    """The projected-atomic-density guess carries the core shape a diagonal guess loses.

    Measured on H2O/DZ: first-iteration energy 0.19 Ha above the converged one
    (the earlier diagonal occupation guess: 20 Ha above; for Si47H60 2700 Ha,
    from which DIIS never recovered).
    """
    from pyxany.scf.guess import sad_density

    cfg = make_config(basis={"set": "dz"}, scf={"max_iter": 80})
    basis = build_basis(h2o, cfg)
    P0 = sad_density(h2o, basis)
    assert np.allclose(P0, P0.T)
    res = run_scf(h2o, basis, cfg, verbose=0)
    assert np.einsum("mn,nm->", P0, res.S) == pytest.approx(10.0, abs=0.05)
    assert abs(res.history[0]["E"] - res.total_energy) < 1.0
