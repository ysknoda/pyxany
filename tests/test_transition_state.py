"""Slater transition state (ADR 0006): Janak, Delta-SCF, core-hole localization."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.config import Config, ConfigError
from pyxany.constants import HARTREE_TO_EV
from pyxany.properties import loewdin_orbital_weights
from pyxany.scf import run_scf

WATER = [("O", 0, 0, 0.1173), ("H", 0, 0.7572, -0.4692), ("H", 0, -0.7572, -0.4692)]
N2 = [("N", 0, 0, 0), ("N", 0, 0, 1.098)]
# l_max = n_a - 1: the multipole truncation is what limits the Janak check
# (measured on H2O: residual 5.8e-3 Ha at l_max = 2, 2.4e-4 at 4, 1.4e-5 at 13)
GRID = {"n_r": 30, "n_a": 12, "poisson_l_max": 11}
SCF = {"max_iter": 200, "e_tol": 1e-11, "d_tol": 1e-9}


def _run(geom, ts=None, basis="dz", charge=0, mult=1):
    mol = make_molecule("m", geom, multiplicity=mult)
    mol.charge = charge
    kw = {"basis": {"set": basis}, "grid": GRID, "scf": SCF}
    if ts is not None:
        kw["transition_state"] = ts
    cfg = make_config(**kw)
    basis_set = build_basis(mol, cfg)
    res = run_scf(mol, basis_set, cfg, verbose=0)
    assert res.converged
    return res, basis_set


def test_config_rejects_inconsistent_requests():
    base = {"molecule": {"structure": "x.xyz"}}
    for bad in ({"remove": "homo", "excite": {"from": "homo", "to": "lumo"}},
                {"remove": 0}, {"remove": {"atom": 1, "shell": "1p"}},
                {"excite": {"from": "homo"}}, {"remove": "homo", "electrons": 1.5}):
        with pytest.raises(ConfigError):
            Config.from_dict({**base, "transition_state": bad})
    with pytest.raises(ConfigError, match="smearing"):
        Config.from_dict({**base, "scf": {"smearing": 0.01},
                          "transition_state": {"remove": "homo"}})


def test_janak_derivative_matches_the_transition_state_eigenvalue():
    """dE/dx (x = electrons removed) by central differences equals -eps(TS)."""
    ts, _ = _run(WATER, {"remove": "homo"})
    d = 0.01
    ep = _run(WATER, {"remove": "homo", "electrons": 0.5 + d})[0].total_energy
    em = _run(WATER, {"remove": "homo", "electrons": 0.5 - d})[0].total_energy
    assert (ep - em) / (2 * d) == pytest.approx(-ts.transition_state["eps_source"], abs=1e-4)


def test_ionization_energy_agrees_with_delta_scf():
    """TS and Delta-SCF differ only at third order (measured 0.03 eV for H2O)."""
    e0 = _run(WATER)[0].total_energy
    ec = _run(WATER, charge=1, mult=2)[0].total_energy
    ts = _run(WATER, {"remove": "homo"})[0]
    ip_ts = -ts.transition_state["eps_source"] * HARTREE_TO_EV
    assert ip_ts == pytest.approx((ec - e0) * HARTREE_TO_EV, abs=0.1)


@pytest.mark.slow
def test_core_hole_localizes_on_one_of_two_equivalent_atoms():
    """N2 starts with 1s sigma_g/sigma_u at 0.50 each; the hole must end on atom 1."""
    res, basis = _run(N2, {"remove": {"atom": 1, "shell": "1s"}})
    i = res.transition_state["source"]
    on_atom1 = np.array([f.atom_idx == 0 for f in basis])
    assert loewdin_orbital_weights(res.C_a, res.S, on_atom1)[i] > 0.95
    assert res.occ_a[i] == pytest.approx(0.5)
    # a 1s binding energy, not a valence one (N2 measured 408 eV here)
    assert 350.0 < -res.transition_state["eps_source"] * HARTREE_TO_EV < 450.0


def test_core_hole_needs_principal_quantum_numbers():
    with pytest.raises(ValueError, match="principal"):
        _run(WATER, {"remove": {"atom": 1, "shell": "1s"}}, basis="631g")


def test_excitation_moves_half_an_electron():
    res, _ = _run(WATER, {"excite": {"from": "homo", "to": "lumo"}})
    t = res.transition_state
    assert res.occ_a[t["source"]] == pytest.approx(0.5)
    assert res.occ_a[t["dest"]] == pytest.approx(0.5)
    assert res.occ_a.sum() + res.occ_b.sum() == pytest.approx(10.0)
    assert t["eps_dest"] > t["eps_source"]
