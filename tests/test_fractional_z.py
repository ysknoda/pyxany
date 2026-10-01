"""Fractional nuclear charges must work everywhere a real element does."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config
from pyxany.basis import build_basis
from pyxany.basis.pseudo_h import derive_exponents
from pyxany.constants import ANG_TO_BOHR
from pyxany.molecule import Atom, Molecule
from pyxany.properties import mulliken
from pyxany.scf import run_scf


def pseudo_molecule(alpha=0.5, distance=1.5):
    return Molecule([Atom.from_element("Se", [0, 0, 0]),
                     Atom.pseudo_hydrogen("HSe", alpha,
                                          [0, 0, distance * ANG_TO_BOHR])],
                    name="se_h")


def test_electron_count_includes_the_fractional_charge():
    mol = pseudo_molecule(0.5)
    assert mol.n_electrons == pytest.approx(34.5)
    assert mol.charges[1] == pytest.approx(0.5)


def test_nuclear_repulsion_uses_the_fractional_charge():
    mol = pseudo_molecule(0.5, distance=1.5)
    d = np.linalg.norm(mol[1].pos - mol[0].pos)
    assert mol.nuclear_repulsion() == pytest.approx(34.0 * 0.5 / d)


def test_scf_runs_with_a_fractional_electron_count():
    mol = pseudo_molecule(0.5)
    cfg = make_config(basis={"set": "sz"}, scf={"max_iter": 60})
    basis = build_basis(mol, cfg, derive_exponents(mol))
    result = run_scf(mol, basis, cfg, verbose=0)
    assert result.energies["trace_PS"] == pytest.approx(34.5, abs=1e-6)


def test_mulliken_charges_reference_the_fractional_nucleus():
    mol = pseudo_molecule(0.5)
    cfg = make_config(basis={"set": "sz"}, scf={"max_iter": 60})
    basis = build_basis(mol, cfg, derive_exponents(mol))
    result = run_scf(mol, basis, cfg, verbose=0)
    pop = mulliken(result.P, result.S, basis, mol)
    assert pop["charges"].sum() == pytest.approx(0.0, abs=1e-6)


def test_pseudo_atom_gets_a_single_1s_function():
    mol = pseudo_molecule(0.75)
    cfg = make_config(basis={"set": "dz"})
    basis = build_basis(mol, cfg, derive_exponents(mol))
    pseudo_functions = [f for f in basis if f.atom_idx == 1]
    assert len(pseudo_functions) == 1
    assert (pseudo_functions[0].n, pseudo_functions[0].l) == (1, 0)


def test_missing_exponent_is_an_explicit_error():
    mol = pseudo_molecule(0.5)
    cfg = make_config(basis={"set": "sz"})
    with pytest.raises(KeyError, match="pseudo-hydrogen"):
        build_basis(mol, cfg, pseudo_zeta={})


def test_grid_radius_scales_with_the_fractional_charge():
    """A weakly charged terminator is more diffuse and needs a wider radial grid."""
    diffuse = Atom.pseudo_hydrogen("Ha", 0.25, [0, 0, 0]).grid_radius()
    compact = Atom.pseudo_hydrogen("Hb", 1.75, [0, 0, 0]).grid_radius()
    plain = Atom.from_element("H", [0, 0, 0]).grid_radius()
    assert diffuse > plain > compact
