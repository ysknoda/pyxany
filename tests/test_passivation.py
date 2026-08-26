"""Pseudo-hydrogen passivation: charges, placement and electron balance."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from dvxa.constants import ANG_TO_BOHR
from dvxa.molecule import Atom, Molecule
from dvxa.passivation import (
    ANION,
    CATION,
    VASP_CHARGES,
    as_fraction,
    neighbour_lists,
    passivate,
    terminator_charge_for,
    terminator_distance,
)

#: (material, cation spec, anion spec, expected charges) from the counting rule.
MATERIALS = [
    ("Si diamond", ("Si", CATION, 4), ("Si", ANION, 4), "1", "1"),
    ("SiC", ("Si", CATION, 4), ("C", ANION, 4), "1", "1"),
    ("GaAs", ("Ga", CATION, 4), ("As", ANION, 4), "5/4", "3/4"),
    ("InP", ("In", CATION, 4), ("P", ANION, 4), "5/4", "3/4"),
    ("GaN", ("Ga", CATION, 4), ("N", ANION, 4), "5/4", "3/4"),
    ("ZnO", ("Zn", CATION, 4), ("O", ANION, 4), "3/2", "1/2"),
    ("CdSe", ("Cd", CATION, 4), ("Se", ANION, 4), "3/2", "1/2"),
    ("ZnS", ("Zn", CATION, 4), ("S", ANION, 4), "3/2", "1/2"),
    ("MgO rock salt", ("Mg", CATION, 6), ("O", ANION, 6), "5/3", "1/3"),
    ("TiO2 rutile", ("Ti", CATION, 6), ("O", ANION, 3), "4/3", "2/3"),
    ("CuCl", ("Cu", CATION, 4), ("Cl", ANION, 4), "7/4", "1/4"),
]


@pytest.mark.parametrize("name,cation,anion,exp_cation,exp_anion", MATERIALS)
def test_electron_counting_rule(name, cation, anion, exp_cation, exp_anion):
    """alpha = (8-Z)/N for anions, (2M-Z)/M for cations  [PRB 85, 195328 (2012)]."""
    a_cat = terminator_charge_for(*cation)
    a_an = terminator_charge_for(*anion)
    assert as_fraction(a_cat) == Fraction(exp_cation), name
    assert as_fraction(a_an) == Fraction(exp_anion), name


@pytest.mark.parametrize("name,cation,anion,exp_cation,exp_anion", MATERIALS)
def test_charges_are_standard_pseudo_hydrogen_values(name, cation, anion,
                                                     exp_cation, exp_anion):
    """Every charge the rule produces for these materials ships with VASP."""
    standard = set(VASP_CHARGES) | {Fraction(1)}
    assert Fraction(exp_cation) in standard, name
    assert Fraction(exp_anion) in standard, name


def test_charge_depends_on_coordination_not_just_the_element():
    """Oxygen carries 1/2 in a tetrahedral site but 1/3 in an octahedral one."""
    assert as_fraction(terminator_charge_for("O", ANION, 4)) == Fraction("1/2")
    assert as_fraction(terminator_charge_for("O", ANION, 6)) == Fraction("1/3")
    assert as_fraction(terminator_charge_for("O", ANION, 3)) == Fraction("2/3")
    assert as_fraction(terminator_charge_for("Mg", CATION, 4)) == Fraction("3/2")
    assert as_fraction(terminator_charge_for("Mg", CATION, 6)) == Fraction("5/3")


def cluster(atoms):
    return Molecule([Atom.from_element(s, np.array(p) * ANG_TO_BOHR) for s, p in atoms])


def tetrahedral(centre, ligand, bond):
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    return [(centre, [0, 0, 0])] + [(ligand, d * bond) for d in dirs]


def octahedral(centre, ligand, bond):
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    return [(centre, [0, 0, 0])] + [(ligand, d * bond) for d in dirs]


CLUSTERS = [
    ("Si5", tetrahedral("Si", "Si", 2.352), 4, 12),
    ("CdSe4", tetrahedral("Cd", "Se", 2.620), 4, 12),
    ("GaAs4", tetrahedral("Ga", "As", 2.448), 4, 12),
    ("MgO6", octahedral("Mg", "O", 2.105), 6, 30),
]


@pytest.mark.parametrize("name,atoms,coordination,expected_added", CLUSTERS)
def test_terminated_cluster_has_an_even_electron_count(name, atoms, coordination,
                                                       expected_added):
    """The electron-counting rule exists to make the terminated cluster closed shell."""
    result = passivate(cluster(atoms), "pseudo_hydrogen", coordination=coordination)
    assert result.n_added == expected_added, name
    n = result.molecule.n_electrons
    assert abs(n - 2.0 * round(n / 2.0)) < 1e-8, f"{name}: {n} electrons"


def test_real_hydrogen_scheme_gives_unit_charges():
    result = passivate(cluster(tetrahedral("Cd", "Se", 2.62)), "hydrogen",
                       coordination=4)
    assert all(sp.alpha == 1.0 for sp in result.species.values())
    assert all(not a.is_pseudo for a in result.molecule)


def test_terminator_distance_matches_known_x_h_bonds():
    """The covalent-radius scaling must land near the real molecular bond lengths."""
    references = {  # element, X-Y bond in the solid, known X-H bond length (Ang)
        "Si": (2.352, 1.48),   # SiH4
        "O": (2.105, 0.958),   # H2O
        "Se": (2.620, 1.46),   # H2Se
        "C": (1.545, 1.09),    # CH4
    }
    for element, (nn, expected) in references.items():
        partner = {"Si": "Si", "O": "Mg", "Se": "Cd", "C": "C"}[element]
        d = terminator_distance(element, partner, nn)
        assert abs(d - expected) < 0.12, f"{element}: got {d:.3f}, expected ~{expected}"


def test_neighbour_detection_finds_the_right_coordination():
    mol = cluster(tetrahedral("Si", "Si", 2.352))
    counts = [len(v) for v in neighbour_lists(mol, 1.20).values()]
    assert counts[0] == 4
    assert counts[1:] == [1, 1, 1, 1]


def test_pseudo_atoms_carry_fractional_nuclear_charge():
    result = passivate(cluster(tetrahedral("Cd", "Se", 2.62)), "pseudo_hydrogen",
                       coordination=4)
    pseudo = [a for a in result.molecule if a.is_pseudo]
    assert pseudo
    assert all(a.element == "H" for a in pseudo)
    assert all(abs(a.Z - 0.5) < 1e-12 for a in pseudo)


@pytest.mark.slow
def test_terminated_cluster_has_a_clean_gap():
    """The criterion passivation exists to satisfy, checked on a real cluster.

    A terminated cluster must have (a) no terminator-derived states at the
    frontier and (b) near-neutral terminators -- that neutrality is the
    signature that the electron-counting rule was satisfied.
    """
    from conftest import make_config
    from dvxa.basis import build_basis
    from dvxa.basis.pseudo_h import derive_exponents
    from dvxa.properties import basis_function_weights, mulliken
    from dvxa.scf import run_scf

    result_p = passivate(cluster(tetrahedral("Si", "Si", 2.352)),
                         "pseudo_hydrogen", coordination=4)
    mol = result_p.molecule
    cfg = make_config(basis={"set": "dz"}, grid={"n_r": 25, "n_a": 10,
                                                 "poisson_l_max": 4},
                      scf={"max_iter": 80})
    basis = build_basis(mol, cfg, derive_exponents(mol))
    scf = run_scf(mol, basis, cfg, verbose=0)
    assert scf.converged

    homo, lumo = scf.homo_lumo()
    assert np.isfinite(homo) and np.isfinite(lumo)
    assert lumo > homo, "a passivated cluster must have a real gap"

    is_terminator = np.array([mol[f.atom_idx].is_pseudo for f in basis])
    weights = basis_function_weights(scf.C, scf.occupations, scf.S, is_terminator)
    finite = np.isfinite(scf.eps)
    order = np.argsort(np.where(finite, scf.eps, np.inf))
    n_occ = int((scf.occupations > 1e-8).sum())
    frontier = [order[r] for r in range(max(n_occ - 3, 0),
                                        min(n_occ + 3, int(finite.sum())))]
    assert weights[frontier].max() < 0.5, "terminator states appear at the frontier"

    pop = mulliken(scf.P, scf.S, basis, mol)
    terminator_q = np.array([pop["charges"][i] for i, a in enumerate(mol)
                             if a.is_pseudo])
    assert np.abs(terminator_q).max() < 0.25, "terminators are not close to neutral"
