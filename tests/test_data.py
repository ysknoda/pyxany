"""The extracted data tables must match the original single-file source exactly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dvxa.data import elements as elem
from dvxa.data.loaders import coverage_report, gto_table, sto_table
from dvxa.data.radii import covalent_radius, source as radii_source

LEGACY = Path(__file__).resolve().parent.parent / "dv-xa.py"


@pytest.fixture(scope="module")
def legacy():
    """The original script, if it is still present (it is removed on release)."""
    if not LEGACY.exists():
        pytest.skip("original dv-xa.py has been removed; extraction already verified")
    spec = importlib.util.spec_from_file_location("dvxa_legacy", LEGACY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dvxa_legacy"] = module
    spec.loader.exec_module(module)
    return module


def test_element_table_matches_source(legacy):
    assert set(elem.symbols()) == set(legacy.ELEMENT_DB)
    for sym in elem.symbols():
        assert elem.atomic_number(sym) == legacy.ELEMENT_DB[sym]["Z"]
        assert elem.occupations(sym) == pytest.approx(legacy.ELEMENT_DB[sym]["occ"])
        assert elem.bragg_slater_radius(sym) == pytest.approx(legacy._BRAGG[sym])


@pytest.mark.parametrize("name,attr", [("sz", "STO_SZ"), ("dz", "STO_DZ"),
                                       ("tz", "STO_TZ"), ("pol_d", "STO_POL"),
                                       ("pol_f", "STO_POL_F")])
def test_sto_tables_match_source(legacy, name, attr):
    ours = sto_table(name)
    theirs = getattr(legacy, attr)
    assert set(ours) == set(theirs)
    for sym in theirs:
        assert len(ours[sym]) == len(theirs[sym])
        for a, b in zip(ours[sym], theirs[sym]):
            assert a[:3] == tuple(b[:3])
            assert a[3] == pytest.approx(b[3])


@pytest.mark.parametrize("name,attr", [("sto3g", "GTO_STO3G"), ("631g", "GTO_631G")])
def test_gto_tables_match_source(legacy, name, attr):
    ours = gto_table(name)
    theirs = getattr(legacy, attr)
    assert set(ours) == set(theirs)
    for sym, shells in theirs.items():
        assert set(ours[sym]) == set(shells)
        for shell, prims in shells.items():
            for a, b in zip(ours[sym][shell], prims):
                assert list(a) == pytest.approx(list(b))


def test_element_coverage_is_reported():
    """The gaps must stay visible rather than silently falling back."""
    rows = dict((name, (n, note)) for name, n, note in coverage_report())
    assert rows["sto_sz"][0] == 86
    assert rows["sto_dz"][0] == 86
    assert rows["sto_tz"][0] == 54          # H-Xe; heavier elements are interpolated
    assert rows["sto_pol_f"][0] == 14       # B-Ar only
    assert "missing" in rows["sto_tz"][1]


def test_covalent_radii_are_the_cordero_values():
    """Spot check against Cordero et al., Dalton Trans. (2008) 2832, Table 2."""
    assert "Cordero" in radii_source()
    published = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
                 "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ga": 1.22,
                 "As": 1.19, "Se": 1.20, "Zn": 1.22, "Cd": 1.44, "Mg": 1.41,
                 "Ti": 1.60, "Cu": 1.32}
    for symbol, value in published.items():
        assert covalent_radius(symbol) == pytest.approx(value, abs=5e-3), symbol


def test_derived_element_properties():
    """Group, valence count and multiplicity are derived, so check them."""
    for symbol, group, valence in [("H", 1, 1), ("C", 14, 4), ("O", 16, 6),
                                   ("Ti", 4, 4), ("Cu", 11, 1), ("Zn", 12, 2),
                                   ("Ga", 13, 3), ("Se", 16, 6), ("Cd", 12, 2),
                                   ("Rn", 18, 8)]:
        assert elem.group(symbol) == group, symbol
        assert elem.valence_electrons(symbol) == valence, symbol
    for symbol, mult in [("H", 2), ("He", 1), ("C", 3), ("N", 4), ("O", 3),
                         ("F", 2), ("Ne", 1), ("Si", 3), ("P", 4), ("S", 3)]:
        assert elem.ground_state_multiplicity(symbol) == mult, symbol
        assert elem.multiplicity_is_reliable(symbol)
    assert not elem.multiplicity_is_reliable("Fe")
