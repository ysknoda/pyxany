"""The shipped data tables must stay internally consistent and unmodified.

These tables were extracted mechanically from the original single-file
``dv-xa.py`` by ``tools/extract_tables.py``, and the extraction was verified
key-by-key against that source before it was removed (see
``docs/legacy_comparison.md``).  That source is gone, so the checks here are
deliberately self-contained and of two independent kinds:

* **Physical/structural invariants** -- a neutral atom's occupations must sum to
  Z, quantum numbers must satisfy ``0 <= l < n`` and ``|m| <= l``, every shell
  must carry a complete set of ``2l+1`` m-components, exponents must be
  positive.  These constrain the *content*, so a deliberate table edit still has
  to be physically coherent.  (The Cr entry carried Fe's ``3d`` occupancy and
  described a 25-electron neutral Cr until this check was added.)
* **Content checksums** -- pinned SHA-256 prefixes catch any *unintended*
  change, including one that happens to satisfy every invariant above.

If you intentionally change a table, update the checksum in the same commit and
say why in the message.  Never update a checksum to make a failing invariant
pass.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import pytest

from pyxany.data import elements as elem
from pyxany.data.loaders import coverage_report, gto_table, sto_table
from pyxany.data.radii import covalent_radius, source as radii_source

TABLES = Path(elem.__file__).resolve().parent / "tables"

STO_TABLES = ["sz", "dz", "tz", "pol_d", "pol_f"]

# SHA-256 prefixes of the verified tables.  See the module docstring before
# changing any of these.
CHECKSUMS = {
    "covalent_radii": "d9f96af78637e405",
    "elements": "2d0e607f2fffaa54",
    "gto_631g": "9f727a2e2746f483",
    "gto_631gd_pol": "f6c457217d1a29a7",
    "gto_sto3g": "941711c4444cd8a0",
    "orbital_labels": "6224dbe90ee67d95",
    # measured by tools/radial_grid_sizes.py (2026-10-01), see ADR 0008
    "radial_points": "ab17c9febceb8135",
    "sto_dz": "5b3b9a388a2a4628",
    "sto_pol_d": "b11eaa7f0d7f51ae",
    "sto_pol_f": "f5219201d99b068f",
    "sto_sz": "60857d4c8864bcba",
    "sto_tz": "bf289182bc9128a2",
    # Schwarz, PRB 5, 2466 (1972) and TCA 34, 225 (1974), transcribed 2026-10-05 (ADR 0011)
    "schwarz_alpha": "efe54b9293bb5fd1",
    # tools/alpha_hf.py; H-Sc so far, rewritten when the H-Rn run finishes (ADR 0011)
    "alpha_hf_polarized": "7d8d2447d4ed3e0d",
}


# --------------------------------------------------------------------------
# content integrity
# --------------------------------------------------------------------------

def test_every_table_is_present_and_ships_with_the_package():
    """A packaging error would silently drop a table; the loader path must find it."""
    found = {p.stem for p in TABLES.glob("*.json")}
    assert found == set(CHECKSUMS), f"table set changed: {found ^ set(CHECKSUMS)}"


@pytest.mark.parametrize("name", sorted(CHECKSUMS))
def test_table_content_is_unchanged(name):
    digest = hashlib.sha256((TABLES / f"{name}.json").read_bytes()).hexdigest()[:16]
    assert digest == CHECKSUMS[name], (
        f"{name}.json changed (got {digest}). If this was intentional, update "
        f"CHECKSUMS and explain the change in the commit message."
    )


# --------------------------------------------------------------------------
# element table: physical invariants
# --------------------------------------------------------------------------

def test_neutral_atom_occupations_sum_to_atomic_number():
    """A neutral atom holds exactly Z electrons.

    This is the check that catches a mis-copied configuration: Cr shipped with
    Fe's ``3d`` occupancy (1.2 per component = 3d^6) alongside its own 4s^1, so
    the "neutral" Cr atom carried 25 electrons.  That biases the atomic
    superposition guess and makes the NAO basis for Cr the basis of Cr-.
    """
    for symbol in elem.symbols():
        z = elem.atomic_number(symbol)
        total = sum(elem.occupations(symbol).values())
        assert total == pytest.approx(z, abs=1e-9), (
            f"{symbol}: occupations sum to {total} but Z = {z}"
        )


def test_atomic_numbers_are_unique_and_cover_h_to_rn():
    numbers = sorted(elem.atomic_number(s) for s in elem.symbols())
    assert numbers == list(range(1, 87)), "expected a contiguous H(1)-Rn(86) table"


def test_subshell_occupations_are_spherically_averaged():
    """Every m-component of one subshell carries the same occupation.

    The code assumes a spherically averaged free atom throughout (the atomic
    solver, the superposition guess and the NAO generator all collapse labels to
    ``(n, l)``).  An uneven split would silently break that assumption.
    """
    for symbol in elem.symbols():
        by_shell = defaultdict(list)
        for label, ne in elem.occupations(symbol).items():
            n = next(int(c) for c in label if c.isdigit())
            by_shell[(n, elem._subshell_l(label))].append(ne)
        for (n, l), values in by_shell.items():
            assert len(set(values)) == 1, f"{symbol} {n}{'spdf'[l]}: uneven {values}"


def test_subshells_never_exceed_their_capacity():
    """A subshell of angular momentum l holds at most 2(2l+1) electrons."""
    for symbol in elem.symbols():
        for (n, l), q in elem._subshell_totals(symbol).items():
            assert -1e-12 <= q <= 2 * (2 * l + 1) + 1e-12, f"{symbol} {n}{'spdf'[l]}: {q}"


def test_bragg_slater_radii_are_positive_and_physical():
    """Grid radii scale every atomic quadrature; a bad one corrupts integration."""
    for symbol in elem.symbols():
        r = elem.bragg_slater_radius(symbol)
        assert 0.3 < r < 6.0, f"{symbol}: Bragg-Slater radius {r} Bohr is out of range"
    # H is the smallest atom in the table.
    assert elem.bragg_slater_radius("H") == min(
        elem.bragg_slater_radius(s) for s in elem.symbols())


# --------------------------------------------------------------------------
# basis tables: structural invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", STO_TABLES)
def test_sto_quantum_numbers_are_valid(name):
    for symbol, functions in sto_table(name).items():
        for n, l, m, zeta in functions:
            assert n >= 1, f"{name}/{symbol}: n = {n}"
            assert 0 <= l < n, f"{name}/{symbol}: l = {l} with n = {n}"
            assert abs(m) <= l, f"{name}/{symbol}: |m| = {abs(m)} > l = {l}"
            assert zeta > 0.0, f"{name}/{symbol}: zeta = {zeta}"


@pytest.mark.parametrize("name", STO_TABLES)
def test_sto_shells_carry_a_complete_set_of_m_components(name):
    """Each (n, l, zeta) shell needs all 2l+1 m values.

    A partial shell is not closed under rotation, so the computed energy would
    depend on how the molecule happens to be oriented in the input file.
    """
    for symbol, functions in sto_table(name).items():
        shells = defaultdict(set)
        for n, l, m, zeta in functions:
            shells[(n, l, zeta)].add(m)
        for (n, l, zeta), ms in shells.items():
            assert ms == set(range(-l, l + 1)), (
                f"{name}/{symbol}: shell {n}{'spdf'[l]} zeta={zeta} has m = {sorted(ms)}"
            )


@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_gto_primitives_are_positive_and_finite(name):
    """Gaussian exponents must be positive; a non-positive one diverges."""
    for symbol, shells in gto_table(name).items():
        for label, primitives in shells.items():
            assert primitives, f"{name}/{symbol}/{label}: empty contraction"
            for primitive in primitives:
                # (exponent, coeff) for a plain shell; (exponent, s, p) for sp.
                assert len(primitive) in (2, 3), f"{name}/{symbol}/{label}: {primitive}"
                exponent, *coeffs = primitive
                assert exponent > 0.0, f"{name}/{symbol}/{label}: exponent {exponent}"
                for c in coeffs:
                    assert math.isfinite(c), f"{name}/{symbol}/{label}: coefficient {c}"


# --------------------------------------------------------------------------
# coverage, radii and derived properties (unchanged, already self-contained)
# --------------------------------------------------------------------------

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


def test_half_filled_shell_anomalies_give_the_spectroscopic_multiplicity():
    """Cr (7S3) and Cu (2S1/2) are the classic 3d/4s exceptions.

    Hund's first rule applied to the tabulated occupations reproduces the NIST
    ground-level multiplicity only if those configurations are stored correctly,
    so this pins the shape of the anomaly, not just the electron count.
    """
    assert elem.ground_state_multiplicity("Cr") == 7   # [Ar] 3d5 4s1
    assert elem.ground_state_multiplicity("Cu") == 2   # [Ar] 3d10 4s1
