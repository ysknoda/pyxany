"""Tier B validation: real molecules against experiment, on X-alpha-robust quantities.

X-alpha (alpha = 0.7, no correlation) is less accurate in absolute terms than
standard DFT, so these tests use only quantities it gets right *qualitatively*,
with tolerances derived from the functional's own measured error rather than
guessed (docs/adr/0003-validation-tiers.md). Total energies, absolute gaps and
orbital ionization energies are deliberately not tested against experiment.

Experimental data: NIST CCCBDB, parsed by tools/fetch_cccbdb.py into
tests/data/cccbdb_experimental.json (with per-value primary references).

How each tolerance was set (measured 2026-10-01):

* Bond lengths. The intrinsic X-alpha error, from PySCF with the same
  functional at cc-pVTZ (basis error removed), is -0.2 % (N2) to +4.0 % (H2);
  X-H bonds come out 1.2-1.8 % long. pyxany with the STO DZP basis adds basis
  incompleteness: measured -0.3 % (HF) to +4.2 % (H2), +3.8 % (H2S).
  Tolerance: |r / r_exp - 1| <= 5 %. LiH is excluded: the STO Li basis puts
  it at +5.8 % against an intrinsic +1.8 %, a basis limitation, not a bug.
* Dipoles. Intrinsic X-alpha / aug-cc-pVTZ is 0.94-1.07 x experiment; pyxany
  DZP (no diffuse functions) measured 0.74 (LiH) - 1.07 (H2S).
  Tolerance: ratio in [0.65, 1.35], and the direction must be right.
  CO is excluded: its 0.11 D dipole is a small difference of large terms.
* Mulliken charges. Only signs and an electronegativity ordering, for bonds
  with a large electronegativity difference. Excluded: CH4 (C at -1.1 e, a
  Mulliken basis artefact), SiH4 (the Si sign flips between DZ and DZP), H2S
  (small, basis-dependent).

Experimental bond types: r_e for H2, CH4, H2O, HF, N2, CO; unspecified in
CCCBDB for NH3, SiH4, H2S (often r_0 / r_s, which differ from r_e by ~0.005 A,
< 0.5 %, well inside the tolerance).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.constants import BOHR_TO_ANG
from pyxany.forces.optimize import optimize_geometry
from pyxany.properties import dipole_moment, mulliken
from pyxany.scf import run_scf

pytestmark = pytest.mark.slow

EXP = json.loads((Path(__file__).parent / "data" / "cccbdb_experimental.json").read_text())

BASIS = "dzp"
GRID = {"n_r": 40, "n_a": 14, "poisson_l_max": 6}
SCF = {"max_iter": 150, "e_tol": 1e-9, "d_tol": 1e-7, "alpha": 0.7}

TOL_BOND_REL = 0.05
DIPOLE_RATIO = (0.65, 1.35)


def _molecule(name):
    return make_molecule(name, [tuple(g) for g in EXP[name]["geometry_angstrom"]])


def _scf(name):
    mol = _molecule(name)
    cfg = make_config(basis={"set": BASIS}, grid=GRID, scf=SCF)
    basis = build_basis(mol, cfg)
    res = run_scf(mol, basis, cfg, verbose=0)
    assert res.converged
    return mol, basis, res


def _bond(positions_ang, elements, label):
    """Shortest distance between the two element types named in e.g. 'rSiH'."""
    a, b = re.findall(r"[A-Z][a-z]?", label[1:])
    n = len(elements)
    return min(
        np.linalg.norm(positions_ang[i] - positions_ang[j])
        for i in range(n)
        for j in range(i + 1, n)
        if {elements[i], elements[j]} == {a, b}
    )


@pytest.mark.parametrize("name", ["H2", "CH4", "NH3", "H2O", "HF", "N2",
                                  "CO", "SiH4", "H2S"])
def test_relaxed_bond_length_against_experiment(name, tmp_path):
    cfg = make_config(
        basis={"set": BASIS},
        grid=GRID,
        scf=SCF,
        forces={"enabled": True},
        optimize={"enabled": True, "f_max": 3e-4, "max_steps": 60},
        output={"dir": str(tmp_path), "prefix": name},
    )
    res = optimize_geometry(_molecule(name), cfg, verbose=0)
    assert res.converged
    pos = res.molecule.positions * BOHR_TO_ANG
    elements = [a.element for a in res.molecule]
    for coord in EXP[name]["internal_coordinates"]:
        if coord["name"].startswith("r"):
            r = _bond(pos, elements, coord["name"])
            assert abs(r / coord["value"] - 1.0) < TOL_BOND_REL, (coord["name"], r)


def _dipole_direction(mol, mu):
    """+1 if the dipole points from the heavy atom toward the hydrogens."""
    el = [a.element for a in mol]
    h = [i for i, e in enumerate(el) if e == "H"]
    heavy = next(i for i, e in enumerate(el) if e != "H")
    return np.sign(mu @ (mol.positions[h].mean(axis=0) - mol.positions[heavy]))


# Physics convention: mu points from - to +. H is the positive end except in
# LiH, where Li+ H- makes it the negative end.
@pytest.mark.parametrize(
    "name, direction", [("LiH", -1), ("NH3", +1), ("H2O", +1), ("HF", +1), ("H2S", +1)]
)
def test_dipole_against_experiment(name, direction):
    mol, _, res = _scf(name)
    mu = dipole_moment(mol, res.grid, res.terms.density(res.P))
    ratio = np.linalg.norm(mu) / EXP[name]["dipole_debye"]["value"]
    assert DIPOLE_RATIO[0] < ratio < DIPOLE_RATIO[1]
    assert _dipole_direction(mol, mu) == direction


@pytest.mark.parametrize("name", ["CH4", "N2", "SiH4"])
def test_nonpolar_molecules_have_no_dipole(name):
    mol, _, res = _scf(name)
    assert EXP[name]["dipole_debye"]["value"] == 0.0
    assert np.linalg.norm(dipole_moment(mol, res.grid, res.terms.density(res.P))) < 1e-2


def test_mulliken_signs_and_electronegativity_ordering():
    q_heavy, q_h = {}, {}
    for name in ["LiH", "NH3", "H2O", "HF"]:
        mol, basis, res = _scf(name)
        q = mulliken(res.P, res.S, basis, mol)["charges"]
        el = [a.element for a in mol]
        q_heavy[name] = q[next(i for i, e in enumerate(el) if e != "H")]
        q_h[name] = np.mean([q[i] for i, e in enumerate(el) if e == "H"])
    assert q_heavy["LiH"] > 0
    assert q_heavy["NH3"] < 0 and q_heavy["H2O"] < 0 and q_heavy["HF"] < 0
    # The more electronegative the partner (N < O < F), the more positive the H.
    assert q_h["NH3"] < q_h["H2O"] < q_h["HF"]
