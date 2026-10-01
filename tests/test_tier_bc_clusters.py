"""Tier B/C validation on the passivated semiconductor clusters in examples/.

Tier B (physics, X-alpha-robust):

* Ionicity ordering. The Mulliken anion charge must follow the
  electronegativity difference of the pair (Pauling: MgO 2.13 > CdSe 0.86 >
  GaAs 0.37). Measured 2026-10-01: O -0.89, Se -0.81, As -0.60. Signs and
  ordering only; Mulliken magnitudes are basis-dependent.
* Pseudo-hydrogen passivation removes gap states. With ordinary hydrogen a
  III-V cluster violates electron counting and has a partly filled frontier
  orbital on the terminators; with fractional-charge terminators the gap opens.
  Measured on GaAs4: gap 0.07 eV with H (frontier weight on terminators 0.31 /
  0.54) against 5.31 eV with pseudo-H -- the effect reported by Deng et al.,
  Phys. Rev. B 85, 195328 (2012).

Deliberately *not* tested: the HOMO-LUMO gap ordering of the cluster series
against bulk band gaps. These 5-7 heavy-atom clusters are dominated by quantum
confinement and differ in size; measured gaps are MgO 5.85 > Si 5.80 >
GaAs 5.31 > CdSe 2.49 eV, nothing like the bulk order, and there is no
physical reason they should be.

Tier C (regression): total energy and gap of each pseudo-passivated example at
its own input settings, pinned to the values measured on 2026-10-01. A change
here is not necessarily wrong, but it must be explained.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest

from pyxany.basis import build_basis
from pyxany.constants import HARTREE_TO_EV
from pyxany.io.yaml_input import load_config, load_molecule
from pyxany.molecule import Atom, Molecule
from pyxany.properties import mulliken
from pyxany.scf import run_scf

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# (total energy / Ha, HOMO-LUMO gap / eV) at each example's own settings
PINNED = {
    "si5h12": (-1443.2765500960, 5.7952),
    "ga4ash": (-10852.8588868731, 5.3114),
    "cdse4h": (-15044.3382876139, 2.4883),
    "mgo6h": (-646.5753034646, 5.8512),
}


def _frontier(res):
    if res.unrestricted:
        eps, occ, C = res.eps_a, res.occ_a, res.C_a
    else:
        eps, occ, C = res.eps, res.occupations, res.C
    homo = int(np.where(occ > 1e-8)[0].max())
    return eps, homo, C


@cache
def _run(name: str, termination: str = "pseudo"):
    cfg = load_config(EXAMPLES / f"{name}.yaml")
    cfg.output.figure = cfg.output.json = cfg.output.timing = False
    mol = load_molecule(cfg)
    if termination == "hydrogen":
        atoms = [Atom.from_element("H", a.pos) if a.is_pseudo else a for a in mol]
        n_e = round(sum(a.Z for a in atoms) - mol.charge)
        mol = Molecule(atoms, charge=mol.charge, multiplicity=1 + n_e % 2, name=f"{name}_H")
    basis = build_basis(mol, cfg)
    res = run_scf(mol, basis, cfg, verbose=0)
    return mol, basis, res


def _gap_ev(res):
    eps, homo, _ = _frontier(res)
    return float((eps[homo + 1] - eps[homo]) * HARTREE_TO_EV)


def _anion_charge(name, anion):
    mol, basis, res = _run(name)
    assert res.converged
    q = mulliken(res.P, res.S, basis, mol)["charges"]
    return float(np.mean([q[i] for i, a in enumerate(mol) if a.element == anion]))


def test_ionicity_follows_electronegativity_difference():
    q_as = _anion_charge("ga4ash", "As")
    q_se = _anion_charge("cdse4h", "Se")
    q_o = _anion_charge("mgo6h", "O")
    assert q_o < q_se < q_as < 0


def test_pseudo_hydrogen_opens_the_gap_that_plain_hydrogen_closes():
    _, _, pseudo = _run("ga4ash", "pseudo")
    mol_h, basis_h, plain = _run("ga4ash", "hydrogen")
    assert pseudo.converged and plain.converged
    assert _gap_ev(pseudo) > 3.0
    assert _gap_ev(plain) < 0.5
    # ... and with plain H the frontier orbital sits largely on the terminators
    _, homo, C = _frontier(plain)
    term = np.array([basis_h[m].atom_idx for m in range(len(basis_h))])
    on_term = np.isin(term, [i for i, a in enumerate(mol_h) if a.element == "H"])
    SC = plain.S @ C
    weight = float(np.sum((C[:, homo + 1] * SC[:, homo + 1])[on_term]))
    assert weight > 0.3


@pytest.mark.parametrize("name", list(PINNED))
def test_pinned_cluster_regression(name):
    _, _, res = _run(name)
    assert res.converged
    e_ref, gap_ref = PINNED[name]
    assert res.total_energy == pytest.approx(e_ref, abs=1e-6)
    assert _gap_ev(res) == pytest.approx(gap_ref, abs=1e-3)
