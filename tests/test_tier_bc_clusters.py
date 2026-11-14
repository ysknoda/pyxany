"""Tier B/C validation on the passivated semiconductor clusters in examples/.

Tier B (physics, X-alpha-robust):

* Ionicity ordering. The Mulliken anion charge must follow the
  electronegativity difference of the pair (Pauling: MgO 2.13 > CdSe 0.86 >
  GaAs 0.37). Measured 2026-10-01: O -0.90, Se -0.81, As -0.60. Signs and
  ordering only; Mulliken magnitudes are basis-dependent.
* Pseudo-hydrogen passivation removes gap states. With ordinary hydrogen a
  III-V cluster violates electron counting and has a partly filled frontier
  orbital on the terminators; with fractional-charge terminators the gap opens.
  Measured on GaAs4: gap 0.07 eV with H against 5.31 eV with pseudo-H -- the
  effect reported by Deng et al., Phys. Rev. B 85, 195328 (2012). The pseudo-H
  HOMO is the cluster interior (Loewdin terminator weight 0.18); its LUMO is
  about half on the terminators (0.52), so the plain gap is quoted with that
  caveat -- see the core gap in the passivation report and
  docs/passivation.md.

Deliberately *not* tested: the HOMO-LUMO gap ordering of the cluster series
against bulk band gaps. These 5-7 heavy-atom clusters are dominated by quantum
confinement and differ in size; measured gaps (n_r 45) are MgO 5.89 > Si 5.80 >
GaAs 5.31 > CdSe 2.55 eV, nothing like the bulk order, and there is no
physical reason they should be.

Tier C (regression): total energy and gap of each pseudo-passivated example at
its own input settings, pinned to the values measured on 2026-11-14. A change
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

# (total energy / Ha, HOMO-LUMO gap / eV) at each example's own settings.
# Re-pinned 2026-11-14 for the default n_r 45 (ADR 0015). The n_r 30 values
# were reproduced first, to 3e-8 Ha, after the P0 fixes of ADR 0012; 30 -> 45
# moved the energies by -0.093 (si5h12), -0.129 (ga4ash), -0.144 (cdse4h),
# +0.003 Ha (mgo6h), radial quadrature error of the heavy atoms, and the gaps
# by -0.0004, +0.0013, +0.0024, +0.0204 eV.
PINNED = {
    "si5h12": (-1443.3934305764, 5.7983),
    "ga4ash": (-10853.8224996065, 5.3124),
    "cdse4h": (-15045.9811701265, 2.5455),
    "mgo6h": (-646.5957621410, 5.8872),
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
    if termination == "hydrogen":
        # Plain H closes the gap (~0.07 eV): a nearly metallic SCF that needs
        # 120-150 iterations, the exact count moving with rounding-level
        # changes (grid pruning, block summation order) -- 150 was too tight.
        cfg.scf.max_iter = 300
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


def _terminator_weights(mol, basis, C, S):
    from pyxany.properties import loewdin_orbital_weights

    mask = np.array([mol[f.atom_idx].is_pseudo or mol[f.atom_idx].element == "H"
                     for f in basis], dtype=bool)
    return loewdin_orbital_weights(C, S, mask)


def test_pseudo_hydrogen_opens_the_gap_that_plain_hydrogen_closes():
    mol_p, basis_p, pseudo = _run("ga4ash", "pseudo")
    mol_h, basis_h, plain = _run("ga4ash", "hydrogen")
    assert pseudo.converged and plain.converged
    assert _gap_ev(pseudo) > 3.0
    assert _gap_ev(plain) < 0.5
    # With plain H the frontier is largely on the terminators (Loewdin weights);
    # with pseudo-H the occupied edge is the cluster interior.
    _, homo, C = _frontier(plain)
    assert _terminator_weights(mol_h, basis_h, C, plain.S)[homo + 1] > 0.3
    _, homo_p, C_p = _frontier(pseudo)
    assert _terminator_weights(mol_p, basis_p, C_p, pseudo.S)[homo_p] < 0.5


@pytest.mark.parametrize("name", list(PINNED))
def test_pinned_cluster_regression(name):
    _, _, res = _run(name)
    assert res.converged
    e_ref, gap_ref = PINNED[name]
    assert res.total_energy == pytest.approx(e_ref, abs=1e-6)
    assert _gap_ev(res) == pytest.approx(gap_ref, abs=1e-3)
