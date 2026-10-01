"""Tier A validation: pyxany against PySCF with the same functional and basis.

Both codes run Slater X-alpha exchange with alpha = 0.7 and no correlation --
in PySCF that is ``1.05 * LDA_X`` (alpha / (2/3), since alpha = 2/3 is Dirac
exchange) -- on the *same* contracted Gaussian basis, translated from
pyxany's own tables so that not even the basis-set source can differ. With the
physics held identical, any disagreement is an implementation error or grid
error, never a limitation of X-alpha. See docs/adr/0003-validation-tiers.md.

PySCF is a development-only cross-check and deliberately not a declared
dependency: the module skips when it is not installed.

Tolerances were set from measurements (2026-10-01, PySCF 2.14.0, PySCF grid
level 9; ratio-damped Poisson solver; per-element radial sizes) at the test grid
n_r=50, n_a=18, poisson_l_max=8, pair 60/20:

    worst |dE|            1.17 mHa (CO)   -> tolerance 2 mHa
    worst MAE(eps_occ)    0.29 mHa (CO)   -> tolerance 0.5 mHa
    worst |dq| Mulliken   1e-4 e          -> tolerance 2e-3 e
    worst |d mu|          8e-4 D (LiH)    -> tolerance 5e-3 D

The energy difference is quadrature error on the pyxany side and vanishes
with the grid: CO -1.17 / -0.08 / -0.16 mHa, SiH4 +0.97 / +0.22 / +0.04 mHa,
H2S +0.91 / +0.20 / +0.07 mHa at n_r = 50 / 70 / 90. Below ~0.2 mHa it is at
the level of PySCF's own grid error, so ``test_energy_converges_to_pyscf``
asserts convergence to that level rather than strict monotonicity.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.basis.gto import GTO_KEYS
from pyxany.data.loaders import gto_polarisation_exponents, gto_table
from pyxany.properties import dipole_moment, mulliken
from pyxany.scf import run_scf

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto  # noqa: E402

pytestmark = pytest.mark.slow

ALPHA = 0.7
_L = {"s": 0, "p": 1, "d": 2}

TEST_GRID = {"n_r": 50, "n_a": 18, "poisson_l_max": 8, "pair_n_r": 60, "pair_n_a": 20}
TOL_ENERGY_HA = 2e-3
TOL_EPS_MAE_HA = 5e-4
TOL_CHARGE_E = 2e-3
TOL_DIPOLE_D = 5e-3

# Geometries in Angstrom. Tier A compares two codes on one geometry, so these
# need only be reasonable, not experimental. (name, geometry, basis, multiplicity)
CASES = [
    ("H2", [("H", 0, 0, 0), ("H", 0, 0, 0.741)], "631g", 1),
    ("LiH", [("Li", 0, 0, 0), ("H", 0, 0, 1.595)], "sto3g", 1),  # 6-31G table lacks Li
    (
        "CH4",
        [
            ("C", 0, 0, 0),
            ("H", 0.629, 0.629, 0.629),
            ("H", -0.629, -0.629, 0.629),
            ("H", -0.629, 0.629, -0.629),
            ("H", 0.629, -0.629, -0.629),
        ],
        "631g",
        1,
    ),
    (
        "NH3",
        [
            ("N", 0, 0, 0.1162),
            ("H", 0, 0.9377, -0.2711),
            ("H", 0.8121, -0.4689, -0.2711),
            ("H", -0.8121, -0.4689, -0.2711),
        ],
        "631g",
        1,
    ),
    (
        "H2O",
        [("O", 0, 0, 0.1173), ("H", 0, 0.7572, -0.4692), ("H", 0, -0.7572, -0.4692)],
        "631g",
        1,
    ),
    ("HF", [("F", 0, 0, 0), ("H", 0, 0, 0.917)], "631g", 1),
    ("N2", [("N", 0, 0, 0), ("N", 0, 0, 1.098)], "631g", 1),
    ("CO", [("C", 0, 0, 0), ("O", 0, 0, 1.128)], "631g", 1),
    (
        "SiH4",
        [
            ("Si", 0, 0, 0),
            ("H", 0.854, 0.854, 0.854),
            ("H", -0.854, -0.854, 0.854),
            ("H", -0.854, 0.854, -0.854),
            ("H", 0.854, -0.854, -0.854),
        ],
        "631g",
        1,
    ),
    (
        "H2S",
        [("S", 0, 0, 0.1030), ("H", 0, 0.9616, -0.8239), ("H", 0, -0.9616, -0.8239)],
        "631g",
        1,
    ),
    # d polarisation functions (real solid harmonics on both sides)
    (
        "H2O-631gd",
        [("O", 0, 0, 0.1173), ("H", 0, 0.7572, -0.4692), ("H", 0, -0.7572, -0.4692)],
        "631gd",
        1,
    ),
    # open shell: exercises the per-spin X-alpha scaling against UKS
    (
        "CH3",
        [("C", 0, 0, 0), ("H", 1.079, 0, 0), ("H", -0.5395, 0.9344, 0), ("H", -0.5395, -0.9344, 0)],
        "631g",
        2,
    ),
]
_BY_NAME = {c[0]: c for c in CASES}


def _pyscf_basis(elements, key):
    """pyxany's GTO table as a PySCF basis dict (6-31G sp shells split into s + p)."""
    table_name, want_pol = GTO_KEYS[key]
    table = gto_table(table_name)
    pol = gto_polarisation_exponents()
    out = {}
    for el in set(elements):
        shells = []
        for label, prims in table[el].items():
            kind = label.split("_")[0].lstrip("0123456789")
            if kind == "sp":
                shells.append([0] + [[a, cs] for a, cs, _ in prims])
                shells.append([1] + [[a, cp] for a, _, cp in prims])
            else:
                shells.append([_L[kind]] + [[a, c] for a, c in prims])
        if want_pol and el != "H" and el in pol:
            shells.append([2, [pol[el], 1.0]])
        out[el] = shells
    return out


def _run_pyscf(geom, key, mult):
    mol = gto.M(
        atom=[[s, (x, y, z)] for s, x, y, z in geom],
        unit="Angstrom",
        basis=_pyscf_basis([g[0] for g in geom], key),
        spin=mult - 1,
        verbose=0,
    )
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
    mf.xc = f"{ALPHA / (2.0 / 3.0):.12f}*LDA_X,"
    mf.grids.level = 9
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    eps = mf.mo_energy if mult == 1 else mf.mo_energy[0]
    return {
        "E": mf.e_tot,
        "eps": eps,
        "q": mf.mulliken_pop(verbose=0)[1],
        "mu": mf.dip_moment(verbose=0, unit="Debye"),
    }


def _run_pyxany(name, geom, key, mult, grid):
    mol = make_molecule(name, geom, multiplicity=mult)
    cfg = make_config(
        basis={"set": key},
        grid=grid,
        scf={"max_iter": 200, "e_tol": 1e-10, "d_tol": 1e-8, "alpha": ALPHA},
    )
    basis = build_basis(mol, cfg)
    r = run_scf(mol, basis, cfg, verbose=0)
    assert r.converged
    if mult == 1:
        eps, n_occ = r.eps, int((r.occupations > 0).sum())
    else:
        eps, n_occ = r.eps_a, int((r.occ_a > 0).sum())
    return {
        "E": r.total_energy,
        "eps": eps,
        "n_occ": n_occ,
        "q": mulliken(r.P, r.S, basis, mol)["charges"],
        "mu": dipole_moment(mol, r.grid, r.terms.density(r.P)),
    }


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_matches_pyscf(name):
    _, geom, key, mult = _BY_NAME[name]
    ref = _run_pyscf(geom, key, mult)
    got = _run_pyxany(name, geom, key, mult, TEST_GRID)
    n = got["n_occ"]

    assert abs(got["E"] - ref["E"]) < TOL_ENERGY_HA
    assert np.abs(got["eps"][:n] - ref["eps"][:n]).mean() < TOL_EPS_MAE_HA
    assert np.abs(got["q"] - ref["q"]).max() < TOL_CHARGE_E
    assert np.linalg.norm(got["mu"] - ref["mu"]) < TOL_DIPOLE_D


@pytest.mark.parametrize("name", ["CO", "SiH4", "H2S"])
def test_energy_converges_to_pyscf(name):
    """The remaining energy gap is quadrature error: it must shrink with the grid."""
    _, geom, key, mult = _BY_NAME[name]
    ref = _run_pyscf(geom, key, mult)["E"]
    grids = [
        TEST_GRID,
        {"n_r": 70, "n_a": 24, "poisson_l_max": 10, "pair_n_r": 80, "pair_n_a": 26},
        {"n_r": 90, "n_a": 30, "poisson_l_max": 10, "pair_n_r": 100, "pair_n_a": 30},
    ]
    gaps = [abs(_run_pyxany(name, geom, key, mult, g)["E"] - ref) for g in grids]
    assert gaps[2] < 5e-4
    assert gaps[2] < gaps[0]
