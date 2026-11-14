"""The pseudo-hydrogen basis function is derived, not fitted -- check the limits."""

from __future__ import annotations

import numpy as np
import pytest

from pyxany.basis.atomic_solver import solve_atom
from pyxany.basis.pseudo_h import _raw_zeta, describe, pseudo_hydrogen_zeta

VASP_ALPHAS = [1 / 4, 1 / 3, 5 / 12, 1 / 2, 7 / 12, 2 / 3, 3 / 4,
               1.0, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 7 / 4]


@pytest.mark.parametrize("Z", [0.5, 0.75, 1.0, 1.5, 2.0])
def test_non_interacting_limit_is_exactly_hydrogenic(Z):
    """With the electron-electron terms off, eps = -Z^2/2 and R = 2 Z^1.5 e^-Zr."""
    sol = solve_atom(Z=Z, occupations={(1, 0): 1e-12}, alpha=1e-12,
                     n_points=2500, r_max=120.0)
    orb = sol.orbitals[(1, 0)]
    assert orb.eigenvalue == pytest.approx(-Z * Z / 2.0, rel=1e-6)
    exact = 2.0 * Z**1.5 * np.exp(-Z * orb.r)
    assert np.abs(orb.radial - exact).max() < 1e-4


def test_calibrated_exponent_reproduces_the_tabulated_hydrogen_value():
    """alpha = 1 must give the basis set's own hydrogen exponent, by construction."""
    from pyxany.data.loaders import sto_table

    assert pseudo_hydrogen_zeta(1.0) == pytest.approx(sto_table("sz")["H"][0][3],
                                                      rel=1e-9)


@pytest.mark.parametrize("alpha", VASP_ALPHAS)
def test_every_vasp_charge_yields_a_usable_exponent(alpha):
    info = describe(alpha)
    assert info["converged"]
    assert 0.05 < info["zeta"] < 5.0
    assert info["eigenvalue_hartree"] < 0.0
    assert info["electrons"] == pytest.approx(alpha, rel=1e-4)


def test_exponent_increases_monotonically_with_charge():
    """A more highly charged terminator must be more compact, not less."""
    zetas = [pseudo_hydrogen_zeta(a) for a in VASP_ALPHAS]
    assert all(b > a for a, b in zip(zetas, zetas[1:], strict=False))
    radii = [describe(a)["mean_radius_bohr"] for a in VASP_ALPHAS]
    assert all(b < a for a, b in zip(radii, radii[1:], strict=False))


def test_raw_exponent_approaches_the_charge_in_the_weak_screening_limit():
    """As the electron count falls the screening vanishes and zeta -> alpha."""
    ratios = []
    for alpha in (1.0, 0.5, 0.25):
        ratios.append(_raw_zeta(alpha) / alpha)
    assert ratios[-1] > ratios[0]


def test_exponent_is_insensitive_to_the_xc_parameter():
    """A +/-20 % change in alpha_xc must not move the exponent qualitatively."""
    base = pseudo_hydrogen_zeta(0.5, 0.7)
    for xc_alpha in (0.56, 0.84):
        assert abs(pseudo_hydrogen_zeta(0.5, xc_alpha) / base - 1.0) < 0.15


@pytest.mark.parametrize("basis_set", ["sz", "dz", "dzp", "631g"])
def test_unit_charge_pseudo_hydrogen_is_exactly_hydrogen_in_every_basis(basis_set):
    """alpha = 1 must reproduce real hydrogen bit for bit, not only in SZ.

    Before the terminator received the basis set's full hydrogen function list
    it carried one SZ-calibrated 1s, and SiH4/DZ gave a different energy and an
    inner Si charge 0.4 e away from the real-hydrogen value.
    """
    from conftest import make_config
    from pyxany.basis import build_basis
    from pyxany.molecule import Atom, Molecule
    from pyxany.scf import run_scf

    geom = [("Si", 0, 0, 0), ("H", 1.614, 1.614, 1.614), ("H", -1.614, -1.614, 1.614),
            ("H", -1.614, 1.614, -1.614), ("H", 1.614, -1.614, -1.614)]   # Bohr
    cfg = make_config(basis={"set": basis_set})
    energies = []
    for pseudo in (False, True):
        atoms = []
        for s, x, y, z in geom:
            a = Atom.from_element(s, np.array([x, y, z], float))
            if pseudo and s == "H":
                a = Atom(symbol="HSi", element="H", Z=1.0, pos=a.pos, is_pseudo=True)
            atoms.append(a)
        mol = Molecule(atoms, name="sih4")
        energies.append(run_scf(mol, build_basis(mol, cfg), cfg, verbose=0).total_energy)
    assert energies[1] == pytest.approx(energies[0], abs=1e-12)


def test_fractional_terminator_scales_every_hydrogen_function_by_one_ratio():
    from conftest import make_config
    from pyxany.basis import build_basis
    from pyxany.basis.pseudo_h import length_scale_ratio, pseudo_hydrogen_zeta
    from pyxany.molecule import Atom, Molecule

    cfg = make_config(basis={"set": "dz"})
    h = Molecule([Atom.from_element("H", np.zeros(3))], name="h", multiplicity=2)
    ps = Molecule([Atom(symbol="HSe", element="H", Z=0.5, pos=np.zeros(3), is_pseudo=True)],
                  name="hse", multiplicity=1)
    ratio = length_scale_ratio(pseudo_hydrogen_zeta(0.5))
    z_h = [f.zeta for f in build_basis(h, cfg)]
    z_ps = [f.zeta for f in build_basis(ps, cfg)]
    assert len(z_ps) == len(z_h) == 2
    assert np.allclose(z_ps, np.array(z_h) * ratio, rtol=1e-14)
    assert 0.5 < ratio < 0.7       # a half-charge terminator is more diffuse


def test_describe_uses_the_requested_atom_grid():
    """describe() solved the atom on a fixed 2000-point grid (ADR 0012 P0)."""
    from pyxany.basis.pseudo_h import describe, pseudo_hydrogen_zeta

    d = describe(0.75, n_points=1200)
    assert d["zeta"] == pseudo_hydrogen_zeta(0.75, n_points=1200)
