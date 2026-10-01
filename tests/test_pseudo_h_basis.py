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
    assert all(b > a for a, b in zip(zetas, zetas[1:]))
    radii = [describe(a)["mean_radius_bohr"] for a in VASP_ALPHAS]
    assert all(b < a for a, b in zip(radii, radii[1:]))


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
