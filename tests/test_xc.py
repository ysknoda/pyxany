"""The X-alpha functional, checked against exact analytic references."""

from __future__ import annotations

import numpy as np

from dvxa.integrals import xc


def test_potential_is_the_functional_derivative():
    """V_x = d(rho eps_x)/d rho, verified numerically as well as by the 4/3 identity."""
    rho = np.logspace(-6, 2, 25)
    alpha = 0.7
    eps = xc.energy_density(rho, alpha)
    V = xc.potential(rho, alpha)
    assert np.allclose(V / eps, 4.0 / 3.0, rtol=1e-12)

    h = 1e-6 * rho
    numeric = ((rho + h) * xc.energy_density(rho + h, alpha)
               - (rho - h) * xc.energy_density(rho - h, alpha)) / (2 * h)
    assert np.abs((numeric - V) / V).max() < 1e-8


def test_alpha_two_thirds_reproduces_dirac_exchange():
    """The decisive absolute-scale check.

    Slater X-alpha at alpha = 2/3 *is* Dirac (LDA) exchange,
    ``e_x = -(3/4)(3/pi)^(1/3) rho^(4/3)``.  Getting this right fixes the
    coefficient with no adjustable freedom; a version of this code that was a
    factor of two off failed exactly here.
    """
    rho = np.array([0.05, 1.0, 2.0, 7.5, 40.0])
    dirac = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (4.0 / 3.0)
    slater = rho * xc.energy_density(rho, 2.0 / 3.0)
    assert np.allclose(slater, dirac, rtol=1e-14)


def test_potential_matches_the_textbook_form():
    """V_x = -3 alpha (3 rho / 8 pi)^(1/3)."""
    rho = np.array([0.1, 1.0, 12.0])
    alpha = 0.7
    textbook = -3.0 * alpha * (3.0 * rho / (8.0 * np.pi)) ** (1.0 / 3.0)
    assert np.allclose(xc.potential(rho, alpha), textbook, rtol=1e-13)


def test_spin_scaling_reduces_to_the_closed_shell_result():
    rho = np.array([0.3, 1.0, 5.0])
    w = np.ones(3)
    closed = xc.exchange_energy(rho, w, 0.7)
    open_shell = xc.exchange_energy_spin(rho / 2, rho / 2, w, 0.7)
    assert abs(closed - open_shell) < 1e-12


def test_exchange_energy_scales_as_rho_four_thirds():
    rho = np.array([1.0])
    w = np.ones(1)
    e1 = xc.exchange_energy(rho, w, 0.7)
    e8 = xc.exchange_energy(8.0 * rho, w, 0.7)
    assert abs(e8 / e1 - 8.0 ** (4.0 / 3.0)) < 1e-12
