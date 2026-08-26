"""Slater X-alpha exchange: energy density and potential.

This module is the single source of truth for the X-alpha coefficients.

Definition
----------
Slater's X-alpha exchange energy is

    E_x[rho] = -(9/8) alpha (3/pi)^(1/3) int rho(r)^(4/3) dr

so the energy *per electron* and the potential are

    eps_x(rho) = -(9/8) alpha (3 rho / pi)^(1/3)
    V_x(rho)   = d(rho eps_x)/d rho = (4/3) eps_x
               = -(3/2) alpha (3 rho / pi)^(1/3)
               = -3 alpha (3 rho / (8 pi))^(1/3)      <- the familiar form

The ``V = (4/3) eps`` identity is asserted in tests/test_xc.py, and the absolute
scale is checked against the uniform-electron-gas exchange energy.

Spin-polarised case
-------------------
For an unrestricted calculation the exchange functional is evaluated per spin
channel using the spin-scaling relation ``E_x[rho_a, rho_b] =
(E_x[2 rho_a] + E_x[2 rho_b]) / 2``, which gives

    V_x,sigma(rho_sigma) = -(3/2) alpha (3 * 2 rho_sigma / pi)^(1/3)

Using the *total* density in the potential while splitting the density between
channels -- as the original single-file version did -- makes the UKS energy and
potential mutually inconsistent.
"""

from __future__ import annotations

import numpy as np

from ..constants import PI, XALPHA_ENERGY_COEFF, XALPHA_POTENTIAL_COEFF

#: Densities below this are treated as zero (avoids rho^(1/3) noise).
RHO_FLOOR = 1e-30


def _cube_root_arg(rho: np.ndarray) -> np.ndarray:
    return (3.0 * np.maximum(rho, RHO_FLOOR) / PI) ** (1.0 / 3.0)


def energy_density(rho: np.ndarray, alpha: float) -> np.ndarray:
    """eps_x(rho): exchange energy *per electron* (Hartree)."""
    return -XALPHA_ENERGY_COEFF * alpha * _cube_root_arg(rho)


def potential(rho: np.ndarray, alpha: float) -> np.ndarray:
    """V_x(rho) = (4/3) eps_x(rho): the exchange potential (Hartree)."""
    return -XALPHA_POTENTIAL_COEFF * alpha * _cube_root_arg(rho)


def energy_density_spin(rho_sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Spin-resolved eps_x, from the spin-scaling relation."""
    return energy_density(2.0 * rho_sigma, alpha)


def potential_spin(rho_sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Spin-resolved V_x, consistent with :func:`energy_density_spin`."""
    return potential(2.0 * rho_sigma, alpha)


def exchange_energy(rho: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """E_x = int rho eps_x dV, closed shell."""
    return float(np.dot(weights, rho * energy_density(rho, alpha)))


def exchange_energy_spin(
    rho_a: np.ndarray, rho_b: np.ndarray, weights: np.ndarray, alpha: float
) -> float:
    """E_x = 1/2 (E_x[2 rho_a] + E_x[2 rho_b]), open shell."""
    ea = np.dot(weights, 2.0 * rho_a * energy_density(2.0 * rho_a, alpha))
    eb = np.dot(weights, 2.0 * rho_b * energy_density(2.0 * rho_b, alpha))
    return float(0.5 * (ea + eb))
