"""The multi-centre Poisson solver against closed-form Coulomb solutions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from pyxany.grid import build_molecular_grid
from pyxany.integrals.coulomb import hartree_potential
from pyxany.molecule import Atom, Molecule

SIGMA = 0.8


def gaussian_density(points, centre, sigma=SIGMA):
    """Unit-charge spherical Gaussian; its potential is erf(r/sigma)/r."""
    r = np.linalg.norm(points - np.asarray(centre), axis=1)
    return np.exp(-((r / sigma) ** 2)) / (np.pi**1.5 * sigma**3)


def exact_potential(points, centres, sigma=SIGMA):
    v = np.zeros(len(points))
    for c in centres:
        r = np.maximum(np.linalg.norm(points - np.asarray(c), axis=1), 1e-30)
        v += erf(r / sigma) / r
    return v


def exact_hartree_energy(centres, sigma=SIGMA):
    """E_H = 1/2 sum_ij J_ij with J_self = sqrt(2/pi)/sigma, J_ij = erf(R/(s sqrt2))/R."""
    centres = np.asarray(centres, dtype=float)
    n = len(centres)
    total = n * np.sqrt(2.0 / np.pi) / sigma
    for i in range(n):
        for j in range(i + 1, n):
            R = float(np.linalg.norm(centres[i] - centres[j]))
            total += 2.0 * erf(R / (sigma * np.sqrt(2.0))) / R
    return 0.5 * total


def hydrogen_grid(centres, n_r=50, n_a=14):
    mol = Molecule([Atom.from_element("H", c) for c in centres])
    return mol, build_molecular_grid(mol, n_r, n_a)


def test_single_centre_matches_the_analytic_potential():
    centres = [[0.0, 0.0, 0.0]]
    mol, grid = hydrogen_grid(centres, n_r=60, n_a=16)
    rho = gaussian_density(grid.points, centres[0], sigma=1.0)
    V = hartree_potential(grid, rho, mol.positions, l_max=0)
    exact = exact_potential(grid.points, centres, sigma=1.0)
    # Measured 2.8e-6 at this grid.  The old bound of 5e-3 was 1800x looser and
    # was met even by the non-convergent cumsum quadrature this replaced.
    assert np.abs(V - exact).max() < 1e-5


def test_charge_is_conserved_on_the_grid():
    centres = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]
    mol, grid = hydrogen_grid(centres)
    rho = sum(gaussian_density(grid.points, c) for c in centres)
    assert float(np.dot(grid.weights, rho)) == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("centres", [
    [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]],
    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 2.6, 0.0]],
])
def test_hartree_energy_matches_the_closed_form(centres):
    mol, grid = hydrogen_grid(centres, n_r=60, n_a=16)
    rho = sum(gaussian_density(grid.points, c) for c in centres)
    V = hartree_potential(grid, rho, mol.positions, l_max=6)
    energy = 0.5 * float(np.dot(grid.weights, rho * V))
    # Measured 3.3e-5 relative at this grid; the old 2e-3 bound could not tell a
    # correct cumulative integral from partial sums of the Gauss-Legendre rule.
    assert energy == pytest.approx(exact_hartree_energy(centres), rel=1e-4)


def test_multipole_expansion_improves_the_potential():
    centres = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]
    mol, grid = hydrogen_grid(centres, n_r=60, n_a=16)
    rho = sum(gaussian_density(grid.points, c) for c in centres)
    exact = exact_potential(grid.points, centres)
    errors = []
    for l_max in (0, 4, 8):
        V = hartree_potential(grid, rho, mol.positions, l_max=l_max)
        errors.append(float(np.dot(grid.weights * rho, np.abs(V - exact))))
    assert errors[-1] < errors[0]


def test_potential_decays_as_one_over_r_outside_the_charge():
    """The analytic multipole tail must survive beyond the outermost radial node."""
    centres = [[0.0, 0.0, 0.0]]
    mol, grid = hydrogen_grid(centres, n_r=50, n_a=12)
    rho = gaussian_density(grid.points, centres[0], sigma=1.0)
    V = hartree_potential(grid, rho, mol.positions, l_max=2)
    r = np.linalg.norm(grid.points, axis=1)
    far = r > 0.9 * r.max()
    assert far.any()
    assert np.abs(V[far] * r[far] - 1.0).max() < 1e-3


def test_evaluation_on_a_subset_matches_the_full_grid():
    """The MPI path evaluates only local points; it must agree with the serial one."""
    centres = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]
    mol, grid = hydrogen_grid(centres, n_r=40, n_a=12)
    rho = sum(gaussian_density(grid.points, c) for c in centres)
    full = hartree_potential(grid, rho, mol.positions, l_max=4)
    subset = hartree_potential(grid, rho, mol.positions, l_max=4,
                               eval_points=grid.points[:100])
    assert np.abs(full[:100] - subset).max() < 1e-12


def test_radial_poisson_converges_at_the_spline_order():
    """The decisive check that the cumulative integrals are genuine integrals.

    ``V_l`` needs ``int_0^r`` and ``int_r^inf`` of the density.  Partial sums of
    the Gauss-Legendre radial rule are *not* those integrals -- a Gauss rule is
    exact only over its whole interval -- and converge as O(n_r^-1), which is
    what this solver used to do.  Integrating the cubic spline of the integrand
    gives O(n_r^-4).  Measured orders here are 3.88, 4.03, 4.01.

    An accuracy threshold alone cannot catch a regression to partial sums (the
    error at a single resolution can always be absorbed by a looser bound); the
    convergence *rate* can.
    """
    centres = [[0.0, 0.0, 0.0]]
    errors, sizes = [], (20, 30, 45, 70)
    for n_r in sizes:
        mol, grid = hydrogen_grid(centres, n_r=n_r, n_a=16)
        rho = gaussian_density(grid.points, centres[0], sigma=1.0)
        V = hartree_potential(grid, rho, mol.positions, l_max=0)
        exact = exact_potential(grid.points, centres, sigma=1.0)
        errors.append(float(np.abs(V - exact).max()))

    orders = [np.log(errors[i] / errors[i + 1]) / np.log(sizes[i + 1] / sizes[i])
              for i in range(len(sizes) - 1)]
    assert min(orders) > 3.0, f"convergence order dropped to {orders}"
    assert errors[-1] < 1e-5


@pytest.mark.parametrize("l_max", [0, 2, 4, 6, 8, 10])
def test_potential_stays_physical_at_every_allowed_l_max(l_max):
    """No multipole order may produce a spike near the nucleus.

    Both radial integrands carry a large negative power of r there
    (``r^-(l+1)`` for the inner one, ``r^(1-l)`` for the outer), so round-off
    and projection noise get amplified enormously unless the exact ``r^l``
    asymptotics are imposed.  Before that guard existed, water at l_max = 12
    reached |V| = 3.2e11 against a physical maximum of ~28, and even l_max = 6
    reached 1.4e4.
    """
    centres = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]
    mol, grid = hydrogen_grid(centres, n_r=30, n_a=12)
    rho = sum(gaussian_density(grid.points, c) for c in centres)
    V = hartree_potential(grid, rho, mol.positions, l_max=l_max)
    assert np.all(np.isfinite(V))
    # The Hartree potential of two unit charges cannot exceed a few tens
    # anywhere on this grid; a near-nucleus spike shows up as orders of
    # magnitude, not a few percent.
    assert np.abs(V).max() < 1e2, f"l_max={l_max}: max|V| = {np.abs(V).max():.3e}"


def test_hartree_energy_is_converged_by_the_default_l_max():
    """l_max = 6 (the default) must already be converged, not merely stable."""
    centres = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]
    mol, grid = hydrogen_grid(centres, n_r=60, n_a=16)
    rho = sum(gaussian_density(grid.points, c) for c in centres)
    energies = {}
    for l_max in (6, 8, 10):
        V = hartree_potential(grid, rho, mol.positions, l_max=l_max)
        energies[l_max] = 0.5 * float(np.dot(grid.weights, rho * V))
    assert abs(energies[6] - energies[10]) < 1e-4, energies
