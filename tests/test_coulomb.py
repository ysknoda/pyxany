"""The multi-centre Poisson solver against closed-form Coulomb solutions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from dvxa.constants import ANG_TO_BOHR
from dvxa.grid import build_molecular_grid
from dvxa.integrals.coulomb import hartree_potential
from dvxa.molecule import Atom, Molecule

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
    assert np.abs(V - exact).max() < 5e-3


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
    assert energy == pytest.approx(exact_hartree_energy(centres), rel=2e-3)


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
