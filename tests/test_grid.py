"""The integration grid must reproduce a known integral exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyxany.basis.types import STO
from pyxany.grid import becke_weights, build_molecular_grid, spherical_product
from pyxany.molecule import Atom, Molecule


def chain(n, spacing=2.6):
    """n hydrogens in a non-collinear arrangement, so cells genuinely overlap."""
    return Molecule([Atom.from_element("H", [spacing * i, 0.9 * (i % 3), 0.7 * (i % 2)])
                     for i in range(n)])


def sto_density(points, centre, zeta=1.24):
    bf = STO(1, 0, 0, zeta, 0)
    r = np.linalg.norm(points - np.asarray(centre), axis=1)
    return (bf.radial(r) / math.sqrt(4.0 * math.pi)) ** 2


def test_angular_weights_sum_to_four_pi():
    for n in (6, 10, 14, 20):
        _dirs, w = spherical_product(n)
        assert abs(w.sum() - 4.0 * np.pi) < 1e-12


@pytest.mark.parametrize("n_atoms", [1, 2, 3, 4, 5])
def test_single_function_normalisation_is_atom_count_independent(n_atoms):
    """int |chi|^2 dV = 1 no matter how many atomic sub-grids overlap.

    Without Becke partitioning this integral grows roughly in proportion to the
    number of atoms; that regression is what this test locks out.
    """
    mol = chain(n_atoms)
    grid = build_molecular_grid(mol, 25, 10)
    value = float(np.dot(grid.weights, sto_density(grid.points, mol[0].pos)))
    assert abs(value - 1.0) < 1e-4


def test_unpartitioned_grid_overcounts():
    """The failure mode the Becke weights exist to prevent, made explicit."""
    mol = chain(4)
    grid = build_molecular_grid(mol, 25, 10, partition="none")
    value = float(np.dot(grid.weights, sto_density(grid.points, mol[0].pos)))
    assert value > 3.0


def test_becke_weights_are_a_partition_of_unity():
    rng = np.random.default_rng(0)
    for n in (2, 3, 6):
        centres = rng.normal(scale=3.0, size=(n, 3))
        radii = rng.uniform(0.6, 3.0, size=n)
        pts = rng.normal(scale=6.0, size=(1500, 3))
        w = becke_weights(pts, centres, radii)
        assert np.abs(w.sum(axis=1) - 1.0).max() < 1e-12
        assert w.min() >= 0.0


def test_weight_at_own_nucleus_is_one():
    centres = np.array([[0.0, 0, 0], [0, 0, 4.0]])
    w = becke_weights(centres, centres, np.array([1.7, 1.32]))
    assert abs(w[0, 0] - 1.0) < 1e-12
    assert abs(w[1, 1] - 1.0) < 1e-12


def test_refining_the_grid_improves_the_integral():
    mol = chain(3)
    errors = []
    for n_r, n_a in ((15, 8), (25, 10), (40, 14)):
        grid = build_molecular_grid(mol, n_r, n_a)
        value = float(np.dot(grid.weights, sto_density(grid.points, mol[0].pos)))
        errors.append(abs(value - 1.0))
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-5


def test_grid_block_structure_matches_radii():
    mol = Molecule([Atom.from_element("C", [0, 0, 0]),
                    Atom.from_element("O", [0, 0, 2.13])])
    grid = build_molecular_grid(mol, 20, 8)
    assert grid.block_size * grid.n_atoms == len(grid)
    block = grid.points[grid.atom_block(1)].reshape(grid.n_r, grid.n_angular, 3)
    r = np.linalg.norm(block - mol[1].pos, axis=2)
    assert np.allclose(r, grid.radii[1][:, None])
    assert np.allclose(grid.weights, grid.raw_weights * grid.cell_weights)
