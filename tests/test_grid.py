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


def sto_density(points, center, zeta=1.24):
    bf = STO(1, 0, 0, zeta, 0)
    r = np.linalg.norm(points - np.asarray(center), axis=1)
    return (bf.radial(r) / math.sqrt(4.0 * math.pi)) ** 2


def test_angular_weights_sum_to_four_pi():
    for n in (6, 10, 14, 20):
        _dirs, w = spherical_product(n)
        assert abs(w.sum() - 4.0 * np.pi) < 1e-12


@pytest.mark.parametrize("n_atoms", [1, 2, 3, 4, 5])
def test_single_function_normalization_is_atom_count_independent(n_atoms):
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
        centers = rng.normal(scale=3.0, size=(n, 3))
        radii = rng.uniform(0.6, 3.0, size=n)
        pts = rng.normal(scale=6.0, size=(1500, 3))
        w = becke_weights(pts, centers, radii)
        assert np.abs(w.sum(axis=1) - 1.0).max() < 1e-12
        assert w.min() >= 0.0


def test_weight_at_own_nucleus_is_one():
    centers = np.array([[0.0, 0, 0], [0, 0, 4.0]])
    w = becke_weights(centers, centers, np.array([1.7, 1.32]))
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
    """Atoms carry different radial sizes; every block must still map to its radii."""
    mol = Molecule([Atom.from_element("Ga", [0, 0, 0]),
                    Atom.from_element("H", [0, 0, 3.0]),
                    Atom.from_element("O", [0, 2.5, 0])])
    grid = build_molecular_grid(mol, 20, 8)
    assert len(set(grid.n_r_atom)) > 1, "test needs atoms of different radial size"
    assert sum(grid.n_r_atom) * grid.n_angular == len(grid)
    for ai in range(len(mol)):
        block = grid.points[grid.atom_block(ai)].reshape(
            int(grid.n_r_atom[ai]), grid.n_angular, 3)
        r = np.linalg.norm(block - mol[ai].pos, axis=2)
        assert np.allclose(r, grid.radii[ai][:, None])
        assert np.all(grid.owner[grid.atom_block(ai)] == ai)
    assert np.allclose(grid.weights, grid.raw_weights * grid.cell_weights)


def test_radial_sizes_follow_the_measured_table():
    """n_r is the second-row size; heavier rows get the measured multiple."""
    from pyxany.grid.molecular import radial_sizes

    mol = Molecule([Atom.from_element(s, [0, 0, 4.0 * i])
                    for i, s in enumerate(("O", "Si", "Ga", "Cd"))])
    assert list(radial_sizes(mol, 30)) == [29, 33, 47, 60]
    uniform = build_molecular_grid(Molecule([mol[0], mol[2]]), 30, 6,
                                   radial_by_element=False)
    assert list(uniform.n_r_atom) == [30, 30]


def test_own_cell_weight_gradient_matches_finite_differences():
    """d w_A(r_k) / d R_B with points fixed, contracted with arbitrary coefficients."""
    from pyxany.grid.becke import own_cell_weight_gradient

    rng = np.random.default_rng(3)
    C = np.array([[0, 0, 0], [0.3, 0.1, 2.1], [-1.5, 1.0, 0.4]], float)
    radii = np.array([0.8, 1.2, 0.5])
    pts = C[rng.integers(0, 3, 400)] + rng.normal(size=(400, 3)) * 0.9
    owner = np.argmin(np.linalg.norm(pts[:, None] - C[None], axis=2), axis=1)
    coeff = rng.normal(size=400)
    M = own_cell_weight_gradient(pts, owner, coeff, C, radii)
    h = 1e-6
    fd = np.zeros_like(M)
    for B in range(3):
        for x in range(3):
            Cp, Cm = C.copy(), C.copy()
            Cp[B, x] += h
            Cm[B, x] -= h
            wp = becke_weights(pts, Cp, radii)[np.arange(400), owner]
            wm = becke_weights(pts, Cm, radii)[np.arange(400), owner]
            d = coeff * (wp - wm) / (2 * h)
            for A in range(3):
                fd[A, B, x] = d[owner == A].sum()
    assert np.abs(M - fd).max() < 1e-7 * np.abs(fd).max()
