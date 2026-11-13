"""One- and two-center integrals against closed-form references."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyxany.basis.types import GTO, STO
from pyxany.grid.radial import log_squared_map
from pyxany.integrals.onecenter import kinetic_1c, overlap_1c
from pyxany.integrals.twocenter import overlap_kinetic_2c


def test_sto_normalization():
    r, w = log_squared_map(500, 1.0)
    for bf in (STO(1, 0, 0, 1.24, 0), STO(2, 0, 0, 1.6083, 0),
               STO(2, 1, 0, 1.5679, 0), STO(3, 2, 0, 1.20, 0),
               STO(4, 3, 0, 0.80, 0)):
        assert abs(float(np.dot(w, bf.radial(r) ** 2 * r * r)) - 1.0) < 1e-9
        assert abs(overlap_1c(bf, bf) - 1.0) < 1e-10


@pytest.mark.parametrize("zeta", [0.8, 1.24, 3.5, 7.66])
def test_hydrogenic_kinetic_energy_is_exact(zeta):
    """T = zeta^2 / 2 for a 1s Slater function -- a closed-form identity."""
    bf = STO(1, 0, 0, zeta, 0)
    assert abs(kinetic_1c(bf, bf) - zeta * zeta / 2.0) < 1e-10


def test_closed_form_and_quadrature_one_center_agree():
    """The analytic Slater formulas and the generic radial quadrature must match."""
    r, w = log_squared_map(500, 1.0)
    pairs = [(STO(1, 0, 0, 1.24, 0), STO(1, 0, 0, 1.24, 0)),
             (STO(1, 0, 0, 5.67, 0), STO(2, 0, 0, 1.61, 0)),
             (STO(2, 1, 0, 1.57, 0), STO(2, 1, 0, 1.57, 0)),
             (STO(3, 2, 0, 1.20, 0), STO(3, 2, 0, 2.40, 0))]
    for a, b in pairs:
        s_quad = float(np.dot(w, a.radial(r) * b.radial(r) * r * r))
        t_quad = -0.5 * float(np.dot(w, a.radial(r) * b.laplacian_radial(r)
                                     * r ** b.l * r * r))
        assert abs(overlap_1c(a, b) - s_quad) < 1e-8
        assert abs(kinetic_1c(a, b) - t_quad) < 1e-8


def test_gto_contraction_uses_normalized_primitives():
    """Reading the tables with the wrong convention shows up as norm != 1."""
    shells = {
        "H 1s": (0, [(3.4252509, 0.1543290), (0.6239137, 0.5353281),
                     (0.1688554, 0.4446345)]),
        "C 1s": (0, [(71.6168370, 0.1543290), (13.0450960, 0.5353281),
                     (3.5305122, 0.4446345)]),
        "C 2p": (1, [(7.8682724, 0.0689991), (1.8812885, 0.3164240),
                     (0.5442493, 0.7443083)]),
    }
    for name, (l, prims) in shells.items():
        g = GTO.contracted(l, 0, prims, 0)
        assert abs(g.norm - 1.0) < 1e-5, f"{name}: residual norm {g.norm}"
        assert abs(overlap_1c(g, g) - 1.0) < 1e-10


def test_sto3g_hydrogen_kinetic_energy():
    """STO-3G H 1s has T = 0.7600 Ha; the underlying zeta = 1.24 gives 0.7688."""
    g = GTO.contracted(0, 0, [(3.4252509, 0.1543290), (0.6239137, 0.5353281),
                              (0.1688554, 0.4446345)], 0)
    assert abs(kinetic_1c(g, g) - 0.7600) < 2e-4


@pytest.mark.parametrize("R", [0.8, 1.4, 2.0, 3.0, 5.0])
def test_two_center_overlap_matches_the_closed_form(R):
    """S(R) = exp(-x)(1 + x + x^2/3), x = zeta R, for two identical 1s Slaters."""
    zeta = 1.24
    S, _T = overlap_kinetic_2c([STO(1, 0, 0, zeta, 0)], [STO(1, 0, 0, zeta, 1)],
                               [0, 0, 0], [0, 0, R], 0.661, 0.661, n_r=40, n_a=14)
    x = zeta * R
    exact = math.exp(-x) * (1.0 + x + x * x / 3.0)
    assert abs(S[0, 0] - exact) < 1e-7


def test_gradient_form_kinetic_energy_reduces_to_one_center():
    """1/2 int grad chi . grad chi equals the closed-form T at zero separation."""
    zeta = 1.24
    _S, T = overlap_kinetic_2c([STO(1, 0, 0, zeta, 0)], [STO(1, 0, 0, zeta, 1)],
                               [0, 0, 0], [0, 0, 1e-6], 0.661, 0.661, n_r=60, n_a=16)
    assert abs(T[0, 0] - zeta * zeta / 2.0) < 1e-6


def test_overlap_kinetic_gradient_matches_finite_differences():
    """d/dR [tr(P T) - tr(W S)] for the S and T the SCF uses, incl. d functions."""
    from conftest import make_config, make_molecule
    from pyxany.basis import build_basis
    from pyxany.integrals.overlap import build_overlap_kinetic, overlap_kinetic_gradient

    geom = [("C", 0, 0, 0), ("O", 0.1, -0.05, 1.128), ("H", -0.9, 0.3, -0.4)]
    cfg = make_config(basis={"set": "dzp"})
    rng = np.random.default_rng(0)
    mol = make_molecule("m", geom)
    n = len(build_basis(mol, cfg))
    P = rng.normal(size=(n, n))
    P += P.T
    W = rng.normal(size=(n, n))
    W += W.T
    grad = overlap_kinetic_gradient(mol, build_basis(mol, cfg), P, W)
    h = 1e-4
    fd = np.zeros_like(grad)
    for A in range(3):
        for x in range(3):
            vals = []
            for sign in (1, -1):
                m = make_molecule("m", geom)
                pos = m.positions
                pos[A, x] += sign * h
                m.positions = pos
                S, T = build_overlap_kinetic(m, build_basis(m, cfg))
                vals.append(np.sum(P * T) - np.sum(W * S))
            fd[A, x] = (vals[0] - vals[1]) / (2 * h)
    assert np.abs(grad - fd).max() < 1e-8 * np.abs(fd).max()
