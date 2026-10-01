"""Real harmonics: orthonormality, gradients, and backend agreement."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyxany.basis.harmonics import (
    MAX_L,
    lm_pairs,
    real_sph_harm,
    solid_harmonic,
    solid_harmonic_grad,
)
from pyxany.basis.solid_harmonics import real_spherical_harmonics
from pyxany.integrals.coulomb import lm_index_pairs, real_sph_harmonics


def sphere_quadrature(n_theta=60, n_phi=120):
    ct, wt = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    C, P = np.meshgrid(ct, phi, indexing="ij")
    Wt, Wp = np.meshgrid(wt, np.full(n_phi, 2 * np.pi / n_phi), indexing="ij")
    st = np.sqrt(np.maximum(1 - C.ravel() ** 2, 0))
    return (st * np.cos(P.ravel()), st * np.sin(P.ravel()), C.ravel(),
            (Wt * Wp).ravel())


def test_real_harmonics_are_orthonormal():
    x, y, z, w = sphere_quadrature()
    r = np.ones_like(x)
    keys = [(l, m) for l in range(MAX_L + 1) for m in lm_pairs(l)]
    vals = {k: real_sph_harm(k[0], k[1], x, y, z, r) for k in keys}
    for a, b in itertools.product(keys, keys):
        overlap = float(np.dot(w, vals[a] * vals[b]))
        expected = 1.0 if a == b else 0.0
        assert abs(overlap - expected) < 1e-12, f"{a} vs {b}"


def test_solid_harmonic_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(300, 3))
    h = 1e-5
    for l in range(MAX_L + 1):
        for m in lm_pairs(l):
            gx, gy, gz = solid_harmonic_grad(l, m, *pts.T)
            analytic = np.column_stack([gx, gy, gz])
            numeric = np.empty_like(analytic)
            for ax in range(3):
                up, dn = pts.copy(), pts.copy()
                up[:, ax] += h
                dn[:, ax] -= h
                numeric[:, ax] = (solid_harmonic(l, m, *up.T)
                                  - solid_harmonic(l, m, *dn.T)) / (2 * h)
            assert np.abs(analytic - numeric).max() < 1e-6, f"(l={l}, m={m})"


@pytest.mark.parametrize("l_max", [2, 4, 6, 8])
def test_recursion_matches_the_reference_implementation(l_max):
    """The fast Cartesian recursion must equal the scipy-based version."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=(500, 3))
    x, y, z = v.T
    r = np.linalg.norm(v, axis=1)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    fast = real_spherical_harmonics(l_max, x, y, z, r)
    reference = real_sph_harmonics(l_max, theta, phi)
    assert np.abs(fast - reference).max() < 1e-12


def test_recursion_agrees_with_the_l_le_3_table():
    """The two harmonic implementations must not drift apart."""
    rng = np.random.default_rng(2)
    v = rng.normal(size=(400, 3))
    x, y, z = v.T
    r = np.linalg.norm(v, axis=1)
    fast = real_spherical_harmonics(3, x, y, z, r)
    for l in range(4):
        for m in range(-l, l + 1):
            col = lm_index_pairs(3).index((l, m))
            table = real_sph_harm(l, m, x, y, z, r)
            assert np.abs(fast[:, col] - table).max() < 1e-12, f"(l={l}, m={m})"
