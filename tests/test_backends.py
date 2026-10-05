"""The numba and numpy code paths must produce identical numbers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.kernels import backend_name, numpy_impl, poisson_radial
from pyxany.scf import run_scf

SCRIPT = """
import json, sys
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.scf import run_scf
from pyxany.kernels import backend_name
mol = make_molecule("h2o", [("O",0,0,0.1173),("H",0,0.7572,-0.4692),("H",0,-0.7572,-0.4692)])
cfg = make_config(basis={{"set":"dz"}}, scf={{"max_iter":60,"e_tol":1e-9,"d_tol":1e-8}})
res = run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)
print(json.dumps({{"backend": backend_name(), "E": res.total_energy}}))
"""


def test_poisson_backends_agree():
    from pyxany.grid.radial import log_squared_map

    r, w = log_squared_map(400, 1.0)
    rho = np.exp(-(r**2)) / np.pi**1.5
    assert np.abs(poisson_radial(r, w, rho)
                  - numpy_impl.poisson_radial(r, w, rho)).max() < 1e-14


@pytest.mark.slow
def test_total_energy_is_backend_independent(tmp_path):
    """Whether numba is active must not change a single digit of the energy."""
    root = Path(__file__).resolve().parent.parent
    script = tmp_path / "probe.py"
    script.write_text(SCRIPT.format(src=str(root), tests=str(root / "tests")))

    results = {}
    for disable in ("0", "1"):
        env = dict(os.environ, PYXANY_DISABLE_NUMBA=disable)
        out = subprocess.run([sys.executable, str(script)], capture_output=True,
                             text=True, env=env, check=True)
        payload = out.stdout.strip().splitlines()[-1]
        import json

        data = json.loads(payload)
        results[data["backend"]] = data["E"]

    assert len(results) == 2, f"both backends must be exercised, got {results}"
    energies = list(results.values())
    assert abs(energies[0] - energies[1]) < 1e-10, results


def test_backend_name_is_reported():
    assert backend_name() in ("numba", "numpy")


def test_chi_matrix_is_backend_independent(h2o):
    """Compare the active backend against the generic path on the same points."""
    from pyxany.basis.evaluate import chi_matrix
    from pyxany.basis.harmonics import solid_harmonic

    cfg = make_config(basis={"set": "dzp"})
    basis = build_basis(h2o, cfg)
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=2.0, size=(500, 3))
    fast = chi_matrix(basis, h2o.positions, pts)

    slow = np.empty_like(fast)
    for mu, f in enumerate(basis):
        d = pts - h2o.positions[f.atom_idx]
        r = np.linalg.norm(d, axis=1)
        slow[:, mu] = f.g(r) * solid_harmonic(f.l, f.m, d[:, 0], d[:, 1], d[:, 2])
    assert np.abs(fast - slow).max() < 1e-13


def _numba_or_skip():
    from pyxany import kernels

    if kernels._numba() is None:
        pytest.skip("numba not installed")
    return kernels


def test_compiled_becke_weights_match_numpy():
    kernels = _numba_or_skip()
    from pyxany.grid.becke import becke_weights, size_adjustment

    rng = np.random.default_rng(1)
    C = rng.normal(size=(6, 3)) * 2.0
    radii = rng.uniform(0.5, 2.0, size=6)
    pts = C[rng.integers(0, 6, 3000)] + rng.normal(size=(3000, 3))
    owner = rng.integers(0, 6, 3000)
    ref = becke_weights(pts, C, radii)[np.arange(3000), owner]
    got = kernels.becke_own_cell(pts, owner, C, size_adjustment(radii), 3)
    assert np.abs(got - ref).max() < 1e-13


def test_compiled_becke_gradient_matches_numpy():
    kernels = _numba_or_skip()
    import pyxany.grid.becke as becke

    rng = np.random.default_rng(2)
    C = rng.normal(size=(5, 3)) * 2.0
    radii = rng.uniform(0.5, 2.0, size=5)
    pts = C[rng.integers(0, 5, 2000)] + rng.normal(size=(2000, 3))
    owner = np.argmin(np.linalg.norm(pts[:, None] - C[None], axis=2), axis=1)
    coeff = rng.normal(size=2000)
    got = kernels.own_cell_weight_gradient(pts, owner, coeff, C, becke.size_adjustment(radii), 3)
    saved = kernels.own_cell_weight_gradient
    try:
        kernels.own_cell_weight_gradient = lambda *a, **k: None   # force the numpy path
        ref = becke.own_cell_weight_gradient(pts, owner, coeff, C, radii)
    finally:
        kernels.own_cell_weight_gradient = saved
    assert np.abs(got - ref).max() < 1e-11 * max(1.0, np.abs(ref).max())


def test_compiled_hartree_summation_matches_numpy(h2o):
    kernels = _numba_or_skip()
    from pyxany.grid import build_molecular_grid
    from pyxany.integrals import coulomb

    grid = build_molecular_grid(h2o, 25, 10)
    rho = sum(float(a.Z) * np.exp(-np.sum((grid.points - a.pos) ** 2, axis=1)) for a in h2o)
    fast = coulomb.hartree_potential(grid, rho, h2o.positions, l_max=9)
    saved = kernels.hartree_evaluate
    try:
        kernels.hartree_evaluate = lambda *a, **k: None
        ref = coulomb.hartree_potential(grid, rho, h2o.positions, l_max=9)
    finally:
        kernels.hartree_evaluate = saved
    assert np.abs(fast - ref).max() < 1e-12 * np.abs(ref).max()


def test_compiled_chi_gradient_matches_numpy(h2o, monkeypatch):
    """Includes points on the nuclei, where rhat is undefined."""
    _numba_or_skip()
    import pyxany.basis.evaluate as ev

    basis = build_basis(h2o, make_config(basis={"set": "dzp"}))
    rng = np.random.default_rng(1)
    pts = np.vstack([h2o.positions, rng.normal(scale=2.0, size=(500, 3))])
    fast = ev.chi_grad_matrix(basis, h2o.positions, pts)
    monkeypatch.setattr(ev, "chi_grad_block_kernel", lambda: None)
    slow = ev.chi_grad_matrix(basis, h2o.positions, pts)
    assert np.abs(fast - slow).max() < 1e-12 * max(1.0, np.abs(slow).max())


def test_compiled_hartree_gradient_matches_finite_differences(h2o):
    """Off the first and last radial knots, where the piecewise radial
    functions have a kink and only one-sided derivatives exist."""
    _numba_or_skip()
    from pyxany.integrals.coulomb import hartree_potential

    cfg = make_config(basis={"set": "dz"})
    res = run_scf(h2o, build_basis(h2o, cfg), cfg, verbose=0)
    grid, rho, lmax = res.grid, res.terms.density(res.P), cfg.grid.l_max
    rng = np.random.default_rng(2)
    pts = np.vstack([grid.points[1::97] + 1e-3, rng.normal(scale=6.0, size=(50, 3))])
    g = hartree_potential(grid, rho, h2o.positions, l_max=lmax, eval_points=pts, gradient=True)
    h = 1e-5
    fd = np.empty_like(g)
    for x in range(3):
        e = np.zeros(3)
        e[x] = h
        vp = hartree_potential(grid, rho, h2o.positions, l_max=lmax, eval_points=pts + e)
        vm = hartree_potential(grid, rho, h2o.positions, l_max=lmax, eval_points=pts - e)
        fd[:, x] = (vp - vm) / (2 * h)
    assert np.abs(g - fd).max() < 1e-6 * max(1.0, np.abs(fd).max())


def test_analytic_hartree_gradient_leaves_forces_unchanged(h2o, monkeypatch):
    _numba_or_skip()
    import pyxany.forces.analytic as fa
    from pyxany.integrals.coulomb import hartree_potential

    cfg = make_config(basis={"set": "dz"})
    res = run_scf(h2o, build_basis(h2o, cfg), cfg, verbose=0)
    F_exact = fa.compute_forces(res, cfg, project=False)

    def no_gradient(*args, gradient=False, **kwargs):
        return None if gradient else hartree_potential(*args, **kwargs)

    monkeypatch.setattr(fa, "hartree_potential", no_gradient)
    F_fd = fa.compute_forces(res, cfg, project=False)
    assert np.abs(F_exact - F_fd).max() < 1e-7


def test_compiled_basis_hessian_matches_finite_differences(h2o):
    _numba_or_skip()
    from pyxany.basis.evaluate import chi_grad_matrix, chi_hess_matrix

    basis = build_basis(h2o, make_config(basis={"set": "dzp"}))
    rng = np.random.default_rng(3)
    pts = rng.normal(scale=1.5, size=(400, 3))
    H = chi_hess_matrix(basis, h2o.positions, pts)
    h = 1e-5
    fd = np.empty_like(H)
    for x in range(3):
        e = np.zeros(3)
        e[x] = h
        fd[..., x] = (chi_grad_matrix(basis, h2o.positions, pts + e)
                      - chi_grad_matrix(basis, h2o.positions, pts - e)) / (2 * h)
    assert np.abs(H - fd).max() < 1e-6 * np.abs(fd).max()
    assert np.abs(H - H.transpose(0, 1, 3, 2)).max() < 1e-10 * np.abs(H).max()


def test_exact_two_center_gradient_matches_finite_differences(h2o):
    """Basis Hessians, moving sub-grid and Becke-weight terms together."""
    _numba_or_skip()
    from pyxany.integrals.twocenter import overlap_kinetic_2c, overlap_kinetic_2c_gradient

    basis = build_basis(h2o, make_config(basis={"set": "dzp"}))
    fa = [f for f in basis if f.atom_idx == 0]
    fb = [f for f in basis if f.atom_idx == 1]
    A, B = h2o.positions[0], h2o.positions[1]
    rA, rB = h2o[0].grid_radius(), h2o[1].grid_radius()
    rng = np.random.default_rng(4)
    Pb = rng.normal(size=(len(fa), len(fb)))
    Wb = rng.normal(size=(len(fa), len(fb)))
    exact = overlap_kinetic_2c_gradient(fa, fb, A, B, rA, rB, Pb, Wb)
    fd = np.empty(3)
    h = 1e-4
    for x in range(3):
        e = np.zeros(3)
        e[x] = h
        Sp, Tp = overlap_kinetic_2c(fa, fb, A, B + e, rA, rB)
        Sm, Tm = overlap_kinetic_2c(fa, fb, A, B - e, rA, rB)
        fd[x] = (np.sum(Pb * (Tp - Tm)) - np.sum(Wb * (Sp - Sm))) / (2 * h)
    assert np.abs(exact - fd).max() < 1e-7 * max(1.0, np.abs(fd).max())
