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
