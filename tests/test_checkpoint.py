"""SCF checkpoint and restart (ADR 0004)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.scf import checkpoint as ckpt
from pyxany.scf import run_scf

WATER = [("O", 0, 0, 0.1173), ("H", 0, 0.7572, -0.4692), ("H", 0, -0.7572, -0.4692)]
METHYL = [("C", 0, 0, 0), ("H", 1.079, 0, 0), ("H", -0.5395, 0.9345, 0),
          ("H", -0.5395, -0.9345, 0)]


def _scf(geom, path=None, restart=None, max_iter=60, every=0, mult=1, shift=0.0):
    g = [(s, x, y, z + (shift if i == 1 else 0.0)) for i, (s, x, y, z) in enumerate(geom)]
    mol = make_molecule("m", g, multiplicity=mult)
    cfg = make_config(scf={"max_iter": max_iter, "e_tol": 1e-9, "d_tol": 1e-8,
                           "checkpoint_every": every})
    return run_scf(mol, build_basis(mol, cfg), cfg, verbose=0, restart=restart,
                   checkpoint_path=path)


def test_write_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "s.npz"
    P = np.arange(9.0).reshape(3, 3)
    ckpt.write(path, P=P, iteration=7, system="abc", geometry="xyz", energy=-1.5,
               converged=False)
    assert not list(tmp_path.glob("*.tmp")), "temporary file left behind"
    back = ckpt.read(path)
    assert np.array_equal(back.P, P) and back.iteration == 7 and back.energy == -1.5
    assert back.P_a is None and back.system == "abc" and back.geometry == "xyz"


@pytest.mark.parametrize("geom, mult", [(WATER, 1), (METHYL, 2)])
def test_interrupted_run_resumes_to_the_same_answer(tmp_path, geom, mult):
    reference = _scf(geom, mult=mult)
    assert reference.converged
    path = tmp_path / "s.npz"
    cut = _scf(geom, path=path, max_iter=4, every=2, mult=mult)       # "killed" early
    assert not cut.converged and ckpt.read(path).iteration == 4
    resumed = _scf(geom, path=path, restart=path, mult=mult)
    assert resumed.converged
    assert resumed.total_energy == pytest.approx(reference.total_energy, abs=1e-8)
    assert ckpt.read(path).converged


def test_other_geometry_uses_the_density_as_a_guess(tmp_path):
    path = tmp_path / "s.npz"
    _scf(WATER, path=path)
    fresh = _scf(WATER, shift=0.02)
    seeded = _scf(WATER, shift=0.02, restart=path)
    assert seeded.total_energy == pytest.approx(fresh.total_energy, abs=1e-8)
    assert seeded.n_iterations < fresh.n_iterations


def test_a_different_system_is_refused(tmp_path):
    path = tmp_path / "s.npz"
    _scf(WATER, path=path)
    with pytest.raises(ValueError, match="different system"):
        _scf(METHYL, mult=2, restart=path)


def _ts_scf(path=None, restart=None, max_iter=100):
    mol = make_molecule("m", WATER)
    cfg = make_config(scf={"max_iter": max_iter, "e_tol": 1e-9, "d_tol": 1e-8},
                      transition_state={"remove": "homo"})
    return run_scf(mol, build_basis(mol, cfg), cfg, verbose=0, restart=restart,
                   checkpoint_path=path)


def test_transition_state_resume_keeps_the_half_hole(tmp_path):
    """The resumed density was rescaled to the integer count, putting the
    removed half electron back: 2.26 Ha off and 122 more iterations (ADR 0012 P0)."""
    path = tmp_path / "ts.npz"
    first = _ts_scf(path=path)
    assert first.converged
    again = _ts_scf(restart=path)
    assert again.converged and again.n_iterations <= 6
    assert again.total_energy == pytest.approx(first.total_energy, abs=1e-7)


def test_restart_at_max_iter_is_an_input_error(tmp_path):
    """Resuming a run that stopped at max_iter with the same max_iter used to
    run zero iterations and crash in the report (ADR 0012 P0)."""
    from pyxany.config import ConfigError

    path = tmp_path / "s.npz"
    cut = _scf(WATER, path=path, max_iter=4)
    assert not cut.converged
    with pytest.raises(ConfigError, match="raise scf.max_iter above 4"):
        _scf(WATER, restart=path, max_iter=4)
