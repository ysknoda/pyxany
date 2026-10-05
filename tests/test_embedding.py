"""Point-charge embedding (ADR 0010)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.cli import EXIT_OK, main
from pyxany.cli.embed import evjen_charges, remove_cluster_sites
from pyxany.config import ConfigError
from pyxany.embedding import PointCharges, read_charges
from pyxany.scf import run_scf

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

ROCK_SALT = [
    ["Mg", 0, 0, 0, 2],
    ["Mg", 0.5, 0.5, 0, 2],
    ["Mg", 0.5, 0, 0.5, 2],
    ["Mg", 0, 0.5, 0.5, 2],
    ["O", 0.5, 0, 0, -2],
    ["O", 0, 0.5, 0, -2],
    ["O", 0, 0, 0.5, -2],
    ["O", 0.5, 0.5, 0.5, -2],
]

WATER = [("O", 0.05, 0.02, 0.1173), ("H", 0.03, 0.7572, -0.4692), ("H", -0.02, -0.7572, -0.4492)]


@pytest.fixture
def charge_file(tmp_path):
    path = tmp_path / "q.xyz"
    path.write_text(
        "# three charges\nQ  3.0  0.5  0.8  +1.0\n-2.5 1.0 -1.5 -0.7\nQ  0.4 -3.0  2.2  +0.4\n"
    )
    return path


# -- the potential ------------------------------------------------------------


def test_charge_file_formats_and_errors(charge_file, tmp_path):
    pos, q = read_charges(charge_file)
    assert q.tolist() == [1.0, -0.7, 0.4]
    assert pos[0] == pytest.approx(np.array([3.0, 0.5, 0.8]) / 0.529177210903)
    bad = tmp_path / "bad.xyz"
    bad.write_text("Q 1 2 3\n")
    with pytest.raises(ValueError, match="bad.xyz:1"):
        read_charges(bad)


def test_gaussian_charge_potential_limits_and_gradient():
    pc = PointCharges(np.array([[0.0, 0.0, 0.0]]), np.array([2.0]), sigma=0.5)
    # finite at the charge: -q * 2 / (sqrt(pi) sigma)
    assert pc.potential(np.zeros((1, 3)))[0] == pytest.approx(-2.0 * 2 / (np.sqrt(np.pi) * 0.5))
    # a point charge far away
    far = np.array([[0.0, 0.0, 10.0]])
    assert pc.potential(far)[0] == pytest.approx(-2.0 / 10.0, rel=1e-12)
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=1.0, size=(50, 3))
    h = 1e-5
    fd = np.stack(
        [(pc.potential(pts + h * e) - pc.potential(pts - h * e)) / (2 * h) for e in np.eye(3)],
        axis=1,
    )
    assert np.abs(pc.gradient(pts) - fd).max() < 1e-7


# -- the Evjen block -------------------------------------------------------------


def test_evjen_block_is_neutral_dipole_free_and_gives_the_madelung_constant():
    unit = [[s, x, y, z, q / 2] for s, x, y, z, q in ROCK_SALT]  # +-1 ions
    pos, q, _ = evjen_charges(np.eye(3), unit, cells=4)
    assert q.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs((q[:, None] * pos).sum(axis=0)).max() < 1e-10
    d = np.linalg.norm(pos, axis=1)
    keep = d > 1e-9
    madelung = -np.sum(q[keep] / d[keep]) * 0.5  # nearest-neighbor distance a/2
    assert madelung == pytest.approx(1.747565, abs=2e-5)  # rock salt


def test_cluster_sites_are_removed_and_their_charge_reported():
    from pyxany.io.xyz import read_xyz

    pos, q, lab = evjen_charges(4.21 * np.eye(3), ROCK_SALT, cells=2)
    cluster = read_xyz(EXAMPLES / "mg19o6.xyz")
    pos2, q2, _, removed = remove_cluster_sites(pos, q, lab, cluster)
    assert len(q2) == len(q) - len(cluster)
    assert removed == pytest.approx(19 * 2 - 6 * 2)
    assert q2.sum() + removed == pytest.approx(0.0, abs=1e-10)
    off_lattice = make_molecule("x", [("Mg", 0.3, 0, 0)])
    with pytest.raises(ValueError, match="not on a lattice site"):
        remove_cluster_sites(pos, q, lab, off_lattice)


def test_embed_command_writes_a_readable_file(tmp_path, capsys):
    out = tmp_path / "charges.xyz"
    assert main(["embed", str(EXAMPLES / "mgo_crystal.yaml"), "-o", str(out)]) == EXIT_OK
    pos, q = read_charges(out)
    assert q.sum() == pytest.approx(-26.0, abs=1e-8)
    assert "molecule.charge = +26" in capsys.readouterr().out


# -- in the SCF --------------------------------------------------------------------


def _cfg(charge_file, **grid):
    return make_config(
        basis={"set": "dz"},
        grid={"n_r": 40, "n_a": 14, "poisson_l_max": "auto", **grid},
        scf={"e_tol": 1e-11, "d_tol": 1e-9, "max_iter": 100},
        embedding={"charges": str(charge_file)},
    )


def test_embedding_energy_terms_are_reported_and_included(charge_file):
    mol = make_molecule("w", WATER)
    cfg = _cfg(charge_file)
    res = run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)
    e = res.energies
    assert e["E_embedding_nuclei"] == pytest.approx(
        PointCharges(*read_charges(charge_file), 0.5).nuclear_energy(mol)
    )
    assert e["E_embedding_electrons"] < 0 < e["E_embedding_nuclei"]
    assert 0 <= e["embedding_spill_electrons"] < 0.01


def test_far_charges_barely_touch_a_neutral_molecule(tmp_path):
    path = tmp_path / "far.xyz"
    path.write_text("Q 60 0 0 +1\nQ -60 0 0 -1\n")
    mol = make_molecule("w", WATER)
    e_far = run_scf(mol, build_basis(mol, _cfg(path)), _cfg(path), verbose=0).total_energy
    plain = make_config(
        basis={"set": "dz"},
        grid={"n_r": 40, "n_a": 14, "poisson_l_max": "auto"},
        scf={"e_tol": 1e-11, "d_tol": 1e-9, "max_iter": 100},
    )
    e0 = run_scf(mol, build_basis(mol, plain), plain, verbose=0).total_energy
    assert abs(e_far - e0) < 1e-4


@pytest.mark.slow
def test_embedded_forces_match_finite_differences(charge_file):
    """No projection: fixed charges make the net force physical."""
    from pyxany.forces.analytic import compute_forces

    cfg = _cfg(charge_file)

    def run(geom):
        mol = make_molecule("w", geom)
        return run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)

    F = compute_forces(run(WATER), cfg)
    h = 1e-3 * 0.529177210903  # 1e-3 Bohr, in Angstrom
    fd = np.zeros_like(F)
    for a in range(3):
        for x in range(3):
            plus = [list(t) for t in WATER]
            minus = [list(t) for t in WATER]
            plus[a][1 + x] += h
            minus[a][1 + x] -= h
            fd[a, x] = -(run(plus).total_energy - run(minus).total_energy) / 2e-3
    assert np.abs(F - fd).max() < 2e-4  # 9e-5 measured; 1e-4 without charges
    assert np.linalg.norm(F.sum(axis=0)) > 1e-3  # the charges push the molecule


def test_checkpoint_fingerprint_depends_on_the_charges(charge_file, tmp_path):
    from pyxany.scf.checkpoint import system_fingerprint

    mol = make_molecule("w", WATER)
    cfg_a = _cfg(charge_file)
    other = tmp_path / "other.xyz"
    other.write_text("Q 3 0 0 +1\n")
    cfg_b = _cfg(other)
    basis = build_basis(mol, cfg_a)
    assert system_fingerprint(mol, basis, cfg_a, False) != system_fingerprint(
        mol, basis, cfg_b, False
    )


@pytest.mark.parametrize("bad", [{"sigma": 0}, {"units": "nm"}, {"charges": 3}])
def test_bad_embedding_settings_are_rejected(bad):
    with pytest.raises(ConfigError, match="embedding"):
        make_config(embedding=bad)
