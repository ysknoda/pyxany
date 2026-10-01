"""Parallel results must be identical to serial ones.

The MPI assertions run under ``mpirun``; the helper script is exercised
serially by default so the decomposition logic is still covered in CI.

    mpirun -n 4 python -m pytest tests/test_mpi.py -q -m mpi
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.parallel import PARALLEL, SIZE, local_slice, round_robin
from pyxany.scf import run_scf

PROBE = """
import json, sys
sys.path.insert(0, {src!r}); sys.path.insert(0, {tests!r})
from conftest import make_config, make_molecule
from pyxany.basis import build_basis
from pyxany.forces import compute_forces
from pyxany.parallel import SIZE, is_root
from pyxany.scf import run_scf
mol = make_molecule("h2o", [("O",0,0,0.1173),("H",0,0.7572,-0.4692),("H",0,-0.7572,-0.4692)])
cfg = make_config(basis={{"set":"sz"}}, grid={{"n_r":25,"n_a":10,"poisson_l_max":4}},
                  scf={{"max_iter":60,"e_tol":1e-9,"d_tol":1e-8}}, forces={{"enabled":True}})
res = run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)
f = compute_forces(res, cfg)
if is_root():
    print("RESULT " + json.dumps({{"ranks": SIZE, "E": res.total_energy,
                                  "n": res.energies["n_electrons_integrated"],
                                  "f": [list(map(float, row)) for row in f]}}))
"""


def test_local_slice_covers_everything_exactly():
    total = 1000
    if not PARALLEL:
        assert local_slice(total) == slice(0, total)
        return
    covered = set()
    for rank in range(SIZE):
        s = local_slice(total, rank=rank, size=SIZE)
        covered |= set(range(s.start, s.stop))
    assert covered == set(range(total))


def test_round_robin_partitions_without_overlap():
    items = list(range(37))
    if not PARALLEL:
        assert round_robin(items) == items
        return
    shares = [round_robin(items, rank=r, size=SIZE) for r in range(SIZE)]
    flat = [x for s in shares for x in s]
    assert sorted(flat) == items


@pytest.mark.mpi
def test_energy_and_forces_match_serial(tmp_path):
    """Run the same calculation on 1 and N ranks and compare."""
    root = Path(__file__).resolve().parent.parent
    script = tmp_path / "probe.py"
    script.write_text(PROBE.format(src=str(root), tests=str(root / "tests")))

    import json

    def run(n):
        cmd = ([sys.executable, str(script)] if n == 1
               else ["mpirun", "-n", str(n), sys.executable, str(script)])
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT "))
        return json.loads(line[len("RESULT "):])

    serial = run(1)
    parallel = run(2)
    assert parallel["ranks"] == 2
    assert abs(serial["E"] - parallel["E"]) < 1e-8
    assert abs(serial["n"] - parallel["n"]) < 1e-8
    import numpy as np

    assert np.abs(np.array(serial["f"]) - np.array(parallel["f"])).max() < 1e-8


def test_serial_run_is_reproducible(h2o):
    """Guards against any hidden dependence on iteration order."""
    cfg = make_config(basis={"set": "sz"}, scf={"max_iter": 60})
    energies = [run_scf(h2o.copy(), build_basis(h2o, cfg), cfg, verbose=0).total_energy
                for _ in range(2)]
    assert abs(energies[0] - energies[1]) < 1e-12
