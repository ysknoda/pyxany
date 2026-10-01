"""Shared fixtures: small molecules and a fast default configuration."""

from __future__ import annotations

import numpy as np
import pytest

from pyxany.config import Config
from pyxany.constants import ANG_TO_BOHR
from pyxany.molecule import Atom, Molecule


def make_molecule(name, geometry_angstrom, charge=0, multiplicity=1):
    atoms = [Atom.from_element(s, np.array([x, y, z]) * ANG_TO_BOHR)
             for s, x, y, z in geometry_angstrom]
    return Molecule(atoms, charge=charge, multiplicity=multiplicity, name=name)


def make_config(**overrides):
    """A deliberately small grid so the whole suite stays quick."""
    raw = {
        "molecule": {"name": "test", "structure": "test.xyz"},
        "basis": {"set": "sz"},
        "grid": {"n_r": 25, "n_a": 10, "poisson_l_max": 4},
        "scf": {"max_iter": 60, "e_tol": 1e-7, "d_tol": 1e-6},
        "forces": {"enabled": False},
        "output": {"figure": False, "json": False, "timing": False},
    }
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    return Config.from_dict(raw)


@pytest.fixture
def h2():
    return make_molecule("h2", [("H", 0, 0, 0), ("H", 0, 0, 0.741)])


@pytest.fixture
def h2o():
    return make_molecule("h2o", [("O", 0, 0, 0.1173),
                                 ("H", 0, 0.7572, -0.4692),
                                 ("H", 0, -0.7572, -0.4692)])


@pytest.fixture
def ch3():
    return make_molecule("ch3", [("C", 0, 0, 0), ("H", 1.079, 0, 0),
                                 ("H", -0.5395, 0.9345, 0),
                                 ("H", -0.5395, -0.9345, 0)], multiplicity=2)


@pytest.fixture
def neon():
    return make_molecule("ne", [("Ne", 0, 0, 0)])
