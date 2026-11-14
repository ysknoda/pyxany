"""XYZ and YAML input handling, including the strict-validation behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pyxany.config import Config, ConfigError
from pyxany.io.xyz import XYZError, append_frame, n_frames, read_xyz, write_xyz
from pyxany.io.yaml_input import load_config, load_molecule

CO_XYZ = """2
CO molecule
C   0.000000   0.000000   0.000000
O   0.000000   0.000000   1.128323
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_xyz_round_trip(tmp_path):
    src = write(tmp_path, "co.xyz", CO_XYZ)
    mol = read_xyz(src)
    out = tmp_path / "out.xyz"
    write_xyz(out, mol, "round trip")
    again = read_xyz(out)
    assert mol.symbols == again.symbols
    assert np.abs(mol.positions - again.positions).max() < 1e-9


def test_bohr_units_are_honored(tmp_path):
    src = write(tmp_path, "co.xyz", CO_XYZ)
    ang = read_xyz(src, units="angstrom")
    bohr = read_xyz(src, units="bohr")
    assert np.abs(bohr.positions).max() < np.abs(ang.positions).max()


def test_unknown_species_is_rejected(tmp_path):
    src = write(tmp_path, "bad.xyz", "2\nbad\nXx 0 0 0\nO 0 0 1\n")
    with pytest.raises(XYZError, match="neither a supported element"):
        read_xyz(src)


def test_declared_pseudo_species_is_accepted(tmp_path):
    src = write(tmp_path, "ps.xyz", "2\npseudo\nCd 0 0 0\nHCd 0 0 1.7\n")
    mol = read_xyz(src, species={"HCd": {"alpha": 1.5}})
    assert mol[1].is_pseudo
    assert mol[1].Z == pytest.approx(1.5)
    assert mol.n_electrons == pytest.approx(49.5)


def test_pseudo_species_round_trips(tmp_path):
    src = write(tmp_path, "ps.xyz", "2\npseudo\nCd 0 0 0\nHCd 0 0 1.7\n")
    mol = read_xyz(src, species={"HCd": {"alpha": 1.5}})
    out = tmp_path / "again.xyz"
    write_xyz(out, mol)
    again = read_xyz(out, species={"HCd": {"alpha": 1.5}})
    assert again.symbols == ["Cd", "HCd"]
    assert again[1].Z == pytest.approx(1.5)


def test_malformed_files_give_useful_errors(tmp_path):
    for text, pattern in (("two\nx\nC 0 0 0\n", "atom count"),
                          ("2\nc\nC 0 0\nO 0 0 1\n", "symbol x y z"),
                          ("2\nc\nC a b c\nO 0 0 1\n", "not numbers")):
        with pytest.raises(XYZError, match=pattern):
            read_xyz(write(tmp_path, "x.xyz", text))


def test_trajectory_frames(tmp_path):
    src = write(tmp_path, "traj.xyz", CO_XYZ)
    mol = read_xyz(src)
    append_frame(src, mol, "frame 2")
    append_frame(src, mol, "frame 3")
    assert n_frames(src) == 3
    assert read_xyz(src, frame=2).symbols == ["C", "O"]


def test_yaml_resolves_the_structure_path_relatively(tmp_path):
    write(tmp_path, "co.xyz", CO_XYZ)
    write(tmp_path, "co.yaml",
          "molecule:\n  name: co\n  structure: co.xyz\nbasis:\n  set: sz\n")
    cfg = load_config(tmp_path / "co.yaml")
    assert cfg.structure_path().exists()
    assert len(load_molecule(cfg)) == 2


def test_unknown_keys_are_errors_not_silent_defaults():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"molecule": {"structure": "x.xyz"}, "scf": {"alfa": 0.7}})
    with pytest.raises(ConfigError, match="unknown top-level section"):
        Config.from_dict({"molecul": {}})


def test_out_of_range_values_are_rejected():
    for section, values, pattern in (
        ("scf", {"alpha": 5.0}, "alpha"),
        ("grid", {"n_r": 2}, "n_r"),
        ("forces", {"method": "magic"}, "method"),
        ("grid", {"partition": "none"}, "overcounts"),
    ):
        with pytest.raises(ConfigError, match=pattern):
            Config.from_dict({"molecule": {"structure": "x.xyz"}, section: values})


def test_checkpoint_every_is_validated():
    """Now implemented (ADR 0004): positive values are accepted, negative rejected."""
    Config.from_dict({"molecule": {"structure": "x.xyz"}, "scf": {"checkpoint_every": 10}})
    Config.from_dict({"molecule": {"structure": "x.xyz"}, "scf": {"checkpoint_every": 0}})
    with pytest.raises(ConfigError, match="checkpoint_every"):
        Config.from_dict({"molecule": {"structure": "x.xyz"},
                          "scf": {"checkpoint_every": -1}})


def test_missing_structure_is_reported():
    with pytest.raises(ConfigError, match="structure"):
        Config.from_dict({"molecule": {"name": "x"}})


def test_dotted_overrides():
    cfg = Config.from_dict({"molecule": {"structure": "x.xyz"}})
    cfg.apply_overrides({"scf.alpha": "0.75", "grid.n_r": "40",
                         "forces.enabled": "false"})
    assert cfg.scf.alpha == 0.75
    assert cfg.grid.n_r == 40
    assert cfg.forces.enabled is False
    with pytest.raises(ConfigError, match="has no key"):
        cfg.apply_overrides({"scf.nonsense": "1"})


def test_set_sub_key_on_a_scalar_is_an_error_or_a_promotion():
    """--set A.B.C must never drop '.C' silently (ADR 0012, F13)."""
    from pyxany.config import Config, ConfigError

    def cfg(**scf):
        return Config.from_dict({"molecule": {"structure": "x.xyz"}, "scf": scf})

    c = cfg()
    c.apply_overrides({"scf.alpha.O": "0.73"})          # uniform alpha -> mapping
    assert c.scf.alpha == {"default": 0.7, "O": 0.73}
    assert c.scf.alpha_for("H") == 0.7 and c.scf.alpha_for("O") == 0.73

    c = cfg(alpha="alpha_hf")
    with pytest.raises(ConfigError, match="not a mapping"):
        c.apply_overrides({"scf.alpha.O": "0.73"})
    with pytest.raises(ConfigError, match="not a mapping"):
        cfg().apply_overrides({"grid.n_r.foo": "40"})
    with pytest.raises(ConfigError, match="has no key"):
        cfg().apply_overrides({"scf.alpha_for": "0.7"})   # a method, not a field

    c = cfg()
    c.apply_overrides({"grid.n_r_by_element.O": "40"})    # new key: YAML-typed
    assert c.grid.n_r_by_element == {"O": 40}


@pytest.mark.parametrize("section,key,value", [
    ("runtime", "mpi", "auto"), ("passivation", "scheme", "pseudo_hydrogen"),
    ("passivation", "basis", "derived"), ("properties", "bde", {"enabled": False}),
])
def test_removed_keys_are_errors_that_say_why(section, key, value):
    """Validated but never read (ADR 0013): now rejected with an explanation."""
    with pytest.raises(ConfigError, match="has been removed"):
        Config.from_dict({"molecule": {"structure": "x.xyz"}, section: {key: value}})


def test_esp_is_refused_until_implemented():
    base = {"molecule": {"structure": "x.xyz"}}
    Config.from_dict({**base, "properties": {"esp": {"enabled": False}}})
    with pytest.raises(ConfigError, match="not implemented"):
        Config.from_dict({**base, "properties": {"esp": {"enabled": True}}})


def test_runtime_numba_and_cli_flags_choose_the_backend(tmp_path):
    """runtime.numba and --numba/--no-numba had no effect (ADR 0013, F18)."""
    import argparse

    import pyxany.kernels as kernels
    from pyxany.cli._shared import prepare
    from pyxany.parallel import numba_enabled, set_numba_mode

    (tmp_path / "w.xyz").write_text("1\n\nH 0 0 0\n")
    yaml_text = "molecule: {structure: w.xyz, mult: 2}\nruntime: {numba: off}\n"
    (tmp_path / "w.yaml").write_text(yaml_text)

    def args(numba):
        return argparse.Namespace(input=str(tmp_path / "w.yaml"), overrides=None, basis=None,
                                  numba=numba, outdir=None, no_figure=True, timing=None,
                                  verbose=0, quiet=False)
    try:
        prepare(args(None))
        assert not numba_enabled() and kernels.backend_name() == "numpy"
        prepare(args(True))      # the flag wins over the file
        assert numba_enabled() == (kernels._load_numba() is not None)
        prepare(args(False))
        assert kernels.backend_name() == "numpy"
    finally:
        set_numba_mode("auto")


def test_runtime_verbosity_is_the_default_without_v_or_q():
    import argparse

    from pyxany.cli._shared import verbosity

    ns = argparse.Namespace(verbose=0, quiet=False)
    assert verbosity(ns, default=0) == 0
    assert verbosity(argparse.Namespace(verbose=2, quiet=False), default=0) == 3
    assert verbosity(argparse.Namespace(verbose=0, quiet=True), default=2) == 0
