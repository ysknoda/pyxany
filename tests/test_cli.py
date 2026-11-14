"""Command-line behavior: exit codes, overrides, and machine-readable output."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyxany.cli import EXIT_INPUT, EXIT_OK, build_parser, main

CO_XYZ = """2
CO
C   0.000000   0.000000   0.000000
O   0.000000   0.000000   1.128323
"""

CO_YAML = """\
molecule:
  name: co
  structure: co.xyz
basis:
  set: sz
grid:
  n_r: 20
  n_a: 8
  poisson_l_max: 2
scf:
  max_iter: 40
forces:
  enabled: false
output:
  dir: {outdir}
  prefix: co
  figure: false
  timing: false
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "co.xyz").write_text(CO_XYZ)
    (tmp_path / "co.yaml").write_text(CO_YAML.format(outdir=tmp_path / "results"))
    return tmp_path


@pytest.mark.parametrize("command", ["run", "opt", "forces", "scan", "passivate",
                                     "gridcheck", "validate", "init", "basis", "info"])
def test_every_subcommand_has_help(command):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([command, "--help"])
    assert exc.value.code == 0


def test_info_runs():
    assert main(["info"]) == EXIT_OK


def test_basis_listing_and_element_report(capsys):
    assert main(["basis"]) == EXIT_OK
    assert "COVERAGE" in capsys.readouterr().out
    assert main(["basis", "dz", "--element", "Se"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Se" in out and "valence electrons 6" in out


def test_unknown_element_is_an_input_error():
    assert main(["basis", "dz", "--element", "Xx"]) == EXIT_INPUT


def test_validate_accepts_a_good_input(project, capsys):
    assert main(["validate", str(project / "co.yaml")]) == EXIT_OK
    assert "input is valid" in capsys.readouterr().out


def test_validate_rejects_a_typo(project, capsys):
    (project / "bad.yaml").write_text(
        "molecule:\n  name: co\n  structure: co.xyz\nscf:\n  alfa: 0.7\n")
    assert main(["validate", str(project / "bad.yaml")]) == EXIT_INPUT
    assert "unknown key" in capsys.readouterr().err


def test_missing_file_is_an_input_error(tmp_path):
    assert main(["run", str(tmp_path / "nope.yaml")]) == EXIT_INPUT


def test_dry_run_reports_resources_without_computing(project, capsys):
    assert main(["run", str(project / "co.yaml"), "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "RESOURCE ESTIMATE" in out and "grid points" in out
    assert not (project / "results").exists()


def test_run_produces_output_files_and_valid_json(project, capsys):
    code = main(["run", str(project / "co.yaml"), "--json",
                 "--set", "output.json=true"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out[capsys.readouterr().out.find("{"):]
                         if False else _last_json(project))
    assert payload["scf"]["converged"]
    assert payload["energies_hartree"]["E_total"] < 0
    results = project / "results"
    assert (results / "co_results.txt").exists()
    assert (results / "co_final.xyz").exists()


def _last_json(project):
    return (project / "results" / "co_results.json").read_text()


def test_set_override_changes_the_calculation(project, capsys):
    main(["validate", str(project / "co.yaml"), "--set", "scf.alpha=0.9",
          "--set", "grid.n_r=33"])
    out = capsys.readouterr().out
    assert "alpha: 0.9" in out
    assert "n_r: 33" in out


def test_bad_override_syntax_is_reported(project, capsys):
    assert main(["run", str(project / "co.yaml"), "--set", "nonsense"]) == EXIT_INPUT
    assert "SECTION.KEY" in capsys.readouterr().err


def test_init_writes_a_runnable_pair(tmp_path, capsys):
    assert main(["init", "demo", "-o", str(tmp_path)]) == EXIT_OK
    assert (tmp_path / "demo.yaml").exists()
    assert (tmp_path / "demo.xyz").exists()
    assert main(["validate", str(tmp_path / "demo.yaml")]) == EXIT_OK


def test_init_refuses_to_overwrite(tmp_path):
    main(["init", "demo", "-o", str(tmp_path)])
    assert main(["init", "demo", "-o", str(tmp_path)]) == 1


def test_passivate_writes_structure_and_yaml(tmp_path, capsys):
    (tmp_path / "si.xyz").write_text(
        "5\nSi5\nSi 0 0 0\nSi 1.35775 1.35775 1.35775\nSi 1.35775 -1.35775 -1.35775\n"
        "Si -1.35775 1.35775 -1.35775\nSi -1.35775 -1.35775 1.35775\n")
    code = main(["passivate", str(tmp_path / "si.xyz"), "--coordination", "4",
                 "-o", str(tmp_path / "si_h.xyz"),
                 "--emit-yaml", str(tmp_path / "si_h.yaml")])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "PASSIVATION" in out and "closed shell" in out
    assert (tmp_path / "si_h.xyz").exists()
    assert main(["validate", str(tmp_path / "si_h.yaml")]) == EXIT_OK


def test_gridcheck_reports_convergence(project, capsys):
    assert main(["gridcheck", str(project / "co.yaml"),
                 "--levels", "15x8,25x10"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "GRID QUALITY" in out and "int rho dV" in out


def test_shell_completion():
    assert main(["--print-completion", "bash"]) == EXIT_OK

def test_module_entry_point_matches_the_console_script():
    """``python -m pyxany`` must work and propagate the same exit codes.

    On a cluster the console script's bin directory is often off PATH, so
    ``python -m pyxany`` (and ``mpirun ... python -m pyxany``) is the reliable
    invocation; it must not drift from the ``pyxany`` entry point.
    """
    import subprocess
    import sys

    ok = subprocess.run([sys.executable, "-m", "pyxany", "--version"],
                        capture_output=True, text=True)
    assert ok.returncode == EXIT_OK, ok.stderr
    assert "pyxany" in ok.stdout

    missing = subprocess.run([sys.executable, "-m", "pyxany", "validate", "no_such_file.yaml"],
                             capture_output=True, text=True)
    assert missing.returncode == EXIT_INPUT


def test_no_figure_flag_overrides_the_input(project):
    pytest.importorskip("matplotlib")
    yaml = (project / "co.yaml").read_text().replace("figure: false", "figure: true")
    (project / "co.yaml").write_text(yaml)
    assert main(["run", str(project / "co.yaml"), "--no-figure", "-q"]) == EXIT_OK
    assert not list((project / "results").glob("*.png"))


def test_run_without_matplotlib_skips_the_figure(project, monkeypatch, capsys):
    """A missing matplotlib must skip the figures, not fail a converged run.

    pyxany.plotting imports matplotlib inside its functions, so the old guard
    (importing pyxany.plotting) passed and the run ended with exit code 3 after
    the SCF had converged -- found by installing the bare wheel.
    """
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib", None)  # import now raises ImportError
    yaml = (project / "co.yaml").read_text().replace("figure: false", "figure: true")
    (project / "co.yaml").write_text(yaml)
    assert main(["run", str(project / "co.yaml")]) == EXIT_OK
    assert "matplotlib is not installed" in capsys.readouterr().out
    assert not list((project / "results").glob("*.png"))


def test_input_errors_exit_1(tmp_path):
    """Missing structure, unknown symbol, bad --coord and wrong types exited 3."""
    from pyxany.cli import EXIT_INPUT

    (tmp_path / "a.yaml").write_text("molecule: {structure: missing.xyz}\n")
    assert main(["validate", str(tmp_path / "a.yaml")]) == EXIT_INPUT
    (tmp_path / "x.xyz").write_text("1\n\nXx 0 0 0\n")
    (tmp_path / "b.yaml").write_text("molecule: {structure: x.xyz}\n")
    assert main(["validate", str(tmp_path / "b.yaml")]) == EXIT_INPUT
    (tmp_path / "c.yaml").write_text("molecule: {structure: x.xyz}\nscf: {smearing: abc}\n")
    assert main(["validate", str(tmp_path / "c.yaml")]) == EXIT_INPUT


def test_truncated_xyz_is_an_input_error(tmp_path):
    from pyxany.io.xyz import XYZError, read_xyz

    (tmp_path / "t.xyz").write_text("3\ncomment\nO 0 0 0\nH 0 0 1\n")   # one atom short
    with pytest.raises(XYZError, match="ends early"):
        read_xyz(tmp_path / "t.xyz")


def test_scan_rejects_bad_angles_and_bends_a_linear_molecule():
    from pyxany.cli.scan import _apply, _check_range
    from pyxany.config import ConfigError
    from pyxany.molecule import Atom, Molecule

    with pytest.raises(ConfigError):
        _check_range("angle", [170.0, 190.0])
    co2 = Molecule([Atom.from_element(s, np.array([0, 0, z])) for s, z in
                    (("O", -2.2), ("C", 0.0), ("O", 2.2))])          # along z
    _apply(co2, "angle", [0, 1, 2], 150.0)
    p = co2.positions
    v1, v2 = p[0] - p[1], p[2] - p[1]
    ang = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
    assert ang == pytest.approx(150.0, abs=1e-6)


def test_basis_key_is_honored(capsys):
    assert main(["basis", "631g", "--element", "Se"]) == EXIT_INPUT
    assert main(["basis", "nonsense"]) == EXIT_INPUT
    assert main(["basis", "dz", "--element", "Se"]) == EXIT_OK


def test_log_records_the_error(tmp_path):
    log = tmp_path / "run.log"
    assert main(["run", str(tmp_path / "missing.yaml"), "--log", str(log)]) == EXIT_INPUT
    assert "input file not found" in log.read_text()


def test_embed_checks_the_element_on_each_removed_site(tmp_path):
    """MgO with the origin shifted by a/2 put Mg on O sites and suggested -26."""
    import shutil

    from pyxany.cli import EXIT_INPUT

    examples = __import__("pathlib").Path(__file__).resolve().parent.parent / "examples"
    for n in ("mgo_crystal.yaml", "mg19o6.xyz"):
        shutil.copy(examples / n, tmp_path / n)
    spec = tmp_path / "mgo_crystal.yaml"
    text = spec.read_text()
    import yaml as _yaml
    d = _yaml.safe_load(text)
    a = d["lattice"][0][0]
    d["origin"] = [a / 2, 0.0, 0.0]
    spec.write_text(_yaml.safe_dump(d))
    assert main(["embed", str(spec), "-o", str(tmp_path / "q.xyz")]) == EXIT_INPUT
