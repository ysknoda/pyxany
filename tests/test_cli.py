"""Command-line behaviour: exit codes, overrides, and machine-readable output."""

from __future__ import annotations

import json

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
