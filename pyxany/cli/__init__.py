"""``pyxany`` command-line interface.

Subcommands
-----------
run         single-point SCF (energy, forces, properties)
opt         geometry optimization
forces      forces only, optionally comparing analytic against finite difference
scan        scan a bond length or angle
passivate   add pseudo-hydrogen (or real hydrogen) terminators to a cluster
embed       point charges of the surrounding crystal (Evjen block)
gridcheck   grid-quality diagnostic
validate    parse and validate an input file without running anything
init        write a template YAML + XYZ pair
basis       basis-set coverage information
info        runtime environment

Exit codes (so batch scripts can branch on the outcome)
------------------------------------------------------
0 success            2 SCF did not converge
1 input error        3 runtime error          4 optimization did not converge
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__

EXIT_OK = 0
EXIT_INPUT = 1
EXIT_SCF = 2
EXIT_RUNTIME = 3
EXIT_OPT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyxany",
        description="DV-Xalpha all-electron molecular orbital calculations",
        epilog="Run 'pyxany <command> --help' for per-command options.",
    )
    parser.add_argument("--version", action="version", version=f"pyxany {__version__}")
    parser.add_argument("--print-completion", choices=["bash", "zsh"],
                        help="print a shell completion script and exit")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def common(p, with_input=True):
        if with_input:
            p.add_argument("input", help="YAML input file")
        p.add_argument("-o", "--outdir", default=None, help="output directory")
        p.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                       help="override a YAML setting, e.g. --set scf.alpha=0.75 "
                            "(repeatable)")
        p.add_argument("--basis", default=None, help="shortcut for --set basis.set=...")
        p.add_argument("-v", "--verbose", action="count", default=0)
        p.add_argument("-q", "--quiet", action="store_true")
        p.add_argument("--json", action="store_true",
                       help="print the machine-readable result to stdout")
        p.add_argument("--log", metavar="FILE", help="also write all output to FILE")
        p.add_argument("--numba", dest="numba", action="store_true", default=None)
        p.add_argument("--no-numba", dest="numba", action="store_false")
        p.add_argument("--timing", dest="timing", action="store_true", default=None)
        p.add_argument("--no-timing", dest="timing", action="store_false")
        p.add_argument("--dry-run", action="store_true",
                       help="resolve settings and estimate resources, then stop")
        return p

    def restartable(p):
        p.add_argument("--restart", metavar="FILE", default=None,
                       help="resume from (or, at another geometry, start from) an SCF "
                            "checkpoint, e.g. <outdir>/<prefix>_scf.npz")
        return p

    p_run = restartable(common(sub.add_parser("run", help="single-point SCF")))
    p_run.add_argument("--no-figure", action="store_true")

    p_opt = restartable(common(sub.add_parser("opt", help="geometry optimization")))
    p_opt.add_argument("--freeze-core", action="store_true",
                       help="relax only the pseudo-hydrogen terminators")
    p_opt.add_argument("--max-steps", type=int, default=None)

    p_f = restartable(common(sub.add_parser("forces", help="forces only")))
    p_f.add_argument("--method", choices=["analytic", "finite_difference"], default=None)
    p_f.add_argument("--compare", action="store_true",
                     help="compute both analytic and finite-difference forces "
                          "and report the difference")
    p_f.add_argument("--fd-step", type=float, default=None, help="Bohr")

    p_scan = restartable(common(sub.add_parser("scan", help="scan a bond length or angle")))
    p_scan.add_argument("--coord", required=True, metavar="SPEC",
                        help="'bond:i,j' or 'angle:i,j,k' (0-based atom indices)")
    p_scan.add_argument("--range", nargs=2, type=float, required=True,
                        metavar=("LO", "HI"), help="Angstrom for bond, degrees for angle")
    p_scan.add_argument("--steps", type=int, default=11)

    p_gc = common(sub.add_parser("gridcheck", help="grid quality diagnostic"))
    p_gc.add_argument("--levels", default="20x10,30x12,40x14,50x16",
                      help="comma-separated n_r x n_a levels to test")

    common(sub.add_parser("validate", help="validate an input file and stop"))

    p_p = sub.add_parser("passivate", help="add hydrogen / pseudo-hydrogen terminators")
    p_p.add_argument("structure", help="cluster .xyz file to terminate")
    p_p.add_argument("--scheme", choices=["pseudo_hydrogen", "hydrogen"],
                     default="pseudo_hydrogen")
    p_p.add_argument("-o", "--output", default=None, help="output .xyz path")
    p_p.add_argument("--emit-yaml", default=None,
                     help="write a matching YAML input file")
    p_p.add_argument("--reference", default=None,
                     help="bulk .xyz used to recover the removed bond directions")
    p_p.add_argument("--coordination", type=int, default=None,
                     help="ideal coordination number (default: detected)")
    p_p.add_argument("--tolerance", type=float, default=1.20,
                     help="bond cutoff as a multiple of the summed covalent radii")
    p_p.add_argument("--role", action="append", metavar="EL=ROLE",
                     help="force an element's role, e.g. --role Mn=cation (repeatable); "
                          "needed where valence counts cannot tell anion from cation")
    p_p.add_argument("--bond-scale", type=float, default=None,
                     help="override the X-H* distance scale factor")
    p_p.add_argument("-q", "--quiet", action="store_true")
    p_p.add_argument("--timing", dest="timing", action="store_true", default=None)
    p_p.add_argument("--no-timing", dest="timing", action="store_false")

    p_e = sub.add_parser("embed", help="point charges of the surrounding crystal")
    p_e.add_argument("input", help="crystal description YAML (lattice, sites, cells, cluster)")
    p_e.add_argument("-o", "--output", default="charges.xyz", help="point-charge file to write")
    p_e.add_argument("--allow-substitution", action="store_true",
                     help="let a cluster atom sit on a site of another element")
    p_e.add_argument("--allow-dipole", action="store_true",
                     help="write the charges even if the block has a dipole (polar cell)")

    p_init = sub.add_parser("init", help="write a template input pair")
    p_init.add_argument("name")
    p_init.add_argument("--template", choices=["molecule", "cluster", "radical"],
                        default="molecule")
    p_init.add_argument("-o", "--outdir", default=".")

    p_b = sub.add_parser("basis", help="basis-set coverage information")
    p_b.add_argument("key", nargs="?", help="basis key; omit to list all")
    p_b.add_argument("--element", default=None, help="report one element only")

    sub.add_parser("info", help="runtime environment")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "print_completion", None):
        print(_completion_script(args.print_completion))
        return EXIT_OK
    if not args.command:
        parser.print_help()
        return EXIT_OK

    from ..config import ConfigError
    from . import basis as basis_cmd
    from . import embed as embed_cmd
    from . import forces as forces_cmd
    from . import gridcheck as gridcheck_cmd
    from . import info as info_cmd
    from . import init as init_cmd
    from . import opt as opt_cmd
    from . import passivate as passivate_cmd
    from . import run as run_cmd
    from . import scan as scan_cmd
    from . import validate as validate_cmd

    handlers = {
        "run": run_cmd.execute, "opt": opt_cmd.execute, "forces": forces_cmd.execute,
        "scan": scan_cmd.execute, "gridcheck": gridcheck_cmd.execute,
        "validate": validate_cmd.execute, "passivate": passivate_cmd.execute,
        "embed": embed_cmd.execute, "init": init_cmd.execute, "basis": basis_cmd.execute,
        "info": info_cmd.execute,
    }
    # The error messages are printed inside the tee, so that --log records why
    # a run failed; they used to go to stderr after the log was closed, and the
    # log of a failed run was empty (ADR 0012, F4).
    with _Tee(getattr(args, "log", None)):
        try:
            return handlers[args.command](args)
        except ConfigError as exc:
            print(f"\n  [input error] {exc}\n", file=sys.stderr)
            return _abort_if_parallel(EXIT_INPUT)
        except FileNotFoundError as exc:
            print(f"\n  [input error] {exc}\n", file=sys.stderr)
            return _abort_if_parallel(EXIT_INPUT)
        except KeyboardInterrupt:  # pragma: no cover
            print("\n  interrupted\n", file=sys.stderr)
            return _abort_if_parallel(EXIT_RUNTIME)
        except Exception as exc:  # pragma: no cover - unexpected
            import traceback

            if getattr(args, "verbose", 0):
                traceback.print_exc()
            print(f"\n  [error] {type(exc).__name__}: {exc}\n", file=sys.stderr)
            return _abort_if_parallel(EXIT_RUNTIME)


def _abort_if_parallel(code: int) -> int:
    """End every rank when one fails, instead of returning on that rank only.

    Rank-0-only work (output directory, result and figure files, checkpoints)
    can fail on rank 0 alone; returning there left the other ranks waiting in
    the next collective until the wall-time limit (ADR 0012, F14).
    """
    from ..parallel import COMM, PARALLEL

    if PARALLEL:  # pragma: no cover - needs mpirun
        sys.stderr.flush()
        sys.stdout.flush()
        COMM.Abort(code)
    return code


class _Tee:
    """Mirror stdout and stderr to a log file, for long unattended runs.

    Only rank 0 opens the file: every rank opening it with mode 'w' let a late
    rank truncate what rank 0 had already written (ADR 0012, F4). Other ranks
    print nothing to stdout anyway (mpi_print); their stderr is not logged.
    """

    def __init__(self, path):
        self.path = path
        self._fh = None
        self._orig = None

    def __enter__(self):
        from ..parallel import is_root

        if not self.path or not is_root():
            return self
        self._fh = open(self.path, "w")
        self._orig = (sys.stdout, sys.stderr)

        class _Writer:
            def __init__(self, a, b):
                self.a, self.b = a, b

            def write(self, s):
                self.a.write(s)
                self.b.write(s)

            def flush(self):
                self.a.flush()
                self.b.flush()

        sys.stdout = _Writer(self._orig[0], self._fh)
        sys.stderr = _Writer(self._orig[1], self._fh)
        return self

    def __exit__(self, *exc):
        if self._fh:
            sys.stdout, sys.stderr = self._orig
            self._fh.close()
        return False


def _completion_script(shell: str) -> str:
    cmds = "run opt forces scan passivate embed gridcheck validate init basis info"
    if shell == "bash":
        return (
            "_pyxany_completion() {\n"
            f'  local cmds="{cmds}"\n'
            '  COMPREPLY=($(compgen -W "$cmds" -- "${COMP_WORDS[COMP_CWORD]}"))\n'
            "}\n"
            "complete -F _pyxany_completion pyxany\n"
        )
    return f"#compdef pyxany\n_arguments '1:command:({cmds})' '*:file:_files'\n"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
