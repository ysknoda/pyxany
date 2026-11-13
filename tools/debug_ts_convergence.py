"""Per-iteration trace of the N2 1s core-hole transition-state SCF.

Diagnostic for ADR 0012 F8: the slow test
``test_core_hole_localizes_on_one_of_two_equivalent_atoms`` converges in ~100
iterations on macOS/Accelerate but not within 200 on Linux x86-64 CI. This
prints, for every iteration, the SCF history (E, |dE|, DIIS error, density
change, method, mixing) next to the hole manager's state (largest Loewdin
weight of an occupied eigenvector on N1 1s, whether the hole was taken from an
eigenvector or from the localized-combination correction, source orbital and
its energy), so the two platforms can be compared line by line.

Usage:  python tools/debug_ts_convergence.py [--max-iter 300] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from conftest import make_config, make_molecule  # noqa: E402
from pyxany.basis import build_basis  # noqa: E402
from pyxany.kernels import backend_name  # noqa: E402
from pyxany.scf import run_scf  # noqa: E402
from pyxany.scf import transition_state as ts_mod  # noqa: E402

N2 = [("N", 0, 0, 0), ("N", 0, 0, 1.098)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--no-stop", action="store_true",
                    help="disable the convergence test and run all --max-iter iterations")
    ap.add_argument("--json", type=Path, default=None, help="write the trace as JSON")
    args = ap.parse_args()

    trace = []
    original = ts_mod.HoleManager.apply

    def traced(self, eps, occ, C):
        out = original(self, eps, occ, C)
        LC = self._sqrtS @ C
        on_target = np.sum(LC[self._proj] ** 2, axis=0)
        pool = np.where(np.isfinite(eps) & (np.asarray(occ) > 1e-8))[0]
        order = pool[np.argsort(-on_target[pool])]
        trace.append(
            {
                "w1": float(on_target[order[0]]),
                "w2": float(on_target[order[1]]) if len(order) > 1 else float("nan"),
                "eigen_hole": self.correction is None,
                "source": int(self.last["source"]),
                "eps_source": float(self.last["eps_source"]),
            }
        )
        return out

    ts_mod.HoleManager.apply = traced

    mol = make_molecule("n2", N2)
    cfg = make_config(
        basis={"set": "dz"},
        grid={"n_r": 30, "n_a": 12, "poisson_l_max": 11},
        scf={"max_iter": args.max_iter, "e_tol": 1e-15 if args.no_stop else 1e-11,
             "d_tol": 1e-15 if args.no_stop else 1e-9},
        transition_state={"remove": {"atom": 1, "shell": "1s"}},
    )
    res = run_scf(mol, build_basis(mol, cfg), cfg, verbose=0)

    print(f"platform {platform.platform()}  numpy {np.__version__}  backend {backend_name()}")
    np.show_config() if "--blas" in sys.argv else None
    print(
        f"converged {res.converged}  iterations {res.n_iterations}  "
        f"eps_source {res.transition_state['eps_source']:.9f} Ha"
    )
    print(
        f"{'it':>4} {'E':>18} {'|dE|':>9} {'err':>9} {'d_rms':>9} {'method':>6} {'mix':>5} "
        f"{'w1':>8} {'w2':>8} {'eig':>4} {'src':>4} {'eps_src':>14}"
    )
    rows = []
    for h, t in zip(res.history, trace, strict=False):
        keys = ("iter", "E", "dE", "error", "d_rms", "commutator", "method", "mix")
        row = {**{k: h[k] for k in keys}, **t}
        rows.append(row)
        print(
            f"{h['iter']:>4} {h['E']:>18.10f} {h['dE']:>9.2e} {h['error']:>9.2e} "
            f"{h['d_rms']:>9.2e} {str(h['method']):>6} {h['mix']:>5.2f} "
            f"{t['w1']:>8.5f} {t['w2']:>8.5f} {'y' if t['eigen_hole'] else 'n':>4} "
            f"{t['source']:>4} {t['eps_source']:>14.9f}"
        )
    if args.json:
        args.json.write_text(json.dumps({"converged": res.converged, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
