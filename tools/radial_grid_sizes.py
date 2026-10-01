"""Measure how many radial points each element needs on pyxany's own DV grid.

One-off generator (not a runtime dependency). For every supported element, in
order of atomic number, the spherical X-alpha atom (alpha = 0.7) and its
cation are solved on a dense logarithmic grid -- the near-exact reference: the
solver reproduces the hydrogenic spectrum to 1e-11. Each density is then
integrated on a one-atom DV grid with the same Treutler-Ahlrichs M4 mapping,
Bragg-Slater scaling and Hartree solver a molecular run uses, for a range of
n_r. Only grid-integrated terms are compared (pyxany's kinetic energy is
analytic): N, E_ne = -Z int rho/r, E_H, E_x.

Two criteria are measured; the table's ``n_r`` uses the second (ADR 0008, Q23):

* ``n_r_total``: max(|dN|, |dE_ne|, |dE_H|, |dE_x|) / N -- the absolute
  total-energy error per electron. Dominated by the cores, which move rigidly
  with their atom and cancel in energy differences, forces and gaps.
* ``n_r_ip``: the error in the grid-integrated part of the ionisation energy
  E(neutral) - E(cation), removing one electron from the last subshell of the
  tabulated configuration -- a chemically meaningful energy difference.

For each, the target is the worst of C, N, O, F at the baseline n_r (30, the
default), and an element's entry is the smallest n_r meeting it for 11
consecutive values. Writes pyxany/data/tables/radial_points.json.

    python tools/radial_grid_sizes.py [--baseline 30] [--max-z 86] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from pyxany.basis.atomic_solver import _shell_occupations, solve_atom
from pyxany.data import elements as elem
from pyxany.grid import build_molecular_grid
from pyxany.integrals import xc
from pyxany.integrals.coulomb import hartree_potential
from pyxany.molecule import Atom, Molecule

DEST = Path(__file__).resolve().parent.parent / "pyxany" / "data" / "tables" / "radial_points.json"
ALPHA = 0.7
SECOND_ROW = ("C", "N", "O", "F")
TERMS = ("E_ne", "E_H", "E_x")


def _cation_occupations(symbol: str) -> dict:
    occ = dict(_shell_occupations(symbol))
    last = list(elem.occupations(symbol))[-1]
    key = (next(int(c) for c in last if c.isdigit()), elem._subshell_l(last))
    occ[key] -= 1.0
    return {k: v for k, v in occ.items() if v > 1e-12}


def reference(symbol: str, occupations: dict | None = None):
    """Near-exact energy terms on the log grid, plus a density interpolant."""
    Z = float(elem.atomic_number(symbol))
    if occupations is not None and not occupations:  # bare nucleus (H+)
        return {"N": 0.0, "E_ne": 0.0, "E_H": 0.0, "E_x": 0.0}, None, 0.0, True
    sol = solve_atom(symbol, occupations=occupations, n_points=4000, alpha=ALPHA, tol=1e-11)
    r, rho = sol.r, sol.density
    w = np.log(r[1] / r[0]) * r  # int f dr on the log grid
    shell = 4.0 * np.pi * r * r * rho
    trap = lambda f: np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] * w[1:] + f[:-1] * w[:-1]))])  # noqa: E731
    inner, outer = trap(shell), trap(shell / r)
    v_h = inner / r + (outer[-1] - outer)
    terms = {
        "N": float(w @ shell),
        "E_ne": float(-Z * (w @ (shell / r))),
        "E_H": float(0.5 * (w @ (shell * v_h))),
        "E_x": float(w @ (shell * xc.energy_density(rho, ALPHA))),
    }
    density = CubicSpline(np.log(r), np.log(np.maximum(rho, 1e-300)))
    return terms, density, r[-1], sol.converged


def on_dv_grid(symbol: str, density, r_max: float, n_r: int) -> dict:
    if density is None:
        return {"N": 0.0, "E_ne": 0.0, "E_H": 0.0, "E_x": 0.0}
    mol = Molecule([Atom.from_element(symbol, np.zeros(3))], name=symbol)
    grid = build_molecular_grid(mol, n_r, 4, radial_by_element=False)
    rr = np.linalg.norm(grid.points, axis=1)
    rho = np.where(rr <= r_max, np.exp(density(np.log(np.maximum(rr, 1e-12)))), 0.0)
    w = grid.weights
    Z = float(elem.atomic_number(symbol))
    v_h = hartree_potential(grid, rho, mol.positions, l_max=0)
    return {
        "N": float(w @ rho),
        "E_ne": float(-Z * (w @ (rho / np.maximum(rr, 1e-12)))),
        "E_H": float(0.5 * (w @ (rho * v_h))),
        "E_x": float(w @ (rho * xc.energy_density(rho, ALPHA))),
    }


def total_error(ref, got) -> float:
    return max(abs(got[k] - ref[k]) for k in ref) / ref["N"]


def ip_error(ref_n, ref_c, got_n, got_c) -> float:
    exact = sum(ref_n[k] - ref_c[k] for k in TERMS)
    return abs(sum(got_n[k] - got_c[k] for k in TERMS) - exact)


def first_stable(errors: dict, target: float) -> int | None:
    for n in sorted(errors):
        window = [errors.get(m) for m in range(n, n + 11)]
        if all(e is not None and e <= target for e in window):
            return n
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--baseline", type=int, default=30, help="n_r whose second-row accuracy defines the targets"
    )
    ap.add_argument("--max-z", type=int, default=86)
    ap.add_argument("--n-max", type=int, default=160)
    ap.add_argument("--out", type=Path, default=DEST)
    args = ap.parse_args()

    symbols = sorted(
        (s for s in elem.symbols() if elem.atomic_number(s) <= args.max_z), key=elem.atomic_number
    )
    data = {}
    for s in symbols:
        neutral, cation = reference(s), reference(s, _cation_occupations(s))
        if not (neutral[3] and cation[3]):
            print(f"WARNING: atomic solver did not converge for {s}")
        tot, ip = {}, {}
        for n in range(10, args.n_max + 1):
            gn = on_dv_grid(s, neutral[1], neutral[2], n)
            gc = on_dv_grid(s, cation[1], cation[2], n)
            tot[n] = total_error(neutral[0], gn)
            ip[n] = ip_error(neutral[0], cation[0], gn, gc)
        data[s] = (tot, ip, neutral[3] and cation[3])

    t_tot = max(data[s][0][args.baseline] for s in SECOND_ROW)
    t_ip = max(data[s][1][args.baseline] for s in SECOND_ROW)
    print(
        f"targets: total {t_tot:.3e} Ha/e, IP {t_ip:.3e} Ha "
        f"(worst of {SECOND_ROW} at n_r = {args.baseline})"
    )

    table = {}
    for s in symbols:
        tot, ip, conv = data[s]
        table[s] = {
            "Z": elem.atomic_number(s),
            "n_r": first_stable(ip, t_ip),
            "n_r_total": first_stable(tot, t_tot),
            "ip_error_at_baseline_ha": ip[args.baseline],
            "total_error_at_baseline_ha_per_e": tot[args.baseline],
            "solver_converged": bool(conv),
        }
        print(
            f"{table[s]['Z']:3d} {s:2s}  n_r(ip) = {table[s]['n_r']}  "
            f"n_r(total) = {table[s]['n_r_total']}  IP error at {args.baseline}: "
            f"{ip[args.baseline] * 1e3:.3f} mHa",
            flush=True,
        )

    base = max(table[s]["n_r"] for s in SECOND_ROW)
    for s in table:
        if table[s]["n_r"] is not None:
            table[s]["ratio_to_second_row"] = round(table[s]["n_r"] / base, 4)
    args.out.write_text(
        json.dumps(
            {
                "_source": "Measured with tools/radial_grid_sizes.py on pyxany's own DV radial grid "
                "(Treutler-Ahlrichs M4, Bragg-Slater scaling, production Hartree solver) "
                "against the near-exact spherical X-alpha atom and cation (alpha = 0.7).",
                "_criterion": f"n_r: smallest n_r whose grid error in the ionisation energy stays <= "
                f"{t_ip:.3e} Ha for 11 consecutive n_r; n_r_total: same for the "
                f"total-energy error per electron <= {t_tot:.3e} Ha. Targets = worst of "
                f"{list(SECOND_ROW)} at n_r = {args.baseline}. ratio_to_second_row = "
                f"n_r / max over {list(SECOND_ROW)}.",
                "_generated": datetime.date.today().isoformat(),
                "elements": table,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
