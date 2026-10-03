"""Recompute Schwarz's alpha_HF for H-Rn (ADR 0011).

alpha_HF is the X-alpha parameter for which the X-alpha total energy of the
free atom equals its Hartree-Fock energy (K. Schwarz, Phys. Rev. B 5, 2466
(1972); Theor. Chim. Acta 34, 225 (1974)). The original tables were not
available, so the values are recomputed with the same definition:

* X-alpha: pyxany's spin-polarised spherical atom (Hund's-rule spins, each
  subshell spherically averaged), ``solve_atom_polarised`` on 4000 points.
* Hartree-Fock: PySCF ROHF of the same configuration and spin, in the
  uncontracted ANO-RCC basis (non-relativistic Hamiltonian). A second run with
  every angular-momentum set extended by one tighter (x3) and one more diffuse
  (/3) exponent checks basis completeness; ``d_alpha_basis`` is the resulting
  change of alpha_HF, estimated as dE * alpha / |E_x| (Hellmann-Feynman: dE/dalpha
  = E_x / alpha at fixed density).
* The ROHF d-electron count is compared with the configuration of the X-alpha
  atom; a mismatch (another configuration found) flags the element.

PySCF is used here only, never by pyxany itself. Results go to a JSONL ledger
(one line per element, so an interrupted run resumes) and then to
pyxany/data/tables/alpha_hf.json.

    python tools/alpha_hf.py --ledger alpha_hf.jsonl            # all of H-Rn
    python tools/alpha_hf.py --ledger alpha_hf.jsonl --only O,Si
    python tools/alpha_hf.py --ledger alpha_hf.jsonl --write    # build the table
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pyxany.basis.atomic_solver import (
    _shell_occupations,
    hund_spin_occupations,
    solve_atom_polarised,
)
from pyxany.data import elements as elem

TABLE = Path(__file__).resolve().parents[1] / "pyxany" / "data" / "tables" / "alpha_hf.json"
FLAG_D_ALPHA = 1e-4


def xalpha_energy(symbol: str, alpha: float) -> float:
    return solve_atom_polarised(symbol, alpha=alpha).energy


def _basis(symbol: str, extended: bool):
    from pyscf import gto

    shells = gto.uncontract(gto.basis.load("ano-rcc", symbol))
    if not extended:
        return shells
    by_l: dict[int, list[float]] = {}
    for shell in shells:
        by_l.setdefault(shell[0], []).append(shell[1][0])
    for l, exps in by_l.items():
        shells = shells + [[l, [3.0 * max(exps), 1.0]], [l, [min(exps) / 3.0, 1.0]]]
    return shells


def hartree_fock(symbol: str, spin: int, extended: bool) -> tuple[float, float, bool]:
    """``(E_HF, d-electron count, converged)`` from PySCF ROHF."""
    from pyscf import gto, scf

    mol = gto.M(
        atom=f"{symbol} 0 0 0", basis={symbol: _basis(symbol, extended)}, spin=spin, verbose=0
    )
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    e = mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)
        e = mf.kernel()
    # d electrons: Mulliken population on functions with l = 2
    dm = mf.make_rdm1()
    dm = dm[0] + dm[1] if np.ndim(dm) == 3 else dm
    ps = np.einsum("ij,ji->i", dm, mol.intor("int1e_ovlp"))
    ls = np.array(
        [
            mol.bas_angular(b)
            for b in range(mol.nbas)
            for _ in range(mol.bas_nctr(b) * (2 * mol.bas_angular(b) + 1))
        ]
    )
    return float(e), float(ps[ls == 2].sum()), bool(mf.converged)


def one_element(symbol: str) -> dict:
    occ = _shell_occupations(symbol)
    up, down = hund_spin_occupations(occ)
    spin = int(round(sum(up.values()) - sum(down.values())))
    n_d = sum(v for (_, l), v in occ.items() if l == 2)
    t0 = time.perf_counter()
    e_hf, d_hf, conv = hartree_fock(symbol, spin, extended=False)
    e_hf2, _, conv2 = hartree_fock(symbol, spin, extended=True)
    e_ref = min(e_hf, e_hf2)
    lo, hi = 0.55, 1.1
    alpha = brentq(lambda a: xalpha_energy(symbol, a) - e_ref, lo, hi, xtol=1e-8)
    # E_x at alpha_HF: dE/dalpha = E_x / alpha
    h = 1e-4
    dE_da = (xalpha_energy(symbol, alpha + h) - xalpha_energy(symbol, alpha - h)) / (2 * h)
    d_alpha = abs(e_hf - e_hf2) / abs(dE_da)
    flags = []
    if not (conv and conv2):
        flags.append("ROHF not converged")
    if abs(d_hf - n_d) > 0.5:
        flags.append(f"ROHF found {d_hf:.2f} d electrons, configuration has {n_d:g}")
    if d_alpha > FLAG_D_ALPHA:
        flags.append(f"basis: d_alpha {d_alpha:.1e}")
    return {
        "symbol": symbol,
        "Z": elem.atomic_number(symbol),
        "alpha_hf": round(alpha, 6),
        "E_hf": e_ref,
        "E_hf_ano_unc": e_hf,
        "E_hf_extended": e_hf2,
        "spin": spin,
        "d_alpha_basis": d_alpha,
        "flags": flags,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ledger", required=True, help="JSONL file, one element per line")
    p.add_argument("--only", default=None, help="comma-separated symbols")
    p.add_argument("--z-max", type=int, default=86)
    p.add_argument("--write", action="store_true", help="build the package table")
    args = p.parse_args()

    ledger = Path(args.ledger)
    done = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            done[row["symbol"]] = row
    if args.write:
        rows = sorted(done.values(), key=lambda r: r["Z"])
        table = {
            "_source": "Recomputed alpha_HF (Schwarz's definition), tools/alpha_hf.py; ADR 0011",
            "_definition": "E_Xalpha(spin-polarised spherical atom, Hund's rule) = E_ROHF "
            "(PySCF, uncontracted ANO-RCC, best of two basis variants)",
            "_n_elements": len(rows),
            "alpha_hf": {r["symbol"]: r["alpha_hf"] for r in rows},
            "flags": {r["symbol"]: r["flags"] for r in rows if r["flags"]},
        }
        TABLE.write_text(json.dumps(table, indent=1) + "\n")
        print(f"wrote {len(rows)} elements to {TABLE}")
        return

    symbols = (
        args.only.split(",")
        if args.only
        else [elem.symbol_for_z(z) for z in range(1, args.z_max + 1)]
    )
    for symbol in symbols:
        if symbol in done:
            continue
        row = one_element(symbol)
        with ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"{row['Z']:>3} {symbol:<2} alpha_HF {row['alpha_hf']:.5f}  E_HF {row['E_hf']:.6f}  "
            f"d_alpha {row['d_alpha_basis']:.1e}  {'; '.join(row['flags'])}  "
            f"({row['seconds']} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
