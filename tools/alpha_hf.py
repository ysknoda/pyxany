"""Spin-polarized alpha_HF for H-Rn, and a recomputation of Schwarz's alpha_HF (ADR 0011).

pyxany's ``alpha_hf`` uses Schwarz's published values (pyxany/data/tables/
schwarz_alpha.json); this tool provides the spin-polarized table, which has no
published counterpart, and recomputes the restricted one as a check.

alpha_HF is the X-alpha parameter for which the X-alpha total energy of the
free atom equals its Hartree-Fock energy (K. Schwarz, Phys. Rev. B 5, 2466
(1972); Theor. Chim. Acta 34, 225 (1974)). The original tables were not
available. Two tables are built from one PySCF ROHF calculation per element:

``alpha_hf`` (Schwarz-compatible)
    spin-restricted spherical X-alpha atom (``solve_atom``) against the HF
    *average-of-configuration* energy E_av. E_av is evaluated with the ROHF
    orbitals rebuilt spherically (one radial function per (n, l) shell, the
    leading singular vector of the shell's occupied orbitals). Checked against
    values recalled from the literature: Li, B, C, O, F, Na, Si, P, Cl within
    6e-5, N 1.9e-4, H 6.6e-4.
``alpha_hf_polarized``
    spin-polarized spherical X-alpha atom (Hund's rule, ``solve_atom_polarized``)
    against the ROHF energy of that configuration.

HF: PySCF ROHF, uncontracted ANO-RCC, non-relativistic. A second run with
every angular-momentum set extended by one tighter (x3) and one more diffuse
(/3) exponent checks basis completeness (``d_alpha_basis``, from
dE * alpha / |E_x|). Flags: ROHF not converged, the ROHF d/f electron count
differs from the configuration (another state found), basis error > 1e-4, or
the per-element time limit exceeded. Flagged elements are reported but left
out of the package table. An unconverged ROHF whose last ten energies agree
within 1e-5 Ha is kept, with a note (alpha moves by < 1e-7; Q62).

PySCF is used here only, never by pyxany itself. One element per subprocess
(so a stuck SCF can be timed out), one JSON line per element (so an
interrupted run resumes).

    python tools/alpha_hf.py --ledger alpha_hf.jsonl                 # H-Rn
    python tools/alpha_hf.py --ledger alpha_hf.jsonl --only O,Si
    python tools/alpha_hf.py --ledger alpha_hf.jsonl --retry         # flagged ones, SO3-fixed
    python tools/alpha_hf.py --ledger alpha_hf.jsonl --write         # build the table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TABLE = (Path(__file__).resolve().parents[1] / "pyxany" / "data" / "tables"
         / "alpha_hf_polarized.json")
FLAG_D_ALPHA = 1e-4
# recomputed vs published alpha_hf; Tb (4f9 6s2 here, 4f8 5d 6s2 there) is 3.7e-4
FLAG_SCHWARZ = 1e-3
NEWTON_MAX_CYCLE = 50  # Q63
SETTLED_E = 1e-5  # Q62: Hartree, spread of the last ten SCF energies
GUESS_SYMMETRIC = "SO3 symmetry, fixed (l, m) occupations"


# -- Hartree-Fock side (PySCF) ------------------------------------------------


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


def _ao_blocks(mol):
    """``{l: [first AO index of each radial function of angular momentum l]}``."""
    by_l, off = {}, 0
    for b in range(mol.nbas):
        l = mol.bas_angular(b)
        for _ in range(mol.bas_nctr(b)):
            by_l.setdefault(l, []).append(off)
            off += 2 * l + 1
    return by_l


def _mo_l(mol, C):
    S = mol.intor("int1e_ovlp")
    ao_l = np.zeros(mol.nao, dtype=int)
    for l, starts in _ao_blocks(mol).items():
        for s in starts:
            ao_l[s : s + 2 * l + 1] = l
    pop = np.array(
        [
            [np.sum((C[:, k] * (S @ C[:, k]))[ao_l == l]) for l in range(5)]
            for k in range(C.shape[1])
        ]
    )
    return pop.argmax(axis=1), pop


def spherical_shells(mol, C, occ, eps, config):
    """One radial function per (n, l) shell, shared by its 2l + 1 components."""
    S = mol.intor("int1e_ovlp")
    blocks = _ao_blocks(mol)
    l_mo, _ = _mo_l(mol, C)
    occupied = [k for k in np.argsort(eps) if occ[k] > 1e-8]
    cols, shells = [], []
    for l in sorted({l for (_, l) in config}):
        d, starts = 2 * l + 1, blocks[l]
        ns = sorted(n for (n, ll) in config if ll == l)
        occ_l = [k for k in occupied if l_mo[k] == l]
        radials, pos = [], 0
        for n in ns:
            q = config[(n, l)]
            k = d if q >= 2 * d - 1e-9 else min(d, int(np.ceil(q / 2 - 1e-9)) if q <= d else d)
            if q > d:  # more than half full: every component is occupied
                k = d
            mats = [
                np.array([[C[s + m, mo] for m in range(d)] for s in starts])
                for mo in occ_l[pos : pos + k]
            ]
            pos += k
            u, _, _ = np.linalg.svd(np.hstack(mats))
            radials.append(u[:, 0])
        Sl = np.array([[S[a, b] for b in starts] for a in starts])
        ortho = []
        for r in radials:  # Gram-Schmidt in n order
            for q_ in ortho:
                r = r - (q_ @ Sl @ r) * q_
            ortho.append(r / np.sqrt(r @ Sl @ r))
        for n, r in zip(ns, ortho, strict=True):
            idx = []
            for m in range(d):
                col = np.zeros(mol.nao)
                for s, c in zip(starts, r, strict=True):
                    col[s + m] = c
                idx.append(len(cols))
                cols.append(col)
            shells.append(((n, l), idx, float(config[(n, l)])))
    return np.array(cols).T, shells


def average_energy(mol, C, shells) -> float:
    """HF average-of-configuration energy for spatial shell orbitals ``C``.

    Shell a holds q_a electrons in g_a = 2(2l+1) spin orbitals; averaging over
    all determinants of the configuration,

        E = sum_a (q_a / d_a) sum_i h_ii
          + sum_{a<b} q_a q_b / (g_a g_b) S_ab + sum_a q_a (q_a - 1) / (2 g_a (g_a - 1)) S_aa,
        S_ab = 4 sum_{i in a, j in b} J_ij - 2 sum K_ij,

    which reduces to the closed-shell HF energy when every shell is full.
    """
    from pyscf import ao2mo

    h = C.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), C.shape[1])
    J = np.einsum("iijj->ij", eri)
    K = np.einsum("ijji->ij", eri)
    E = 0.0
    for a, (_, ia, qa) in enumerate(shells):
        ga = 2 * len(ia)
        E += qa / len(ia) * sum(h[i, i] for i in ia)
        for b, (_, ib, qb) in enumerate(shells):
            if b < a:
                continue
            s_ab = 4.0 * J[np.ix_(ia, ib)].sum() - 2.0 * K[np.ix_(ia, ib)].sum()
            gb = 2 * len(ib)
            c = qa * qb / (ga * gb) if a != b else qa * (qa - 1) / (2.0 * ga * (ga - 1))
            E += c * s_ab
    return float(E)


def targeted_guess(mol, config: dict):
    """Initial orbitals and occupations that realise ``config`` (Hund's-rule spins).

    Spherical orbitals come from PySCF's spherically averaged RHF; per angular
    momentum they are taken in energy order and dealt to the (n, l) shells of
    the configuration (virtual orbitals fill shells that this reference leaves
    empty). Spin-up electrons occupy the first components of each shell, then
    spin-down ones. Returns ``(mo_coeff, (occ_alpha, occ_beta))``.
    """
    from pyscf.scf import atom_hf

    from pyxany.basis.atomic_solver import hund_spin_occupations

    ref = atom_hf.AtomSphAverageRHF(mol)
    ref.verbose = 0
    ref.kernel()
    C, eps = ref.mo_coeff, ref.mo_energy
    l_mo, _ = _mo_l(mol, C)
    up, down = hund_spin_occupations(config)
    occ_a = np.zeros(C.shape[1])
    occ_b = np.zeros(C.shape[1])
    for l in sorted({l for (_, l) in config}):
        d = 2 * l + 1
        order = [k for k in np.argsort(eps) if l_mo[k] == l]
        for i, n in enumerate(sorted(n for (n, ll) in config if ll == l)):
            comps = order[i * d:(i + 1) * d]
            for j, k in enumerate(comps):
                occ_a[k] = 1.0 if j < round(up.get((n, l), 0.0)) else 0.0
                occ_b[k] = 1.0 if j < round(down.get((n, l), 0.0)) else 0.0
    return C, (occ_a, occ_b)


def symmetric_occupations(mol, config: dict) -> dict:
    """``irrep_nelec`` for SO3 symmetry: (alpha, beta) electrons per (l, m) irrep.

    Hund's-rule spins per (n, l) shell; within a partly filled shell each spin
    fills m = +l, l-1, ... in turn (Q55). The irreps are real spherical
    harmonics, so for a single open-shell electron next to filled or half-filled
    shells (Gd 4f7 5d1) the determinant is a component of the Hund term, and
    otherwise it is one fixed determinant of the configuration.
    """
    from pyxany.basis.atomic_solver import hund_spin_occupations

    up, down = hund_spin_occupations(config)
    nelec = {name: [0, 0] for name in mol.irrep_name}
    for (n, l) in config:
        for spin, count in ((0, up.get((n, l), 0.0)), (1, down.get((n, l), 0.0))):
            if abs(count - round(count)) > 1e-9:
                raise ValueError(f"fractional occupation {count} in shell {(n, l)}")
            for m in range(l, l - int(round(count)), -1):
                nelec[f"{'spdfghi'[l]}{m:+d}"][spin] += 1
    return {k: tuple(v) for k, v in nelec.items()}


def hartree_fock(symbol: str, spin: int, config: dict, extended: bool,
                 targeted: bool = False, max_cycle: int = 200,
                 symmetric: bool = False) -> dict:
    from pyscf import gto, scf

    mol = gto.M(
        atom=f"{symbol} 0 0 0", basis={symbol: _basis(symbol, extended)}, spin=spin, verbose=0,
        symmetry="SO3" if symmetric else False,
    )
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = max_cycle
    if symmetric:
        # the configuration is fixed by symmetry: electrons cannot change l
        mf.irrep_nelec = symmetric_occupations(mol, config)
        # Q60: in the extended basis Nd alternates between two solutions 2.1 Ha
        # apart (orbital swaps within an irrep); a level shift settles it, and an
        # unshifted pass from that density confirms the stationary point
        mf.level_shift = 0.5
        e = mf.kernel()
        mf.level_shift = 0.0
        e = mf.kernel(mf.make_rdm1())
    elif targeted:
        # start in the wanted configuration and keep it (maximum overlap method)
        C0, (occ_a, occ_b) = targeted_guess(mol, config)
        mf = scf.addons.mom_occ(mf, C0, (occ_a, occ_b))
        dm0 = np.array([(C0 * occ_a) @ C0.T, (C0 * occ_b) @ C0.T])
        e = mf.kernel(dm0)
    else:
        e = mf.kernel()
    if not mf.converged:
        # second order; the MOM occupation rule (mf.get_occ) is kept by newton
        mf = scf.newton(mf)
        # Q63: Sm settles within 4-6 macro cycles and never improves afterwards,
        # and each one costs ~25 s, so 50 suffice for the settled-energy check
        mf.max_cycle = min(max_cycle, NEWTON_MAX_CYCLE)
        history = []
        mf.callback = lambda env: history.append(env["e_tot"])
        e = mf.kernel(mf.make_rdm1())
    else:
        history = []
    # Q62: spread of the last ten energies; Sm stalls within 2e-6 Ha with the
    # orbital gradient stuck near 0.08 (near-degenerate f orbitals swap)
    spread = float(np.ptp(history[-10:])) if len(history) >= 10 else None
    l_mo, _ = _mo_l(mol, mf.mo_coeff)
    counts = {
        l: float(sum(mf.mo_occ[k] for k in range(len(l_mo)) if l_mo[k] == l)) for l in range(4)
    }
    try:
        C, shells = spherical_shells(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy, config)
        e_av = average_energy(mol, C, shells)
    except (ValueError, IndexError):     # occupied shells do not match the configuration
        e_av = None
    return {
        "E_rohf": float(e),
        "E_av": e_av,
        "converged": bool(mf.converged),
        "energy_spread": spread,
        "l_counts": counts,
    }


# -- X-alpha side (pyxany) and the root search ------------------------------------


def _alpha_for(target: float, energy) -> tuple[float, float]:
    """``(alpha, dE/dalpha)`` with ``energy(alpha) == target``."""
    from scipy.optimize import brentq

    a = brentq(lambda x: energy(x) - target, 0.5, 1.3, xtol=1e-8)
    h = 1e-4
    return a, (energy(a + h) - energy(a - h)) / (2 * h)


def one_element(symbol: str, targeted: bool = False, max_cycle: int = 200,
                symmetric: bool = False) -> dict:
    from pyxany.basis.atomic_solver import (
        _shell_occupations,
        hund_spin_occupations,
        solve_atom,
        solve_atom_polarized,
        total_energy,
    )
    from pyxany.data import elements as elem

    config = _shell_occupations(symbol)
    up, down = hund_spin_occupations(config)
    spin = int(round(sum(up.values()) - sum(down.values())))
    t0 = time.perf_counter()
    hf = hartree_fock(symbol, spin, config, extended=False, targeted=targeted,
                      max_cycle=max_cycle, symmetric=symmetric)
    hf2 = hartree_fock(symbol, spin, config, extended=True, targeted=targeted,
                       max_cycle=max_cycle, symmetric=symmetric)

    def restricted(a):
        return total_energy(solve_atom(symbol, alpha=a, n_points=4000, tol=1e-11), a)

    def polarized(a):
        return solve_atom_polarized(symbol, alpha=a).energy

    flags = []
    e_rohf = min(hf["E_rohf"], hf2["E_rohf"])
    a_p, slope_p = _alpha_for(e_rohf, polarized)
    d_p = abs(hf["E_rohf"] - hf2["E_rohf"]) / abs(slope_p)
    if hf["E_av"] is None or hf2["E_av"] is None:
        e_av, a_r, d_r = None, None, 0.0
        flags.append("occupied shells do not match the configuration (no E_av)")
    else:
        e_av = min(hf["E_av"], hf2["E_av"])
        a_r, slope_r = _alpha_for(e_av, restricted)
        d_r = abs(hf["E_av"] - hf2["E_av"]) / abs(slope_r)

    notes = []
    if not (hf["converged"] and hf2["converged"]):
        spreads = [h["energy_spread"] for h in (hf, hf2) if not h["converged"]]
        if all(x is not None and x < SETTLED_E for x in spreads):
            # accepted: alpha moves by spread / |dE/dalpha|, below 1e-7
            notes.append(f"ROHF energy settled within {max(spreads):.1e} Ha (not converged)")
        else:
            flags.append("ROHF not converged")
    for l, name in ((2, "d"), (3, "f")):
        want = sum(v for (_, ll), v in config.items() if ll == l)
        got = hf["l_counts"][l]
        if abs(got - want) > 0.5:
            flags.append(f"ROHF has {got:.2f} {name} electrons, configuration {want:g}")
    if max(d_r, d_p) > FLAG_D_ALPHA:
        flags.append(f"basis: d_alpha {max(d_r, d_p):.1e}")
    return {
        "symbol": symbol,
        "Z": elem.atomic_number(symbol),
        "spin": spin,
        "alpha_hf": None if a_r is None else round(a_r, 6),
        "alpha_hf_polarized": round(a_p, 6),
        "E_av": e_av,
        "E_rohf": e_rohf,
        "d_alpha_basis": max(d_r, d_p),
        "flags": flags,
        "notes": notes,
        "seconds": round(time.perf_counter() - t0, 1),
        "initial_guess": GUESS_SYMMETRIC if symmetric
        else "targeted (configuration, MOM)" if targeted else "PySCF default",
        "max_cycle": max_cycle,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ledger", required=True, help="JSONL file, one element per line")
    p.add_argument("--only", default=None, help="comma-separated symbols")
    p.add_argument("--z-max", type=int, default=86)
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per element")
    p.add_argument("--write", action="store_true", help="build the package table")
    p.add_argument("--retry", action="store_true",
                   help="recompute flagged elements with the configuration fixed by SO3 symmetry")
    p.add_argument("--max-cycle", type=int, default=200, help="ROHF iterations (retry: 500)")
    p.add_argument("--redo", action="store_true",
                   help="recompute the --only symbols even if the ledger has them")
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    p.add_argument("--targeted", action="store_true",
                   help="start every ROHF from the configuration-targeted guess")
    p.add_argument("--symmetry", action="store_true",
                   help="fix the (l, m) occupations by SO3 symmetry (used by --retry)")
    args = p.parse_args()

    if args.worker:  # child process: one element to stdout
        print(json.dumps(one_element(args.worker, targeted=args.targeted,
                                     max_cycle=args.max_cycle, symmetric=args.symmetry)))
        return

    from pyxany.data import elements as elem

    ledger = Path(args.ledger)
    done = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            done[row["symbol"]] = row
    if args.write:
        from pyxany.data.loaders import schwarz_alpha

        rows = sorted(done.values(), key=lambda r: r["Z"])
        # a wrong state can pass every per-run check (Fe with the MOM guess sat
        # 0.39 Ha above the HF ground state, alpha_hf 6e-3 below Schwarz), so the
        # recomputed alpha_hf is also held against the published value
        published = {k: v["alpha_hf"] for k, v in schwarz_alpha().items()
                     if v["origin"] == "published"}
        for r in rows:
            ref = published.get(r["symbol"])
            if ref is not None and r["alpha_hf"] is not None \
                    and abs(r["alpha_hf"] - ref) > FLAG_SCHWARZ:
                r["flags"] = r["flags"] + [
                    f"alpha_hf {r['alpha_hf']:.5f} differs from Schwarz's {ref:.5f}"]
        good = [r for r in rows if not r["flags"]]
        table = {
            "_source": "Computed with tools/alpha_hf.py (ADR 0011); no published counterpart",
            "_alpha_hf_polarized": "spin-polarized spherical X-alpha atom (Hund's rule) = "
                                   "ROHF energy of the same configuration",
            "_alpha_hf_recomputed": "Schwarz's definition recomputed (spin-restricted X-alpha = "
                                    "HF average-of-configuration energy); a check against the "
                                    "published values, which pyxany uses for 'alpha_hf'",
            "_hf": "PySCF ROHF, uncontracted ANO-RCC (+ extended-basis check), non-relativistic",
            "_n_elements": len(good),
            "alpha_hf_polarized": {r["symbol"]: r["alpha_hf_polarized"] for r in good},
            "alpha_hf_recomputed": {r["symbol"]: r["alpha_hf"] for r in good},
            # Q59: ROHF with or without SO3 symmetry; for Sc the symmetry-broken
            # solution (s-d hybridization) is 2.1 mHa lower and alpha_hf_polarized
            # differs by 4e-5, alpha_hf (from E_av) not at all
            "rohf": {r["symbol"]: r.get("initial_guess", "PySCF default") for r in good},
            "notes": {r["symbol"]: r["notes"] for r in good if r.get("notes")},
            "excluded": {r["symbol"]: r["flags"] for r in rows if r["flags"]},
        }
        TABLE.write_text(json.dumps(table, indent=1) + "\n")
        print(f"wrote {len(good)} elements ({len(rows) - len(good)} excluded) to {TABLE}")
        return

    symbols = (
        args.only.split(",")
        if args.only
        else [elem.symbol_for_z(z) for z in range(1, args.z_max + 1)]
    )
    if args.retry:
        # flagged elements once more, with the configuration fixed by symmetry
        # (Q56) and up to 500 iterations (Q53); the new line is appended and
        # supersedes the old one when read back. An element already run with
        # the targeted guess is retried unless its only flag is the basis check.
        def wanted(row):
            if not row["flags"] or row.get("initial_guess") == GUESS_SYMMETRIC:
                return False
            if row.get("initial_guess") != "targeted (configuration, MOM)":
                return True
            return not all(f.startswith("basis:") for f in row["flags"])

        symbols = [s_ for s_ in symbols if s_ in done and wanted(done[s_])]
        args.max_cycle = max(args.max_cycle, 500)
        print(f"retrying {len(symbols)} flagged element(s): {' '.join(symbols)}", flush=True)
    for symbol in symbols:
        if symbol in done and not (args.retry or args.redo):
            continue
        t0 = time.perf_counter()
        try:
            out = subprocess.run(
                [sys.executable, __file__, "--ledger", str(ledger), "--worker", symbol]
                + (["--symmetry"] if args.retry or args.symmetry
                   else ["--targeted"] if args.targeted else [])
                + ["--max-cycle", str(args.max_cycle)],
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            row = json.loads(out.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            row = {
                "symbol": symbol,
                "Z": elem.atomic_number(symbol),
                "alpha_hf": None,
                "alpha_hf_polarized": None,
                "d_alpha_basis": None,
                "flags": [f"time limit {args.timeout:g} s exceeded"],
                "seconds": round(time.perf_counter() - t0, 1),
            }
        except (IndexError, json.JSONDecodeError):
            row = {
                "symbol": symbol,
                "Z": elem.atomic_number(symbol),
                "alpha_hf": None,
                "alpha_hf_polarized": None,
                "d_alpha_basis": None,
                "flags": [f"worker failed: {out.stderr.strip()[-200:]}"],
                "seconds": round(time.perf_counter() - t0, 1),
            }
        with ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        a_r = "-" if row["alpha_hf"] is None else f"{row['alpha_hf']:.5f}"
        a_p = "-" if row["alpha_hf_polarized"] is None else f"{row['alpha_hf_polarized']:.5f}"
        print(
            f"{row['Z']:>3} {symbol:<2} alpha_HF {a_r}  polarized {a_p}  "
            f"{'; '.join(row['flags'] + row.get('notes', []))}  ({row['seconds']} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
