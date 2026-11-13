#!/usr/bin/env python3
"""Extract the element/basis-set data tables out of the original dv-xa.py into JSON.

Run once during the packaging migration.  The companion test
(tests/test_data.py) re-imports the original script and asserts that every key
and every value survived the conversion bit-for-bit.
"""
import importlib.util
import json
import sys
from pathlib import Path

SRC = Path("/Users/yusuke/Claude_Code/dv-xa/dv-xa.py")
DEST = Path(__file__).resolve().parent.parent / "pyxany" / "data" / "tables"


def load_legacy():
    spec = importlib.util.spec_from_file_location("pyxany_legacy", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pyxany_legacy"] = mod
    spec.loader.exec_module(mod)
    return mod


def dump(name, obj):
    path = DEST / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    return path


def main():
    L = load_legacy()
    DEST.mkdir(parents=True, exist_ok=True)
    report = []

    # --- elements: Z, occupations, Bragg-Slater radius -----------------------
    elements = {}
    for sym, rec in L.ELEMENT_DB.items():
        elements[sym] = {
            "Z": int(rec["Z"]),
            "occ": {k: float(v) for k, v in rec["occ"].items()},
            "bragg_slater_bohr": float(L._BRAGG[sym]) if sym in L._BRAGG else None,
        }
    missing_bragg = [s for s, r in elements.items() if r["bragg_slater_bohr"] is None]
    report.append(("elements.json", len(elements),
                   f"missing Bragg radius: {missing_bragg or 'none'}"))
    dump("elements", elements)

    # --- STO exponent tables -------------------------------------------------
    # entries are (n, l, m, zeta) -> [n, l, m, zeta]
    for name, table in [("sto_sz", L.STO_SZ), ("sto_dz", L.STO_DZ), ("sto_tz", L.STO_TZ),
                        ("sto_pol_d", L.STO_POL), ("sto_pol_f", L.STO_POL_F)]:
        conv = {sym: [[int(n), int(l), int(m), float(z)] for n, l, m, z in funcs]
                for sym, funcs in table.items()}
        nfunc = sum(len(v) for v in conv.values())
        report.append((f"{name}.json", len(conv), f"{nfunc} basis functions"))
        dump(name, conv)

    # --- contracted GTO tables ----------------------------------------------
    for name, table in [("gto_sto3g", L.GTO_STO3G), ("gto_631g", L.GTO_631G)]:
        conv = {sym: {shell: [[float(x) for x in prim] for prim in prims]
                      for shell, prims in shells.items()}
                for sym, shells in table.items()}
        nshell = sum(len(v) for v in conv.values())
        report.append((f"{name}.json", len(conv), f"{nshell} shells"))
        dump(name, conv)

    dump("gto_631gd_pol", {k: float(v) for k, v in L.GTO_631Gd_pol.items()})
    report.append(("gto_631gd_pol.json", len(L.GTO_631Gd_pol), "d polarization exponents"))

    # --- orbital label map: (n,l,m) -> label, stored as "n,l,m" keys ---------
    orb = {f"{n},{l},{m}": lbl for (n, l, m), lbl in L._ORB_LABEL.items()}
    dump("orbital_labels", orb)
    report.append(("orbital_labels.json", len(orb), "(n,l,m) -> label"))

    print(f"{'file':22s} {'entries':>8}   note")
    print("-" * 78)
    for f, n, note in report:
        print(f"{f:22s} {n:>8}   {note}")
    print(f"\nwritten to {DEST}")


if __name__ == "__main__":
    main()
