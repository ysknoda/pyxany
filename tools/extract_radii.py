#!/usr/bin/env python3
"""Extract the Cordero (2008) covalent radii into the package's own JSON table.

Source
------
B. Cordero, V. Gomez, A. E. Platero-Prats, M. Reves, J. Echeverria, E. Cremades,
F. Barragan and S. Alvarez, "Covalent radii revisited",
*Dalton Trans.* (2008) 2832-2838.  doi:10.1039/B801115J

The numeric values are read from the ``ase.data.covalent_radii`` transcription
of that table, which is the same data set.  ASE is used *only* by this
extraction script -- it is not a runtime dependency of pyxany.  A spot check
against the published paper is printed so the transcription can be verified.
"""
import json
from pathlib import Path

from ase.data import chemical_symbols, covalent_radii

DEST = Path(__file__).resolve().parent.parent / "pyxany" / "data" / "tables"

# Values quoted directly from Cordero et al. (2008), Table 2, in Angstrom.
SPOT_CHECK = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
              "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
              "Ga": 1.22, "As": 1.19, "Se": 1.20, "Zn": 1.22,
              "Cd": 1.44, "Mg": 1.41, "Ti": 1.60, "Cu": 1.32}


def main() -> None:
    table = {}
    for z, sym in enumerate(chemical_symbols):
        if z == 0:
            continue
        r = float(covalent_radii[z])
        if r > 0:
            table[sym] = round(r, 4)

    bad = {s: (table[s], v) for s, v in SPOT_CHECK.items()
           if abs(table[s] - v) > 5e-3}
    print(f"extracted {len(table)} elements (Z = 1..{max(chemical_symbols.index(s) for s in table)})")
    print(f"spot check against Cordero Table 2 ({len(SPOT_CHECK)} entries): "
          f"{'all match' if not bad else bad}")
    if bad:
        raise SystemExit("transcription mismatch -- refusing to write the table")

    payload = {
        "_source": "Cordero et al., Dalton Trans. (2008) 2832, doi:10.1039/B801115J",
        "_units": "angstrom",
        "_note": ("Single averaged value per element; the paper additionally gives "
                  "separate low-/high-spin radii for Mn, Fe and Co, and sp3/sp2/sp "
                  "values for C. The averaged value is used here."),
        "radii": table,
    }
    (DEST / "covalent_radii.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {DEST / 'covalent_radii.json'}")


if __name__ == "__main__":
    main()
