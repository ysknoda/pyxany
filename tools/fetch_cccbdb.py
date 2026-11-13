"""Fetch experimental reference data from NIST CCCBDB for the Tier B tests.

One-off extraction script (not a runtime dependency of pyxany): downloads the
CCCBDB "experimental data" page for each molecule, caches the raw HTML, and
writes the parsed geometry, dipole and ionization energy -- each with the
primary reference CCCBDB cites -- to tests/data/cccbdb_experimental.json.

Requests are rate-limited (one every --delay seconds) and served from the
cache on reruns, so the site is hit at most once per molecule.

    python tools/fetch_cccbdb.py --cache /path/to/cache
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import time
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

URL = "https://cccbdb.nist.gov/exp2x.asp?casno={cas}&charge=0"
DEST = Path(__file__).resolve().parent.parent / "tests" / "data" / "cccbdb_experimental.json"

#: name -> CAS registry number (CCCBDB key)
MOLECULES = {
    "H2": "1333740",
    "LiH": "7580678",
    "CH4": "74828",
    "NH3": "7664417",
    "H2O": "7732185",
    "HF": "7664393",
    "N2": "7727379",
    "CO": "630080",
    "SiH4": "7803625",
    "H2S": "7783064",
    "C6H6": "71432",
}


def _text(raw: str) -> list[str]:
    """Page text as a list of non-empty stripped cells/lines."""
    s = re.sub(r"<script.*?</script>", "", raw, flags=re.S)
    s = re.sub(r"<[^>]+>", "\n", s)
    s = html.unescape(s)
    return [t.strip() for t in s.split("\n") if t.strip()]


_REF = re.compile(r"\d{4}[A-Za-z]|NIST|NSRDS|webbook|Herzberg", re.I)
_COORD = re.compile(r"[rad][A-Z][A-Za-z]*\d*")


def _float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def parse(raw: str) -> dict:
    cells = _text(raw)
    out: dict = {}

    # Cartesians: "Cartesians" ... header "z (Å)", then (label, x, y, z) quads
    # until a non-numeric cell after a label.
    i = cells.index("Cartesians")
    j = next(k for k in range(i, len(cells)) if cells[k].startswith("z ("))
    atoms = []
    k = j + 1
    while k + 3 < len(cells):
        label = cells[k]
        xyz = [_float(c) for c in cells[k + 1 : k + 4]]
        m = re.fullmatch(r"([A-Z][a-z]?)\d+", label)
        if not m or None in xyz:
            break
        atoms.append([m.group(1), *xyz])
        k += 4
    out["geometry_angstrom"] = atoms

    # Internal coordinates: rows "rXY value unc atom atom ref comment"
    internal = []
    a = cells.index("Internal coordinates")
    b = cells.index("Cartesians")
    for k in range(a, b):
        if _COORD.fullmatch(cells[k]) and _float(cells[k + 1]) is not None:
            # value, unc, one connectivity index per atom, reference, then an
            # optional free-text comment ("re", "r0", "+- 0.0004", "by symmetry").
            n_idx = {"r": 2, "a": 3, "d": 4}[cells[k][0]]
            has_unc = "." in cells[k + 2] and _float(cells[k + 2]) is not None  # unc. is optional
            start = k + 2 + has_unc + n_idx
            tail = cells[start : start + 2]
            ref = tail[0] if tail and _REF.match(tail[0]) else None
            comment = None
            if ref and len(tail) > 1 and not _COORD.fullmatch(tail[1]) and tail[1] != "Cartesians":
                comment = tail[1]
            internal.append(
                {
                    "name": cells[k],
                    "value": _float(cells[k + 1]),
                    "reference": ref,
                    "comment": comment,
                }
            )
    out["internal_coordinates"] = internal

    # Dipole: first "total" value after the "Electric dipole moment" table header
    out["dipole_debye"] = None
    if "Electric dipole moment" in cells:
        d = cells.index("Electric dipole moment")
        hdr = next(k for k in range(d, len(cells)) if cells[k] == "quadrupole")
        for k in range(hdr + 1, min(hdr + 40, len(cells))):
            v = _float(cells[k])
            if v is not None and "." in cells[k]:
                ref = next((c for c in cells[k + 1 : k + 4] if _REF.match(c)), None)
                out["dipole_debye"] = {"value": v, "reference": ref}
                break

    # Ionization energy (adiabatic), eV
    out["ionization_energy_ev"] = None
    if "Ionization Energies (eV)" in cells:
        e = cells.index("Ionization Energies (eV)")
        hdr = next(k for k in range(e, len(cells)) if cells[k] == "reference")
        # an optional "Comment" header column may follow "reference"
        k = hdr + 1 + (cells[hdr + 1] == "Comment")
        ie = _float(cells[k])
        ref = next((c for c in cells[k + 1 : k + 5] if _REF.match(c)), None)
        out["ionization_energy_ev"] = None if ie is None else {"value": ie, "reference": ref}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True, help="directory for raw HTML")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between requests")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    result = {
        "_source": "NIST CCCBDB, Release 22 (https://cccbdb.nist.gov), exp2x.asp",
        "_retrieved": datetime.date.today().isoformat(),
        "_units": {
            "geometry": "Angstrom",
            "dipole": "Debye",
            "ionization_energy": "eV (adiabatic)",
        },
    }
    for name, cas in MOLECULES.items():
        path = args.cache / f"{name}.html"
        if not path.exists():
            req = urllib.request.Request(
                URL.format(cas=cas),
                headers={"User-Agent": "pyxany-validation-fetch/0.1 (one-off research script)"},
            )
            with opener.open(req, timeout=60) as r:
                path.write_bytes(r.read())
            print(f"fetched {name}")
            time.sleep(args.delay)
        entry = parse(path.read_text(encoding="latin-1"))
        entry["cas"] = cas
        entry["url"] = URL.format(cas=cas)
        result[name] = entry
        internal = [(c['name'], c['value'], c['comment']) for c in entry['internal_coordinates']]
        print(
            f"{name:5s} {len(entry['geometry_angstrom'])} atoms  "
            f"internal={internal}"
            f"  mu={entry['dipole_debye']}  IE={entry['ionization_energy_ev']}"
        )
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
