"""Cached loaders for the JSON data tables shipped inside the package.

The tables were extracted mechanically from the original single-file script by
``tools/extract_tables.py``; ``tests/test_data.py`` re-imports that script and
asserts every key and value survived the conversion.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_TABLES = Path(__file__).resolve().parent / "tables"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    path = _TABLES / f"{name}.json"
    if not path.exists():  # pragma: no cover - packaging error
        raise FileNotFoundError(f"data table '{name}' not found at {path}")
    with path.open() as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def element_table() -> dict:
    """{symbol: {"Z": int, "occ": {label: float}, "bragg_slater_bohr": float}}"""
    return _load("elements")


@lru_cache(maxsize=None)
def sto_table(name: str) -> dict:
    """STO exponents. ``name`` in {sz, dz, tz, pol_d, pol_f}.

    Returns {symbol: [(n, l, m, zeta), ...]}.
    """
    key = {"sz": "sto_sz", "dz": "sto_dz", "tz": "sto_tz",
           "pol_d": "sto_pol_d", "pol_f": "sto_pol_f"}[name]
    return {sym: [tuple(e[:3]) + (float(e[3]),) for e in funcs]
            for sym, funcs in _load(key).items()}


@lru_cache(maxsize=None)
def gto_table(name: str) -> dict:
    """Contracted GTO table. ``name`` in {sto3g, 631g}.

    Returns {symbol: {shell_label: [(alpha, d, [d_p]), ...]}}.
    """
    key = {"sto3g": "gto_sto3g", "631g": "gto_631g"}[name]
    return {sym: {shell: [tuple(p) for p in prims] for shell, prims in shells.items()}
            for sym, shells in _load(key).items()}


@lru_cache(maxsize=None)
def gto_polarisation_exponents() -> dict:
    """6-31G(d) d-polarisation exponents, Hariharan & Pople (1973)."""
    return {k: float(v) for k, v in _load("gto_631gd_pol").items()}


@lru_cache(maxsize=None)
def orbital_labels() -> dict:
    """{(n, l, m): "2pz"} short orbital labels."""
    out = {}
    for key, label in _load("orbital_labels").items():
        n, l, m = (int(x) for x in key.split(","))
        out[(n, l, m)] = label
    return out


def coverage_report() -> list[tuple[str, int, str]]:
    """(table, n_elements, note) rows describing which elements each table covers.

    Used by ``dvxa basis`` and by test_data.py to keep the gaps visible rather
    than silently falling back to a smaller basis.
    """
    rows = []
    el = set(element_table())
    for name in ("sz", "dz", "tz", "pol_d", "pol_f"):
        t = sto_table(name)
        missing = sorted(el - set(t))
        note = "complete" if not missing else f"{len(missing)} missing, e.g. {missing[:5]}"
        rows.append((f"sto_{name}", len(t), note))
    for name in ("sto3g", "631g"):
        t = gto_table(name)
        covered = ", ".join(sorted(t, key=lambda s: element_table()[s]["Z"]))
        rows.append((f"gto_{name}", len(t), f"covers {covered}"))
    return rows
