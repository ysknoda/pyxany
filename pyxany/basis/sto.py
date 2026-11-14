"""Slater-type basis-set construction from the tabulated exponents."""

from __future__ import annotations

from collections import defaultdict

from ..config import ConfigError
from ..data import elements as elem
from ..data.loaders import orbital_labels, sto_table
from .types import STO

#: Recognized STO basis keys.
STO_KEYS = ("sz", "szp", "dz", "dzp", "tz", "tzp", "tzpp")


def _triple_zeta_from_double(symbol: str) -> list:
    """Approximate TZ exponents for elements missing from the TZ table.

    Each subshell's double-zeta pair (zeta_in, zeta_out) gains a geometric-mean
    middle exponent sqrt(zeta_in * zeta_out).  Elements with only one exponent
    per subshell are passed through unchanged.
    """
    dz = sto_table("dz").get(symbol) or sto_table("sz").get(symbol, [])
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for n, l, m, z in dz:
        grouped[(n, l, m)].append(z)
    out = []
    for (n, l, m), zetas in grouped.items():
        if len(zetas) >= 2:
            z_in, z_out = max(zetas), min(zetas)
            for z in (z_in, (z_in * z_out) ** 0.5, z_out):
                out.append((n, l, m, z))
        else:
            out += [(n, l, m, z) for z in zetas]
    return out


def sto_exponents(basis_key: str) -> dict[str, list]:
    """{symbol: [(n, l, m, zeta), ...]} for one of the STO basis keys.

    ``sz``   single-zeta                  Clementi & Raimondi (1963)
    ``dz``   double-zeta                  Clementi & Roetti (1974)
    ``tz``   triple-zeta                  Clementi & Roetti (1974), extended
    ``*p``   adds d polarization
    ``*pp``  adds d and f polarization
    """
    key = basis_key.lower().strip()
    if key not in STO_KEYS:
        raise ConfigError(f"unknown STO basis '{basis_key}'; choose from {STO_KEYS}")

    if key.startswith("tz"):
        tz = sto_table("tz")
        base = {s: (tz[s] if s in tz else _triple_zeta_from_double(s)) for s in elem.symbols()}
    elif key.startswith("dz"):
        base = sto_table("dz")
    else:
        base = sto_table("sz")

    pol_d = sto_table("pol_d") if key.endswith(("p", "pp")) else {}
    pol_f = sto_table("pol_f") if key.endswith("pp") else {}

    sz = sto_table("sz")
    out: dict[str, list] = {}
    for sym in elem.symbols():
        funcs = list(base.get(sym) or sz.get(sym, []))
        funcs += list(pol_d.get(sym, []))
        funcs += list(pol_f.get(sym, []))
        out[sym] = funcs
    return out


def build_sto_basis(molecule, basis_key: str, pseudo_zeta=None) -> list[STO]:
    """One STO per tabulated exponent, for every atom of ``molecule``.

    ``pseudo_zeta`` maps a pseudo-hydrogen species label to the exponent derived
    for its fractional charge (see :mod:`pyxany.basis.pseudo_h`); the terminator
    gets this basis set's full hydrogen function list scaled by
    :func:`~pyxany.basis.pseudo_h.length_scale_ratio`.
    """
    table = sto_exponents(basis_key)
    labels = orbital_labels()
    pseudo_zeta = pseudo_zeta or {}
    functions: list[STO] = []

    for ai, atom in enumerate(molecule):
        if atom.is_pseudo:
            zeta = pseudo_zeta.get(atom.symbol)
            if zeta is None:
                raise KeyError(
                    f"no exponent available for pseudo-hydrogen species "
                    f"'{atom.symbol}'; it must be derived first"
                )
            # every hydrogen function of this basis, contracted by zeta(alpha)/zeta(1)
            from .pseudo_h import length_scale_ratio

            scale = length_scale_ratio(zeta)
            for n, l, m, z_h in table["H"]:
                name = labels.get((n, l, m), f"{n}{l}{m}")
                functions.append(
                    STO(n=n, l=l, m=m, zeta=float(z_h) * scale, atom_idx=ai,
                        label=f"{atom.symbol}{ai + 1}_{name}")
                )
            continue

        entries = table.get(atom.element)
        if not entries:
            raise KeyError(f"basis '{basis_key}' has no entry for element '{atom.element}'")
        for n, l, m, zeta in entries:
            name = labels.get((n, l, m), f"{n}{l}{m}")
            functions.append(
                STO(n=n, l=l, m=m, zeta=float(zeta), atom_idx=ai,
                    label=f"{atom.symbol}{ai + 1}_{name}")
            )
    return functions
