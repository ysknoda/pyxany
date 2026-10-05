"""Contracted Gaussian basis-set construction."""

from __future__ import annotations

from ..data.loaders import gto_polarization_exponents, gto_table
from .harmonics import lm_pairs
from .types import GTO

#: Recognized GTO basis keys, mapped to (table name, add d polarization).
GTO_KEYS = {
    "sto3g": ("sto3g", False),
    "631g": ("631g", False),
    "6-31g": ("631g", False),
    "631gd": ("631g", True),
    "6-31g*": ("631g", True),
    "6-31g(d)": ("631g", True),
}

_SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3}


def _shell_angular_momentum(label: str) -> str:
    """Classify a table key such as '2sp_core', '1s_val', '3d' by shell type."""
    stem = label.split("_")[0].lower()
    letters = [c for c in stem if c in _SHELL_L]
    if not letters:
        raise ValueError(f"cannot classify GTO shell label '{label}'")
    return "sp" if letters == ["s", "p"] else letters[-1]


def _expand_shells(shells: dict) -> list[tuple[int, int, tuple]]:
    """Flatten {shell_label: primitives} into [(l, m, prims), ...].

    ``sp`` shells carry two coefficient columns and are split into an s shell
    and three p shells.
    """
    out: list[tuple[int, int, tuple]] = []
    for label, prims in shells.items():
        if not prims:
            continue
        kind = _shell_angular_momentum(label)
        if kind == "sp":
            s_prims = tuple((p[0], p[1]) for p in prims)
            p_prims = tuple((p[0], p[2]) for p in prims) if len(prims[0]) >= 3 else s_prims
            out.append((0, 0, s_prims))
            for m in lm_pairs(1):
                out.append((1, m, p_prims))
        else:
            l = _SHELL_L[kind]
            base = tuple((p[0], p[1]) for p in prims)
            for m in lm_pairs(l):
                out.append((l, m, base))
    return out


def build_gto_basis(molecule, basis_key: str, pseudo_zeta=None) -> list:
    """Contracted GTOs for every atom; pseudo-hydrogens get scaled hydrogen shells."""
    key = basis_key.lower().strip()
    if key not in GTO_KEYS:
        raise ValueError(f"unknown GTO basis '{basis_key}'; choose from {sorted(GTO_KEYS)}")
    table_name, want_polarization = GTO_KEYS[key]
    table = gto_table(table_name)
    pol = gto_polarization_exponents()
    pseudo_zeta = pseudo_zeta or {}

    functions: list = []
    for ai, atom in enumerate(molecule):
        if atom.is_pseudo:
            from .pseudo_h import length_scale_ratio

            zeta = pseudo_zeta.get(atom.symbol)
            if zeta is None:
                raise KeyError(f"no exponent derived for pseudo-hydrogen '{atom.symbol}'")
            # this basis set's hydrogen shells, contracted by zeta(alpha)/zeta(1):
            # a Gaussian exponent scales with the inverse length squared
            scale2 = length_scale_ratio(zeta) ** 2
            counts: dict[tuple[int, int], int] = {}
            for l, m, prims in _expand_shells(table["H"]):
                counts[(l, m)] = counts.get((l, m), 0) + 1
                tag = "spdf"[l] + (f"{m:+d}" if l else "")
                scaled = tuple((a * scale2, c) for a, c in prims)
                functions.append(GTO.contracted(l, m, scaled, ai,
                                                f"{atom.symbol}{ai + 1}_{counts[(l, m)]}{tag}"))
            continue

        shells = table.get(atom.element)
        if shells is None:
            raise KeyError(
                f"GTO basis '{basis_key}' does not cover element '{atom.element}'. "
                f"Covered: {', '.join(sorted(table))}. Use an STO basis instead."
            )
        counts: dict[tuple[int, int], int] = {}
        for l, m, prims in _expand_shells(shells):
            counts[(l, m)] = counts.get((l, m), 0) + 1
            tag = "spdf"[l] + (f"{m:+d}" if l else "")
            functions.append(
                GTO.contracted(l, m, prims, ai,
                               f"{atom.symbol}{ai + 1}_{counts[(l, m)]}{tag}")
            )

        # 6-31G(d): one uncontracted d shell per non-hydrogen atom
        if want_polarization and atom.element != "H":
            alpha = pol.get(atom.element)
            if alpha is not None:
                for m in lm_pairs(2):
                    functions.append(
                        GTO.contracted(2, m, [(alpha, 1.0)], ai,
                                       f"{atom.symbol}{ai + 1}_pol_d{m:+d}")
                    )
    return functions
