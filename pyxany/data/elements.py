"""Element properties: Z, occupations, grid radii, group, valence count, multiplicity.

Everything here is either read from the extracted JSON tables or *derived* from
the periodic-table structure -- no hand-typed per-element numbers beyond the
group boundaries, so there is nothing to mistype for 86 elements.
"""

from __future__ import annotations

from functools import cache

from .loaders import element_table

# --------------------------------------------------------------------------
# Group assignment, generated from the periodic-table block boundaries.
# --------------------------------------------------------------------------
# (Z_first, Z_last, group) for the contiguous runs of every period.  The
# lanthanides (58-71) are the f-block; they are given group 3 by the usual
# convention and flagged by is_f_block().
_GROUP_RUNS: list[tuple[int, int, int]] = [
    (1, 1, 1), (2, 2, 18),                                     # period 1
    (3, 3, 1), (4, 4, 2), (5, 10, None),                       # period 2 (B..Ne -> 13..18)
    (11, 11, 1), (12, 12, 2), (13, 18, None),                  # period 3 (Al..Ar)
    (19, 19, 1), (20, 20, 2), (21, 30, None), (31, 36, None),  # period 4 (Sc..Zn, Ga..Kr)
    (37, 37, 1), (38, 38, 2), (39, 48, None), (49, 54, None),  # period 5
    (55, 55, 1), (56, 56, 2), (57, 57, 3), (58, 71, 3),        # period 6 (La, lanthanides)
    (72, 80, None), (81, 86, None),                            # Hf..Hg, Tl..Rn
]


@cache
def _z_to_group() -> dict[int, int]:
    g: dict[int, int] = {}
    for z0, z1, fixed in _GROUP_RUNS:
        for z in range(z0, z1 + 1):
            if fixed is not None:
                g[z] = fixed
            elif z0 in (5, 13):        # p-block of periods 2-3: B..Ne, Al..Ar
                g[z] = 13 + (z - z0)
            elif z0 in (21, 39):       # d-block: Sc..Zn, Y..Cd
                g[z] = 3 + (z - z0)
            elif z0 == 72:             # d-block period 6: Hf..Hg
                g[z] = 4 + (z - z0)
            else:                      # p-block of periods 4-6: Ga.., In.., Tl..
                g[z] = 13 + (z - z0)
    return g


@cache
def _symbol_by_z() -> dict[int, str]:
    return {rec["Z"]: sym for sym, rec in element_table().items()}


def symbols() -> list[str]:
    """All supported element symbols, ordered by atomic number."""
    return [s for _, s in sorted(_symbol_by_z().items())]


def is_supported(symbol: str) -> bool:
    return symbol in element_table()


def atomic_number(symbol: str) -> int:
    _require(symbol)
    return int(element_table()[symbol]["Z"])


def symbol_for_z(z: int) -> str:
    return _symbol_by_z()[int(z)]


def occupations(symbol: str) -> dict[str, float]:
    """Spherically-averaged aufbau occupations, {orbital_label: electrons}."""
    _require(symbol)
    return dict(element_table()[symbol]["occ"])


def bragg_slater_radius(symbol: str) -> float:
    """Bragg-Slater radius in Bohr, used to scale the atomic radial grid.

    Source: Slater (1964) J. Chem. Phys. 41, 3199.
    """
    _require(symbol)
    return float(element_table()[symbol]["bragg_slater_bohr"])


def group(symbol: str) -> int:
    """Periodic-table group (1-18)."""
    return _z_to_group()[atomic_number(symbol)]


def is_f_block(symbol: str) -> bool:
    return 58 <= atomic_number(symbol) <= 71


def valence_electrons(symbol: str) -> int:
    """Number of electrons available for bonding, in the electron-counting sense.

    Groups 1-2      -> 1, 2
    Groups 3-10     -> the group number (Sc 3, Ti 4, ... Ni 10)
    Group 11 (Cu..) -> 1     d10 s1: the filled d shell is core-like
    Group 12 (Zn..) -> 2     d10 s2
    Groups 13-18    -> group - 10  (B 3, C 4, N 5, O 6, F 7, Ne 8)

    This is the ``Z`` that enters the pseudo-hydrogen electron-counting rule
    (pyxany.passivation.counting); it reproduces the standard values Zn/Cd -> 2,
    Ga/In -> 3, Si -> 4, P/As -> 5, S/Se -> 6, Cl -> 7, Cu -> 1, Ti -> 4.
    """
    g = group(symbol)
    if g <= 2:
        return g
    if g == 11:
        return 1
    if g == 12:
        return 2
    if g <= 10:
        return g
    return g - 10


def _subshell_capacity(label: str) -> int:
    return 2 * (2 * _subshell_l(label) + 1)


def _subshell_l(label: str) -> int:
    """Angular momentum from an orbital label such as '3dxz' -> 2."""
    if "f" in label:
        return 3
    if "d" in label:
        return 2
    if "p" in label:
        return 1
    return 0


@cache
def _subshell_totals(symbol: str) -> dict[tuple[int, int], float]:
    """{(n, l): total electrons} collapsed from the per-m occupation labels."""
    totals: dict[tuple[int, int], float] = {}
    for label, ne in occupations(symbol).items():
        l = _subshell_l(label)
        n = next((int(c) for c in label if c.isdigit()), 1)
        totals[(n, l)] = totals.get((n, l), 0.0) + float(ne)
    return totals


def ground_state_multiplicity(symbol: str) -> int:
    """Spin multiplicity 2S+1 of the neutral atom, from Hund's first rule.

    Derived from the tabulated subshell occupations: a subshell of capacity
    2(2l+1) holding q electrons has ``q`` unpaired electrons while
    q <= 2l+1, and ``capacity - q`` beyond half filling.

    Exact for main-group elements (C -> 3, N -> 4, O -> 3, F -> 2, Ne -> 1).
    For d- and f-block elements the s/d configuration exceptions (Cr, Cu, Pd,
    Nb, ...) are not captured; use :func:`multiplicity_is_reliable` to check.
    """
    unpaired = 0.0
    for (_n, l), q in _subshell_totals(symbol).items():
        cap = 2 * (2 * l + 1)
        half = cap / 2
        unpaired += q if q <= half else cap - q
    return int(round(unpaired)) + 1


def multiplicity_is_reliable(symbol: str) -> bool:
    """False for d/f-block elements, where aufbau exceptions are common."""
    return group(symbol) <= 2 or group(symbol) >= 13


def _require(symbol: str) -> None:
    if symbol not in element_table():
        raise KeyError(
            f"Element '{symbol}' is not in the element database. "
            f"Supported: {', '.join(symbols())}"
        )
