"""Covalent radii, used for bonding cutoffs and pseudo-hydrogen placement.

Source: B. Cordero et al., "Covalent radii revisited",
*Dalton Trans.* (2008) 2832, doi:10.1039/B801115J -- the full 118-element table,
extracted by ``tools/extract_radii.py``.
"""

from __future__ import annotations

from functools import lru_cache

from .loaders import _load

#: Fallback for anything missing from the table (Angstrom).
DEFAULT_RADIUS = 1.5


@lru_cache(maxsize=1)
def covalent_radii() -> dict[str, float]:
    return dict(_load("covalent_radii")["radii"])


@lru_cache(maxsize=1)
def source() -> str:
    return _load("covalent_radii")["_source"]


def covalent_radius(symbol: str) -> float:
    """Covalent radius in Angstrom."""
    return covalent_radii().get(symbol, DEFAULT_RADIUS)


def bond_cutoff(symbol_a: str, symbol_b: str, tolerance: float = 1.20) -> float:
    """Distance below which A and B count as bonded, in Angstrom.

    Combination rule: **additive**, ``tolerance * (r_cov(A) + r_cov(B))``.
    The default 20 % slack accommodates the spread of real bond lengths around
    the sum of covalent radii.  Callers are expected to report the combined
    value alongside the observed distances so the choice can be checked against
    the actual structure rather than trusted blindly.
    """
    return tolerance * (covalent_radius(symbol_a) + covalent_radius(symbol_b))
