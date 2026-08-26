"""Electron-counting rule for pseudo-hydrogen terminators.

Reference
---------
H.-X. Deng, S.-S. Li, J. Li and S.-H. Wei, "Effect of hydrogen passivation on
the electronic structure of ionic semiconductor nanostructures",
*Phys. Rev. B* **85**, 195328 (2012), which prescribes

    "the fractional charge alpha = (8 - Z)/N for passivation of anion atoms,
     and alpha = (2M - Z)/M for passivation of cation atoms should be used.
     Here, N and M are the number of the bonds (coordination number) of the
     corresponding anions and cations, respectively."

so

    anion  terminator:  alpha = (8 - Z) / N
    cation terminator:  alpha = (2M - Z) / M  =  2 - Z/M

with ``Z`` the number of valence electrons of the atom being terminated.

**The charge depends on the coordination number as well as on the element.**
For tetrahedral coordination (N = M = 4) both expressions collapse to
``2 - Z/4``, but away from it they differ -- in rock salt (N = M = 6) an oxygen
terminator carries 1/3 while a magnesium terminator carries 5/3.

Worked values, all of which coincide with the pseudo-hydrogen charges shipped
with VASP:

===========  ===========  ==========================  ==========================
material     structure    anion  (Z, N) -> alpha      cation (Z, M) -> alpha
===========  ===========  ==========================  ==========================
Si, Ge, C    diamond      (4, 4) -> 1                 (4, 4) -> 1
SiC          zinc blende  C  (4, 4) -> 1              Si (4, 4) -> 1
GaAs, InP    zinc blende  As/P (5, 4) -> 3/4          Ga/In (3, 4) -> 5/4
GaN          wurtzite     N  (5, 4) -> 3/4            Ga (3, 4) -> 5/4
ZnO/ZnS/CdSe wurtzite/zb  O/S/Se (6, 4) -> 1/2        Zn/Cd (2, 4) -> 3/2
MgO          rock salt    O  (6, 6) -> 1/3            Mg (2, 6) -> 5/3
TiO2         rutile       O  (6, 3) -> 2/3            Ti (4, 6) -> 4/3
CuCl         zinc blende  Cl (7, 4) -> 1/4            Cu (1, 4) -> 7/4
===========  ===========  ==========================  ==========================
"""

from __future__ import annotations

from fractions import Fraction

from ..data import elements as elem

ANION = "anion"
CATION = "cation"


def terminator_charge(role: str, valence_electrons: int, coordination: int) -> float:
    """Fractional nuclear charge of the pseudo-hydrogen terminating one bond."""
    if coordination <= 0:
        raise ValueError(f"coordination number must be positive, got {coordination}")
    z = int(valence_electrons)
    if role == ANION:
        alpha = (8 - z) / coordination
    elif role == CATION:
        alpha = (2 * coordination - z) / coordination
    else:
        raise ValueError(f"role must be '{ANION}' or '{CATION}', got {role!r}")
    if alpha <= 0:
        raise ValueError(
            f"electron counting gives a non-positive charge (alpha = {alpha:g}) for a "
            f"{role} with {z} valence electrons and coordination {coordination}; "
            f"check the role assignment and the coordination number"
        )
    return alpha


def terminator_charge_for(symbol: str, role: str, coordination: int) -> float:
    """Terminator charge for an element, using its tabulated valence count."""
    return terminator_charge(role, elem.valence_electrons(symbol), coordination)


def as_fraction(alpha: float, max_denominator: int = 24) -> Fraction:
    """Nearest simple fraction, for reporting (``0.75`` -> ``3/4``)."""
    return Fraction(alpha).limit_denominator(max_denominator)


def assign_roles(molecule, neighbours: dict[int, list[int]],
                 overrides: dict[str, str] | None = None) -> dict[int, str]:
    """Label each atom anion or cation from its bonding environment.

    Within a compound the atom with **more valence electrons** is the anion:
    O(6) beats Ti(4), Se(6) beats Cd(2), As(5) beats Ga(3), Cl(7) beats Cu(1).
    An atom is compared against the average of its actual neighbours rather than
    against a global rule, so a mixed structure is handled site by site.

    Ties (an elemental solid, or SiC where both are group IV) are labelled
    cation; that choice is immaterial because both expressions agree whenever
    the two coordination numbers are equal, which is the only way a tie can
    arise in a binary compound.  Explicit ``overrides`` always win.
    """
    overrides = overrides or {}
    roles: dict[int, str] = {}
    for i, atom in enumerate(molecule):
        if atom.symbol in overrides:
            roles[i] = overrides[atom.symbol]
            continue
        if atom.is_pseudo:
            continue
        own = elem.valence_electrons(atom.element)
        others = [elem.valence_electrons(molecule[j].element)
                  for j in neighbours.get(i, []) if not molecule[j].is_pseudo]
        if not others:
            roles[i] = CATION if own <= 4 else ANION
            continue
        roles[i] = ANION if own > (sum(others) / len(others)) else CATION
    return roles
