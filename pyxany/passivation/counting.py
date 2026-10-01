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


#: Largest fractional charge still meaningfully a pseudo-*hydrogen*.  The VASP
#: set tops out at 7/4; 2 leaves a little room above it.
MAX_TERMINATOR_CHARGE = 2.0


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
    # A terminator with more nuclear charge than helium is not a pseudo-*hydrogen*
    # any more; the largest value in the VASP set is 7/4.  Reaching this almost
    # always means the role or the coordination number is wrong.
    if alpha > MAX_TERMINATOR_CHARGE:
        raise ValueError(
            f"electron counting gives alpha = {alpha:g} for a {role} with {z} "
            f"valence electrons and coordination {coordination}, above the "
            f"{MAX_TERMINATOR_CHARGE:g} ceiling: that is no longer a pseudo-hydrogen. "
            f"Check the role assignment and the coordination number."
        )
    return alpha


def terminator_charge_for(symbol: str, role: str, coordination: int) -> float:
    """Terminator charge for an element, using its tabulated valence count."""
    return terminator_charge(role, elem.valence_electrons(symbol), coordination)


def as_fraction(alpha: float, max_denominator: int = 24) -> Fraction:
    """Nearest simple fraction, for reporting (``0.75`` -> ``3/4``)."""
    return Fraction(alpha).limit_denominator(max_denominator)


def assign_roles(molecule, neighbours: dict[int, list[int]],
                 overrides: dict[str, str] | None = None,
                 ideal: dict[str, int] | None = None) -> dict[int, str]:
    """Label each atom anion or cation from its bonding environment.

    Within a compound the atom with **more valence electrons** is the anion:
    O(6) beats Ti(4), Se(6) beats Cd(2), As(5) beats Ga(3), Cl(7) beats Cu(1).
    An atom is compared against the average of its actual neighbours rather than
    against a global rule, so a mixed structure is handled site by site.

    This is a proxy for electronegativity, and it has two known failure modes.
    Rather than guess in either, this raises and asks for an explicit role --
    a wrong role gives a terminator the wrong fractional charge, which is
    exactly the unsaturated or over-saturated boundary that passivation exists
    to prevent, and it is not visible in the electron-balance check (a
    symmetric role inversion in a stoichiometric AB compound changes the total
    electron count by exactly zero).

    **Ties.**  The two expressions agree only at coordination 4, where both
    reduce to ``2 - Z/4`` -- *not*, as an earlier version of this docstring
    claimed, whenever the two coordination numbers are equal.  Rock-salt TiC
    (both group IV, both 6-coordinate) gives 2/3 as an anion and 4/3 as a
    cation, a factor of two.  So a tie between *different* elements raises
    unless the ideal coordination is 4.  A tie among atoms of the *same*
    element is different: an elemental solid has no anion/cation distinction to
    get wrong, and its terminator simply replaces a like neighbour, so it is
    resolved silently as a cation.  ``ideal`` maps an element to the
    coordination the charge will actually be computed with -- pass it, or the
    observed neighbour count is used, which undercounts at a surface site.

    **Transition metals.**  ``valence_electrons`` returns the group number for
    groups 3-10, so Mn(7), Fe(8) and Ni(10) all outscore O(6) and would be
    labelled anions in their own oxides.  A d-block element is never the anion
    against a main-group partner, so that combination raises too.

    Explicit ``overrides`` (``{"Mn": "cation"}``) always win and bypass both
    checks.
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
        partners = [j for j in neighbours.get(i, []) if not molecule[j].is_pseudo]
        others = [elem.valence_electrons(molecule[j].element) for j in partners]
        if not others:
            roles[i] = CATION if own <= 4 else ANION
            continue

        average = sum(others) / len(others)
        role = ANION if own > average else CATION

        elemental = all(molecule[j].element == atom.element for j in partners)
        n_ideal = (ideal or {}).get(atom.element, len(partners))
        if own == average and not elemental and n_ideal != 4:
            raise ValueError(
                f"cannot tell whether {atom.element} (site {i}) is the anion or the "
                f"cation: it has the same valence-electron count as its neighbours "
                f"({own}), at coordination {n_ideal}. The two counting rules only "
                f"coincide at coordination 4, so the choice changes the terminator "
                f"charge here ((8-Z)/N = {(8 - own) / n_ideal:.4g} as an anion vs "
                f"(2M-Z)/M = {(2 * n_ideal - own) / n_ideal:.4g} as a cation). "
                f"Set it explicitly, e.g. role_overrides="
                f"{{'{atom.element}': 'anion'}}."
            )

        if role == ANION and 3 <= elem.group(atom.element) <= 12:
            main_group = [molecule[j].element for j in partners
                          if not 3 <= elem.group(molecule[j].element) <= 12]
            if main_group:
                raise ValueError(
                    f"{atom.element} (site {i}) came out as the anion against "
                    f"{sorted(set(main_group))}, which cannot be right: a d-block "
                    f"element is not the anion against a main-group partner. The "
                    f"valence-electron count used to rank them is the group number "
                    f"for groups 3-10, so {atom.element}({own}) outscores its "
                    f"neighbours. Set it explicitly: role_overrides="
                    f"{{'{atom.element}': 'cation'}}."
                )
        roles[i] = role
    return roles
