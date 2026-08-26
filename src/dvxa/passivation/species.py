"""Pseudo-hydrogen species: the charge, the label and how it was derived."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

#: Fractional charges shipped as pseudo-hydrogen potentials with VASP.  The
#: electron-counting rule generates whatever charge a structure needs, and any
#: value works here because the basis function is solved for rather than
#: tabulated -- this set is used only to flag a charge as "standard".
VASP_CHARGES: tuple[Fraction, ...] = tuple(
    Fraction(x) for x in
    ("1/4", "1/3", "5/12", "1/2", "7/12", "2/3", "3/4",
     "5/4", "4/3", "3/2", "5/3", "7/4")
)


@dataclass
class PseudoSpecies:
    """One terminator species, e.g. ``HCd`` with alpha = 3/2."""

    label: str
    passivates: str          # element being terminated
    role: str                # anion | cation
    valence_electrons: int
    coordination: int
    alpha: float

    def is_vasp_standard(self, tol: float = 1e-6) -> bool:
        return any(abs(self.alpha - float(f)) < tol for f in VASP_CHARGES)

    def fraction(self) -> Fraction:
        return Fraction(self.alpha).limit_denominator(24)

    def to_yaml_entry(self) -> dict:
        return {
            "passivates": self.passivates,
            "role": self.role,
            "Z_val": self.valence_electrons,
            "coordination": self.coordination,
            "alpha": round(float(self.alpha), 8),
        }

    def describe(self) -> str:
        origin = ("(8 - Z)/N" if self.role == "anion" else "(2M - Z)/M")
        std = "VASP standard" if self.is_vasp_standard() else "non-standard charge"
        return (f"{self.label}: terminates {self.passivates} ({self.role}, "
                f"Z_val={self.valence_electrons}, coordination={self.coordination}) "
                f"-> alpha = {origin} = {self.fraction()} = {self.alpha:.6f}  [{std}]")


def label_for(element: str, prefix: str = "H") -> str:
    """Species label used in the XYZ file for a terminator of ``element``."""
    return f"{prefix}{element}"


def real_hydrogen_species(element: str, role: str, valence: int,
                          coordination: int) -> PseudoSpecies:
    """A plain hydrogen terminator (alpha = 1), for the PH-vs-RH comparison."""
    return PseudoSpecies(label="H", passivates=element, role=role,
                         valence_electrons=valence, coordination=coordination,
                         alpha=1.0)
