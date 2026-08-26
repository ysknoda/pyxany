"""Atoms and molecules.

Positions are stored in Bohr throughout the package; conversion happens only at
the I/O boundary.  ``Atom.Z`` is a *float* so that pseudo-hydrogen terminators
carrying a fractional nuclear charge (see :mod:`dvxa.passivation`) are ordinary
atoms everywhere else in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import BOHR_TO_ANG
from .data import elements as elem


@dataclass
class Atom:
    """One nucleus plus the bookkeeping needed to find its basis and grid.

    Attributes
    ----------
    symbol
        Label as it appears in the XYZ file.  For a real element this is the
        element symbol; for a pseudo-hydrogen it is a user-chosen species tag
        such as ``"HCd"``.
    element
        The element whose data tables (basis, occupations, radii) apply.  Equal
        to ``symbol`` for real atoms; ``"H"`` for pseudo-hydrogens.
    Z
        Nuclear charge.  Integer-valued for real elements, fractional for
        pseudo-hydrogens.
    pos
        Cartesian position in Bohr.
    """

    symbol: str
    element: str
    Z: float
    pos: np.ndarray
    is_pseudo: bool = False
    basis: list = field(default_factory=list, repr=False)

    @classmethod
    def from_element(cls, symbol: str, pos_bohr) -> "Atom":
        return cls(symbol=symbol, element=symbol, Z=float(elem.atomic_number(symbol)),
                   pos=np.asarray(pos_bohr, dtype=float))

    @classmethod
    def pseudo_hydrogen(cls, label: str, alpha: float, pos_bohr) -> "Atom":
        return cls(symbol=label, element="H", Z=float(alpha),
                   pos=np.asarray(pos_bohr, dtype=float), is_pseudo=True)

    @property
    def pos_angstrom(self) -> np.ndarray:
        return self.pos * BOHR_TO_ANG

    def grid_radius(self) -> float:
        """Bragg-Slater radius (Bohr) used to scale this atom's radial grid.

        For a pseudo-hydrogen the hydrogen radius is scaled hydrogenically as
        ``R_H / alpha``: an alpha < 1 terminator is more diffuse than hydrogen
        and an alpha > 1 one more compact.  Validated by the grid-normalisation
        test rather than assumed (tests/test_grid.py).
        """
        r_h = elem.bragg_slater_radius(self.element)
        if not self.is_pseudo:
            return r_h
        return float(np.clip(r_h / max(self.Z, 1e-3), 0.25 * r_h, 4.0 * r_h))


@dataclass
class Molecule:
    """A list of atoms plus charge / multiplicity."""

    atoms: list[Atom]
    charge: int = 0
    multiplicity: int = 1
    name: str = "molecule"

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self):
        return iter(self.atoms)

    def __getitem__(self, i):
        return self.atoms[i]

    @property
    def positions(self) -> np.ndarray:
        """(n_atoms, 3) array of positions in Bohr."""
        return np.array([a.pos for a in self.atoms], dtype=float)

    @positions.setter
    def positions(self, new: np.ndarray) -> None:
        new = np.asarray(new, dtype=float).reshape(len(self.atoms), 3)
        for a, p in zip(self.atoms, new):
            a.pos = p.copy()

    @property
    def charges(self) -> np.ndarray:
        """(n_atoms,) nuclear charges, float (fractional for pseudo-hydrogens)."""
        return np.array([a.Z for a in self.atoms], dtype=float)

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.atoms]

    @property
    def n_electrons(self) -> float:
        """Total electron count = sum of nuclear charges minus the net charge."""
        return float(self.charges.sum()) - self.charge

    def has_fractional_electrons(self, tol: float = 1e-8) -> bool:
        n = self.n_electrons
        return abs(n - round(n)) > tol

    def nuclear_repulsion(self) -> float:
        """E_nn = sum_{A<B} Z_A Z_B / R_AB  (Hartree)."""
        pos = self.positions
        Z = self.charges
        n = len(Z)
        if n < 2:
            return 0.0
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        iu = np.triu_indices(n, k=1)
        return float(np.sum(Z[iu[0]] * Z[iu[1]] / d[iu]))

    def distance_matrix(self) -> np.ndarray:
        pos = self.positions
        return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    def centre_of_charge(self) -> np.ndarray:
        Z = self.charges
        return (self.positions * Z[:, None]).sum(axis=0) / Z.sum()

    def copy(self) -> "Molecule":
        return Molecule(
            atoms=[Atom(a.symbol, a.element, a.Z, a.pos.copy(), a.is_pseudo) for a in self.atoms],
            charge=self.charge, multiplicity=self.multiplicity, name=self.name,
        )

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for a in self.atoms:
            counts[a.symbol] = counts.get(a.symbol, 0) + 1
        formula = " ".join(f"{s}{n}" if n > 1 else s for s, n in counts.items())
        return (f"{self.name}: {formula}  ({len(self)} atoms, "
                f"charge {self.charge:+d}, mult {self.multiplicity}, "
                f"{self.n_electrons:g} electrons)")
