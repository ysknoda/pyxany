"""Point-charge (Madelung) embedding of a cluster (ADR 0010).

The cluster sits in the electrostatic field of fixed point charges ``q_i`` at
``R_i`` -- typically the ions of the surrounding crystal. Each charge is a
normalized Gaussian of width ``sigma``, so the potential energy of an electron
is finite everywhere on the grid:

    v_ext(r) = - sum_i q_i erf(|r - R_i| / sigma) / |r - R_i|

and the energy of nucleus A is ``Z_A * (-v_ext(R_A))``. The charges do not
move; the forces on the cluster atoms contain the nuclear term here plus the
electronic one, which enters through the potential like the nuclear
attraction does.

Charges are read from a text file: one charge per line, ``x y z q`` or
``label x y z q`` (label ignored), ``#`` comments allowed, in the units given
by ``embedding.units``. ``pyxany embed`` writes such files from a crystal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erf

from .constants import ANG_TO_BOHR

_TWO_OVER_SQRT_PI = 2.0 / np.sqrt(np.pi)


@dataclass
class PointCharges:
    positions: np.ndarray  # (n, 3) Bohr
    charges: np.ndarray  # (n,)
    sigma: float  # Bohr

    def __len__(self) -> int:
        return len(self.charges)

    @property
    def total_charge(self) -> float:
        return float(self.charges.sum())

    def potential(self, points: np.ndarray, block: int = 200_000) -> np.ndarray:
        """Potential energy of an electron at ``points`` (Hartree)."""
        from .kernels import point_charge_field

        fast = point_charge_field(points, self.positions, self.charges, self.sigma)
        if fast is not None:
            return fast
        pts = np.asarray(points, dtype=float)
        out = np.zeros(len(pts))
        for lo in range(0, len(pts), block):
            p = pts[lo : lo + block]
            for R, q in zip(self.positions, self.charges, strict=True):
                d = np.linalg.norm(p - R, axis=1)
                out[lo : lo + block] -= q * _erf_over_r(d, self.sigma)
        return out

    def gradient(self, points: np.ndarray, block: int = 200_000) -> np.ndarray:
        """``grad v_ext`` at ``points``, shape (n, 3)."""
        from .kernels import point_charge_field

        fast = point_charge_field(points, self.positions, self.charges, self.sigma,
                                  gradient=True)
        if fast is not None:
            return fast[1]
        pts = np.asarray(points, dtype=float)
        out = np.zeros((len(pts), 3))
        for lo in range(0, len(pts), block):
            p = pts[lo : lo + block]
            for R, q in zip(self.positions, self.charges, strict=True):
                diff = p - R
                d = np.linalg.norm(diff, axis=1)
                out[lo : lo + block] -= (
                    q * (_d_erf_over_r(d, self.sigma) / np.maximum(d, 1e-300))[:, None] * diff
                )
        return out

    def nuclear_energy(self, molecule) -> float:
        """Interaction of the cluster nuclei with the charges."""
        return float(-np.dot(molecule.charges, self.potential(molecule.positions)))

    def nuclear_gradient(self, molecule) -> np.ndarray:
        """``d E_nuc-charges / d R_A``, shape (n_atoms, 3)."""
        return -np.asarray(molecule.charges)[:, None] * self.gradient(molecule.positions)

    def electrons_near(self, points, weights, rho, radius: float, positive_only: bool = True):
        """Electrons within ``radius`` (Bohr) of any (positive) charge."""
        sel = self.charges > 0 if positive_only else np.ones(len(self), dtype=bool)
        if not sel.any():
            return 0.0
        near = np.zeros(len(points), dtype=bool)
        for R in self.positions[sel]:
            near |= np.linalg.norm(points - R, axis=1) < radius
        return float(np.dot(weights[near], rho[near]))


def _erf_over_r(d: np.ndarray, sigma: float) -> np.ndarray:
    """``erf(d / sigma) / d``, with its limit ``2 / (sqrt(pi) sigma)`` at d = 0."""
    small = d < 1e-8 * sigma
    safe = np.where(small, 1.0, d)
    return np.where(small, _TWO_OVER_SQRT_PI / sigma, erf(safe / sigma) / safe)


def _d_erf_over_r(d: np.ndarray, sigma: float) -> np.ndarray:
    """``d/dd [erf(d / sigma) / d]`` (zero at d = 0 by symmetry)."""
    small = d < 1e-8 * sigma
    safe = np.where(small, 1.0, d)
    val = (
        _TWO_OVER_SQRT_PI / sigma * np.exp(-((safe / sigma) ** 2)) - erf(safe / sigma) / safe
    ) / safe
    return np.where(small, 0.0, val)


def read_charges(path, units: str = "angstrom") -> tuple[np.ndarray, np.ndarray]:
    """``(positions in Bohr, charges)`` from an ``x y z q`` / ``label x y z q`` file."""
    pos, q = [], []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        try:
            if len(fields) == 4:
                values = [float(x) for x in fields]
            elif len(fields) == 5:
                values = [float(x) for x in fields[1:]]
            else:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"{path}:{number}: expected 'x y z q' or 'label x y z q', got {line!r}"
            ) from None
        pos.append(values[:3])
        q.append(values[3])
    if not q:
        raise ValueError(f"{path}: no point charges found")
    scale = ANG_TO_BOHR if units == "angstrom" else 1.0
    return np.asarray(pos, dtype=float) * scale, np.asarray(q, dtype=float)


def from_config(config) -> PointCharges | None:
    """The point charges an input file asks for, or None (cached on the config)."""
    cached = getattr(config, "_embedding_charges", "unset")
    if cached != "unset":
        return cached
    emb = config.embedding
    out = None
    if emb.charges:
        path = Path(emb.charges)
        if not path.is_absolute():
            path = config.source_dir / path
        pos, q = read_charges(path, emb.units)
        out = PointCharges(pos, q, float(emb.sigma))  # sigma is always in Bohr
    config._embedding_charges = out
    return out


def write_charges(path, positions_ang, charges, labels=None, header: str = "") -> None:
    """Write an ``x y z q`` file (Angstrom) that :func:`read_charges` reads back."""
    lines = [f"# {header}"] if header else []
    lines.append(
        f"# {len(charges)} point charges, total {float(np.sum(charges)):+.6f}; x y z (Angstrom) q"
    )
    for i, (p, q) in enumerate(zip(positions_ang, charges, strict=True)):
        lab = labels[i] if labels is not None else "Q"
        lines.append(f"{lab:<4s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f} {q:+12.8f}")
    Path(path).write_text("\n".join(lines) + "\n")
