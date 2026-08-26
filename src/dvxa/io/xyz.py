"""XYZ structure files, including pseudo-hydrogen species labels.

Standard format::

    <n_atoms>
    <comment>
    <symbol>  <x>  <y>  <z>
    ...

Species labels need not be element symbols: a pseudo-hydrogen terminator is
written as its own species (``HCd``, ``HO``, ...) and must be declared in the
``passivation.species`` block of the YAML input.  Reading is strict -- an
undeclared, non-element label is an error rather than a silent skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import ANG_TO_BOHR, BOHR_TO_ANG
from ..data import elements as elem
from ..molecule import Atom, Molecule


class XYZError(ValueError):
    """Raised for a malformed or inconsistent .xyz file."""


def read_xyz(path, units: str = "angstrom", species: dict | None = None,
             name: str | None = None, charge: int = 0, multiplicity: int = 1,
             frame: int = 0) -> Molecule:
    """Read one frame of an .xyz file into a :class:`~dvxa.molecule.Molecule`.

    Parameters
    ----------
    species : optional ``{label: {"alpha": float, ...}}`` declaring pseudo-atom
        species that are not element symbols.
    frame : index of the frame to read from a multi-frame trajectory file.
    """
    path = Path(path)
    frames = _parse_frames(path)
    if not frames:
        raise XYZError(f"{path}: file contains no frames")
    if frame >= len(frames):
        raise XYZError(f"{path}: requested frame {frame} but the file has {len(frames)}")
    comment, rows = frames[frame]

    scale = ANG_TO_BOHR if units == "angstrom" else 1.0
    species = species or {}
    atoms = []
    for line_no, (label, x, y, z) in rows:
        pos = np.array([x, y, z]) * scale
        if label in species:
            spec = species[label]
            alpha = float(spec.get("alpha", spec.get("Z")))
            atoms.append(Atom.pseudo_hydrogen(label, alpha, pos))
        elif elem.is_supported(label):
            atoms.append(Atom.from_element(label, pos))
        else:
            declared = ", ".join(sorted(species)) or "(none)"
            raise XYZError(
                f"{path}:{line_no}: '{label}' is neither a supported element nor a "
                f"declared pseudo-atom species. Declared species: {declared}. "
                f"Supported elements: {', '.join(elem.symbols())}"
            )
    return Molecule(atoms=atoms, charge=charge, multiplicity=multiplicity,
                    name=name or comment.strip() or path.stem)


def _parse_frames(path: Path):
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise XYZError(f"structure file not found: {path}") from None
    lines = text.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].split()[0])
        except (ValueError, IndexError):
            raise XYZError(
                f"{path}:{i + 1}: expected an atom count, got {lines[i]!r}"
            ) from None
        if i + 1 + n >= len(lines) + 1 and i + 2 + n > len(lines):
            raise XYZError(f"{path}:{i + 1}: header says {n} atoms but the file ends early")
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        rows = []
        for k in range(n):
            ln = i + 2 + k
            parts = lines[ln].split()
            if len(parts) < 4:
                raise XYZError(
                    f"{path}:{ln + 1}: expected 'symbol x y z', got {lines[ln]!r}")
            try:
                xyz = tuple(float(v) for v in parts[1:4])
            except ValueError:
                raise XYZError(
                    f"{path}:{ln + 1}: coordinates are not numbers: {lines[ln]!r}"
                ) from None
            rows.append((ln + 1, (parts[0], *xyz)))
        frames.append((comment, rows))
        i += 2 + n
    return frames


def n_frames(path) -> int:
    return len(_parse_frames(Path(path)))


def write_xyz(path, molecule: Molecule, comment: str = "", mode: str = "w") -> None:
    """Write a molecule as a single .xyz frame (Angstrom)."""
    lines = [str(len(molecule)), comment or molecule.summary()]
    for atom in molecule:
        x, y, z = atom.pos * BOHR_TO_ANG
        lines.append(f"{atom.symbol:<6s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
    with Path(path).open(mode) as fh:
        fh.write("\n".join(lines) + "\n")


def append_frame(path, molecule: Molecule, comment: str = "") -> None:
    """Append a frame to a trajectory file."""
    write_xyz(path, molecule, comment, mode="a")
