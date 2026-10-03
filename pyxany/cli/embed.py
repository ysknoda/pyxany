"""``pyxany embed`` -- point charges of the surrounding crystal (ADR 0010).

Reads a small YAML description of the crystal::

    lattice:            # conventional cell vectors, Angstrom (rows)
      - [4.21, 0, 0]
      - [0, 4.21, 0]
      - [0, 0, 4.21]
    sites:              # [label, fractional x, y, z, formal charge]
      - [Mg, 0.0, 0.0, 0.0, +2]
      - [O,  0.5, 0.0, 0.0, -2]
      ...
    origin: [0, 0, 0]   # Cartesian position (Angstrom) of fractional (0, 0, 0)
    cells: 3            # block 2 cells + 1 wide, centred on the origin
    cluster: mgo6.xyz   # quantum atoms; lattice sites on them are left out

and writes ``label x y z q`` lines for ``embedding.charges``. The block is
neutral by Evjen's construction (Phys. Rev. 39, 675 (1932)): sites on a face
of the block carry 1/2 of their charge, on an edge 1/4, on a corner 1/8, so a
neutral conventional cell gives a neutral, dipole-free block whose Madelung
potential converges quickly with ``cells``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..embedding import write_charges
from ..io.xyz import read_xyz
from ..parallel import mpi_print
from . import EXIT_INPUT, EXIT_OK


def evjen_charges(lattice, sites, cells: int, origin=(0.0, 0.0, 0.0), tol: float = 1e-6):
    """``(positions_ang, charges, labels)`` of an Evjen-weighted block of cells."""
    L = np.asarray(lattice, dtype=float)
    # 2 cells + 1 cells wide, centred on fractional (0, 0, 0): the cluster sees a
    # symmetric surrounding, and any integer width keeps Evjen's neutrality
    lo, hi = -cells - 0.5, cells + 0.5
    pos, q, lab = [], [], []
    rng = range(-cells - 1, cells + 2)
    for i in rng:
        for j in rng:
            for k in rng:
                for label, fx, fy, fz, charge in sites:
                    u = np.array([i + fx, j + fy, k + fz], dtype=float)
                    if np.any(u < lo - tol) or np.any(u > hi + tol):
                        continue
                    on_face = (np.abs(u - lo) < tol) | (np.abs(u - hi) < tol)
                    weight = 0.5 ** int(on_face.sum())
                    pos.append(u @ L + np.asarray(origin, dtype=float))
                    q.append(weight * float(charge))
                    lab.append(str(label))
    return np.asarray(pos), np.asarray(q), lab


def remove_cluster_sites(positions, charges, labels, cluster, tol: float = 0.1):
    """Drop lattice sites occupied by (non-terminator) cluster atoms.

    Returns the remaining charges and the formal charge of the removed sites,
    which the cluster must carry for the whole system to stay neutral.
    """
    keep = np.ones(len(charges), dtype=bool)
    removed = 0.0
    for atom in cluster:
        if atom.is_pseudo:
            continue
        d = np.linalg.norm(positions - atom.pos_angstrom, axis=1)
        hit = np.where(keep & (d < tol))[0]
        if len(hit) == 0:
            raise ValueError(
                f"cluster atom {atom.symbol} at {atom.pos_angstrom} is not on a "
                f"lattice site (within {tol} A); check origin and lattice"
            )
        keep[hit] = False
        removed += float(charges[hit].sum())
    return positions[keep], charges[keep], [lab for lab, k in zip(labels, keep) if k], removed


def execute(args) -> int:
    try:
        spec = yaml.safe_load(Path(args.input).read_text())
        base = Path(args.input).resolve().parent
        lattice, sites = spec["lattice"], spec["sites"]
        cells = int(spec.get("cells", 3))
        origin = spec.get("origin", [0.0, 0.0, 0.0])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        mpi_print(f"  [error] {args.input}: {exc}")
        return EXIT_INPUT

    pos, q, lab = evjen_charges(lattice, sites, cells, origin)
    cell_charge = sum(float(s[4]) for s in sites)
    removed = 0.0
    if spec.get("cluster"):
        path = Path(spec["cluster"])
        cluster = read_xyz(path if path.is_absolute() else base / path)
        try:
            pos, q, lab, removed = remove_cluster_sites(pos, q, lab, cluster)
        except ValueError as exc:
            mpi_print(f"  [error] {exc}")
            return EXIT_INPUT

    header = (
        f"Evjen block of {2 * cells + 1}^3 cells (+ faces) from {args.input}; "
        f"cluster sites removed carry {removed:+g}"
    )
    write_charges(args.output, pos, q, lab, header)
    mpi_print(f"  {len(q)} point charges written to {args.output}")
    mpi_print(
        f"  net charge of the point charges {q.sum():+.6f}; "
        f"of the conventional cell {cell_charge:+g}"
    )
    if spec.get("cluster"):
        mpi_print(
            f"  the lattice sites replaced by the cluster carry {removed:+g}: give the "
            f"cluster molecule.charge = {removed:+g} for overall neutrality"
        )
    mpi_print("  in the input file:")
    mpi_print(f"    embedding: {{charges: {args.output}, units: angstrom}}")
    return EXIT_OK
