"""``dvxa scan`` -- energy along a bond length or angle."""

from __future__ import annotations

import csv

import numpy as np

from ..basis import build_basis
from ..constants import ANG_TO_BOHR, BOHR_TO_ANG
from ..parallel import is_root, mpi_print
from ..scf import run_scf
from ..timing import REGISTRY, timer
from . import EXIT_INPUT, EXIT_OK
from ._shared import out_path, prepare, report_timing


def execute(args) -> int:
    REGISTRY.reset()
    with timer("total"):
        cfg, molecule = prepare(args)
        kind, indices = _parse_coord(args.coord, len(molecule))
        values = np.linspace(args.range[0], args.range[1], args.steps)
        unit = "Angstrom" if kind == "bond" else "degrees"

        mpi_print(f"  SCAN  {kind} {indices}  from {values[0]:g} to {values[-1]:g} {unit}"
                  f"  ({args.steps} points)")
        mpi_print(f"  {'value':>12} {'E (Ha)':>18} {'dE (kcal/mol)':>15} {'converged':>10}")
        mpi_print("  " + "-" * 60)

        rows = []
        for v in values:
            trial = molecule.copy()
            _apply(trial, kind, indices, float(v))
            basis = build_basis(trial, cfg, getattr(cfg, "_pseudo_zeta", None))
            res = run_scf(trial, basis, cfg, verbose=0)
            rows.append({"value": float(v), "energy_hartree": res.total_energy,
                         "converged": bool(res.converged),
                         "n_iterations": res.n_iterations})
            e0 = rows[0]["energy_hartree"]
            mpi_print(f"  {v:>12.5f} {res.total_energy:>18.9f} "
                      f"{(res.total_energy - e0) * 627.5095:>15.4f} "
                      f"{str(res.converged):>10}")

        best = min(rows, key=lambda r: r["energy_hartree"])
        mpi_print("  " + "-" * 60)
        mpi_print(f"  minimum at {best['value']:.5f} {unit}   "
                  f"E = {best['energy_hartree']:.9f} Ha")

        if is_root():
            path = out_path(cfg, "_scan.csv")
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            mpi_print(f"  written to {path}")
    report_timing(cfg)
    return EXIT_OK


def _parse_coord(spec: str, n_atoms: int):
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    if kind not in ("bond", "angle"):
        raise ValueError(f"--coord: expected 'bond:i,j' or 'angle:i,j,k', got {spec!r}")
    indices = [int(t) for t in rest.split(",")]
    want = 2 if kind == "bond" else 3
    if len(indices) != want or any(not 0 <= i < n_atoms for i in indices):
        raise ValueError(f"--coord {spec!r}: need {want} atom indices in 0..{n_atoms - 1}")
    return kind, indices


def _apply(molecule, kind: str, indices: list[int], value: float) -> None:
    pos = molecule.positions
    if kind == "bond":
        i, j = indices
        d = pos[j] - pos[i]
        unit = d / np.linalg.norm(d)
        pos[j] = pos[i] + unit * value * ANG_TO_BOHR
    else:
        i, j, k = indices
        v1 = pos[i] - pos[j]
        v2 = pos[k] - pos[j]
        axis = np.cross(v1, v2)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / norm
        current = np.degrees(np.arccos(np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        pos[k] = pos[j] + _rotate(v2, axis, np.radians(value - current))
    molecule.positions = pos


def _rotate(v, axis, angle):
    """Rodrigues rotation of ``v`` about ``axis`` by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)
