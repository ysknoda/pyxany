"""``pyxany scan`` -- energy along a bond length or angle."""

from __future__ import annotations

import csv

import numpy as np

from ..basis import build_basis
from ..config import ConfigError
from ..constants import ANG_TO_BOHR, EV_TO_KCAL_MOL, HARTREE_TO_EV
from ..parallel import is_root, mpi_print
from ..scf import run_scf
from ..timing import REGISTRY, timer
from . import EXIT_OK, EXIT_SCF
from ._shared import out_path, prepare, report_timing


def execute(args) -> int:
    REGISTRY.reset()
    with timer("total"):
        cfg, molecule = prepare(args)
        kind, indices = _parse_coord(args.coord, len(molecule))
        values = np.linspace(args.range[0], args.range[1], args.steps)
        _check_range(kind, values)
        unit = "Angstrom" if kind == "bond" else "degrees"

        mpi_print(f"  SCAN  {kind} {indices}  from {values[0]:g} to {values[-1]:g} {unit}"
                  f"  ({args.steps} points)")
        mpi_print(f"  {'value':>12} {'E (Ha)':>18} {'dE (kcal/mol)':>15} {'converged':>10}")
        mpi_print("  " + "-" * 60)

        rows = []
        guess, restart = None, args.restart
        for v in values:
            trial = molecule.copy()
            _apply(trial, kind, indices, float(v))
            basis = build_basis(trial, cfg, getattr(cfg, "_pseudo_zeta", None))
            # each point starts from the previous point's density (ADR 0004)
            res = run_scf(trial, basis, cfg, verbose=0, initial=guess, restart=restart,
                          checkpoint_path=out_path(cfg, "_scf.npz"))
            guess, restart = {"P": res.P, "P_a": res.P_a, "P_b": res.P_b}, None
            rows.append({"value": float(v), "energy_hartree": res.total_energy,
                         "converged": bool(res.converged),
                         "n_iterations": res.n_iterations})
            e0 = rows[0]["energy_hartree"]
            mpi_print(f"  {v:>12.5f} {res.total_energy:>18.9f} "
                      f"{(res.total_energy - e0) * HARTREE_TO_EV * EV_TO_KCAL_MOL:>15.4f} "
                      f"{str(res.converged):>10}")

        mpi_print("  " + "-" * 60)
        good = [r for r in rows if r["converged"]]
        if good:
            best = min(good, key=lambda r: r["energy_hartree"])
            mpi_print(f"  minimum at {best['value']:.5f} {unit}   "
                      f"E = {best['energy_hartree']:.9f} Ha  (converged points only)")
        n_bad = len(rows) - len(good)
        if n_bad:
            # used to exit 0 and report a minimum taken from unconverged SCFs
            mpi_print(f"  [warning] the SCF did not converge at {n_bad} of {len(rows)} "
                      f"points; raise scf.max_iter")

        if is_root():
            path = out_path(cfg, "_scan.csv")
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            mpi_print(f"  written to {path}")
    report_timing(cfg)
    return EXIT_OK if all(r["converged"] for r in rows) else EXIT_SCF


def _parse_coord(spec: str, n_atoms: int):
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    if kind not in ("bond", "angle"):
        raise ConfigError(f"--coord: expected 'bond:i,j' or 'angle:i,j,k', got {spec!r}")
    try:
        indices = [int(t) for t in rest.split(",")]
    except ValueError:
        raise ConfigError(f"--coord {spec!r}: atom indices must be integers") from None
    want = 2 if kind == "bond" else 3
    if (len(indices) != want or any(not 0 <= i < n_atoms for i in indices)
            or len(set(indices)) != want):
        raise ConfigError(f"--coord {spec!r}: need {want} different atom indices in "
                          f"0..{n_atoms - 1}")
    return kind, indices


def _check_range(kind: str, values) -> None:
    if kind == "bond" and min(values) <= 0:
        raise ConfigError("--range: bond lengths must be > 0 Angstrom")
    if kind == "angle" and not (0 < min(values) and max(values) < 180):
        # 190 deg used to fold back to 170 while the CSV recorded 190
        raise ConfigError("--range: angles must lie strictly between 0 and 180 degrees")


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
            # linear: any axis perpendicular to the bond. A fixed z axis did
            # nothing for a molecule along z (CO2 stayed at 180 deg).
            helper = np.eye(3)[int(np.argmin(np.abs(v2)))]
            axis = np.cross(v2, helper)
            axis /= np.linalg.norm(axis)
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
