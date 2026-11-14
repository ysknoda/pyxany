"""Machine-readable result output (JSON / npz) plus the final structure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..constants import HARTREE_TO_EV
from ..properties import bond_orders, core_gap, loewdin, loewdin_orbital_weights, mulliken
from ..timing import REGISTRY


def result_dict(result, config, forces=None, dipole=None, extra: dict | None = None) -> dict:
    """A JSON-serialisable summary of a calculation."""
    mol, basis = result.molecule, result.basis
    pop = mulliken(result.P, result.S, basis, mol)
    bo = bond_orders(result)
    homo, lumo = result.homo_lumo()
    out = {
        "code": {"name": "pyxany", "version": _version()},
        "molecule": {
            "name": mol.name,
            "symbols": mol.symbols,
            "positions_angstrom": [list(map(float, a.pos_angstrom)) for a in mol],
            "nuclear_charges": [float(z) for z in mol.charges],
            "charge": mol.charge,
            "multiplicity": mol.multiplicity,
            "n_electrons": float(mol.n_electrons),
        },
        "settings": config.to_dict(),
        "grids": _grids(result, config),
        "scf": {
            "converged": bool(result.converged),
            "n_iterations": int(result.n_iterations),
            "unrestricted": bool(result.unrestricted),
            "history": [
                {
                    k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in h.items()
                }
                for h in result.history
            ],
        },
        "energies_hartree": {k: float(v) for k, v in result.energies.items()},
        "orbitals": {
            # For an unrestricted run these are the ALPHA channel; homo/lumo/gap
            # below span both channels, and "by_spin" carries each separately.
            "spin_channel": "alpha" if result.unrestricted else "restricted",
            "eigenvalues_hartree": _finite_list(result.eps),
            "occupations": [float(o) for o in result.occupations],
            "eigenvalues_beta_hartree": (
                _finite_list(result.eps_b) if result.unrestricted else None
            ),
            "occupations_beta": ([float(o) for o in result.occ_b] if result.unrestricted else None),
            "homo_hartree": float(homo) if np.isfinite(homo) else None,
            "lumo_hartree": float(lumo) if np.isfinite(lumo) else None,
            "gap_ev": float(result.gap_ev()) if np.isfinite(result.gap_ev()) else None,
            "by_spin": {
                k: {
                    "homo_hartree": None if not np.isfinite(v[0]) else float(v[0]),
                    "lumo_hartree": None if not np.isfinite(v[1]) else float(v[1]),
                }
                for k, v in result.homo_lumo_by_spin().items()
            },
        },
        "population": {
            "mulliken_gross": [float(x) for x in pop["gross"]],
            "mulliken_charges": [float(x) for x in pop["charges"]],
            "loewdin_charges": [
                float(x) for x in loewdin(result.P, result.S, basis, mol)["charges"]
            ],
            "mayer_bond_orders": [[float(x) for x in row] for row in bo],
            # same values; old key kept for existing readers (ADR 0012, F15)
            "wiberg_bond_orders": [[float(x) for x in row] for row in bo],
        },
        "units": {
            "energy": "hartree",
            "length": "angstrom",
            "force": "hartree/bohr",
            "dipole": "debye",
            "hartree_to_ev": HARTREE_TO_EV,
        },
    }
    if getattr(result, "transition_state", None):
        t = dict(result.transition_state)
        t["orbital"] = int(t.pop("source")) + 1
        if "dest" in t:
            t["orbital_to"] = int(t.pop("dest")) + 1
        t["estimate_ev"] = float(
            (t["eps_dest"] - t["eps_source"]) * HARTREE_TO_EV
            if t["mode"] == "excite"
            else -t["eps_source"] * HARTREE_TO_EV
        )
        out["transition_state"] = t
    if config.properties.population:
        from ..properties import bonding

        out["bonding"] = bonding.to_dict(bonding.analyze(result, config), mol)
    pseudo = [i for i, a in enumerate(mol) if a.is_pseudo]
    if pseudo:
        mask = np.array([f.atom_idx in set(pseudo) for f in basis], dtype=bool)
        C, eps, occ = (
            (result.C_a, result.eps_a, result.occ_a)
            if result.unrestricted
            else (result.C, result.eps, result.occupations)
        )
        weights = loewdin_orbital_weights(C, result.S, mask)
        g = core_gap(eps, occ, weights, config.properties.terminator_weight_max)
        out["passivation"] = {
            "spin_channel": "alpha" if result.unrestricted else "restricted",
            "terminator_weight_loewdin": [float(x) for x in weights],
            "terminator_weight_max": float(config.properties.terminator_weight_max),
            "core_homo_index": None if g["core_homo"] is None else int(g["core_homo"]),
            "core_lumo_index": None if g["core_lumo"] is None else int(g["core_lumo"]),
            "core_gap_ev": float(g["core_gap"] * HARTREE_TO_EV)
            if np.isfinite(g["core_gap"])
            else None,
            "terminator_states_below_core_lumo": g["n_terminator_states_below_core_lumo"],
        }
    if result.unrestricted:
        out["orbitals"]["eigenvalues_alpha_hartree"] = _finite_list(result.eps_a)
        out["orbitals"]["eigenvalues_beta_hartree"] = _finite_list(result.eps_b)
        out["orbitals"]["occupations_alpha"] = [float(o) for o in result.occ_a]
        out["orbitals"]["occupations_beta"] = [float(o) for o in result.occ_b]
    if forces is not None:
        out["forces_hartree_per_bohr"] = [[float(v) for v in row] for row in forces]
        out["max_force"] = float(np.abs(forces).max())
        out["force_translation_residual"] = float(np.linalg.norm(forces.sum(axis=0)))
        from ..forces.analytic import compute_forces as _compute_forces

        raw = getattr(_compute_forces, "last_translation_residual", None)
        # The projected residual is ~0 by construction; this is the informative one.
        out["force_translation_residual_raw"] = None if raw is None else float(raw)
    if dipole is not None:
        out["dipole_debye"] = [float(v) for v in dipole]
        out["dipole_magnitude_debye"] = float(np.linalg.norm(dipole))
    if config.output.timing:
        out["timing"] = REGISTRY.to_dict()
    if extra:
        out.update(extra)
    return out


def _grids(result, config) -> dict:
    from ..grid.summary import grid_summary

    return grid_summary(config, result.grid, result.molecule, result.terms)


def _finite_list(arr):
    return [float(v) if np.isfinite(v) else None for v in arr]


def _version() -> str:
    from .. import __version__

    return __version__


def write_json(path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path, text: str) -> None:
    Path(path).write_text(text if text.endswith("\n") else text + "\n")


def write_npz(path, result) -> None:
    """Save the raw matrices for post-processing."""
    data = {
        "P": result.P,
        "C": result.C,
        "eps": result.eps,
        "S": result.S,
        "T": result.T,
        "V_nuc": result.V_nuc,
        "occupations": result.occupations,
    }
    if result.unrestricted:
        data.update(
            {
                "P_a": result.P_a,
                "P_b": result.P_b,
                "C_a": result.C_a,
                "C_b": result.C_b,
                "eps_a": result.eps_a,
                "eps_b": result.eps_b,
            }
        )
    np.savez_compressed(path, **data)
