"""Bond overlap populations and partial DOS for one SCF result (ADR 0005).

One place assembles what the text report, the JSON output and the figures
show, so the three cannot disagree:

* the atom-pair bond overlap population matrix;
* the MO-resolved overlap population of the selected pairs (bonded pairs by
  Mayer bond order, or the pairs the input names), and the same summed per
  element-pair type ("Ga-As", "As-HAs"), which is what a COOP-type plot shows;
* Loewdin partial-DOS weights per element and per (element, l).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .population import (
    bonded_pairs,
    loewdin_orbital_groups,
    mayer_bond_orders,
    mo_overlap_population,
    overlap_population,
)


@dataclass
class Channel:
    """One spin channel (or the restricted one): energies, occupations, analyses."""

    name: str
    eps: np.ndarray
    occ: np.ndarray
    mo_op: np.ndarray  # (n_mo, n_pairs), per electron
    mo_op_by_type: dict  # {"Ga-As": (n_mo,)}, per electron
    pdos_element: dict  # {"Ga": (n_mo,)}, Loewdin weights
    pdos_element_l: dict  # {"Ga d": (n_mo,)}


@dataclass
class BondingAnalysis:
    op: np.ndarray  # (n_atoms, n_atoms) bond overlap population
    bond_orders: np.ndarray
    pairs: list
    pair_types: list
    channels: list
    homo_ev: float


def pair_type(molecule, a: int, b: int) -> str:
    """'Ga-As', 'As-HAs' style label: hydrogen-like atoms last, otherwise by Z."""

    def key(i):
        atom = molecule[i]
        return (atom.element == "H", atom.Z, atom.symbol)

    first, second = sorted((a, b), key=key)
    return f"{molecule[first].symbol}-{molecule[second].symbol}"


def analyse(result, config) -> BondingAnalysis:
    from ..constants import HARTREE_TO_EV

    mol, basis, S = result.molecule, result.basis, result.S
    settings = config.properties.overlap_population
    op = overlap_population(result.P, S, basis, mol)
    bo = mayer_bond_orders(result.P, S, basis, mol, result.P_a, result.P_b)
    pairs = [tuple(sorted(map(int, p))) for p in settings.get("pairs", [])] or bonded_pairs(
        bo, settings.get("min_bond_order", 0.2)
    )
    types = [pair_type(mol, a, b) for a, b in pairs]
    el = [mol[f.atom_idx].symbol for f in basis]
    el_l = [f"{mol[f.atom_idx].symbol} {'spdfg'[f.l]}" for f in basis]
    # species axes in order of nuclear charge (pseudo-hydrogens first), then l
    z_of = {mol[f.atom_idx].symbol: float(mol[f.atom_idx].Z) for f in basis}

    def by_z(groups: dict) -> dict:
        def key(item):
            label = str(item[0])
            shell = "spdfg".find(label[-1]) if " " in label else 0
            return (z_of[label.split()[0]], shell)

        return dict(sorted(groups.items(), key=key))

    spins = (
        [
            ("alpha", result.eps_a, result.occ_a, result.C_a),
            ("beta", result.eps_b, result.occ_b, result.C_b),
        ]
        if result.unrestricted
        else [("", result.eps, result.occupations, result.C)]
    )
    channels = []
    for name, eps, occ, C in spins:
        mo_op = mo_overlap_population(C, S, basis, pairs) if pairs else np.zeros((len(eps), 0))
        by_type = {}
        for k, t in enumerate(types):
            by_type[t] = by_type.get(t, 0.0) + mo_op[:, k]
        channels.append(
            Channel(
                name,
                eps * HARTREE_TO_EV,
                occ,
                mo_op,
                by_type,
                by_z(loewdin_orbital_groups(C, S, el)),
                by_z(loewdin_orbital_groups(C, S, el_l)),
            )
        )
    homo, _ = result.homo_lumo()
    return BondingAnalysis(op, bo, pairs, types, channels, float(homo * HARTREE_TO_EV))


def report_lines(analysis: BondingAnalysis, molecule, max_rows: int = 12) -> list[str]:
    """Text block: the strongest bonded pairs and the per-type totals."""
    out = ["  BOND OVERLAP POPULATION  (Mulliken; > 0 bonding, < 0 antibonding)", "  " + "-" * 62]
    if not analysis.pairs:
        return out + ["  no pair reaches the bond-order threshold", ""]
    rows = sorted(analysis.pairs, key=lambda p: -abs(analysis.op[p]))
    out.append(f"  {'pair':<16}{'overlap pop.':>14}{'Mayer B.O.':>12}")
    for a, b in rows[:max_rows]:
        label = f"{molecule[a].symbol}{a + 1}-{molecule[b].symbol}{b + 1}"
        out.append(f"  {label:<16}{analysis.op[a, b]:>14.4f}{analysis.bond_orders[a, b]:>12.4f}")
    if len(rows) > max_rows:
        out.append(f"  ... {len(rows) - max_rows} more pairs in the JSON output")
    totals = {}
    for (a, b), t in zip(analysis.pairs, analysis.pair_types, strict=True):
        totals.setdefault(t, []).append(analysis.op[a, b])
    out.append(
        "  per type: " + "   ".join(f"{t} {np.mean(v):+.4f} (x{len(v)})" for t, v in totals.items())
    )
    out.append("")
    return out


def to_dict(analysis: BondingAnalysis, molecule) -> dict:
    return {
        "overlap_population": analysis.op.tolist(),
        "pairs": [[int(a), int(b)] for a, b in analysis.pairs],
        "pair_types": analysis.pair_types,
        "channels": [
            {
                "spin": ch.name or "restricted",
                "eigenvalues_ev": [float(x) for x in ch.eps],
                "mo_overlap_population_by_type": {
                    t: [float(x) for x in v] for t, v in ch.mo_op_by_type.items()
                },
                "loewdin_pdos_weights_by_element": {
                    k: [float(x) for x in v] for k, v in ch.pdos_element.items()
                },
            }
            for ch in analysis.channels
        ],
    }
