"""Every integration grid a calculation uses, as configured and as built.

Reported by ``--dry-run``, the SCF header, the text report and the JSON output
so that a change to any ``grid:`` setting can be traced to its effect.
"""

from __future__ import annotations

import numpy as np


def grid_summary(cfg, grid, molecule, terms=None) -> dict:
    """Settings and actual sizes of all grids; ``terms`` adds block sparsity."""
    g = cfg.grid
    by_element: dict[str, int] = {}
    for atom, n in sorted(zip(molecule, grid.n_r_atom, strict=True), key=lambda x: x[0].Z):
        by_element.setdefault(atom.symbol, int(n))
    unpruned = int(sum(int(n) * grid.n_ang(grid.n_a) for n in grid.n_r_atom))
    regions = []
    if grid.segments is not None and grid.pruned:
        regions = [[b, n] for b, n in _regions_used(cfg, grid.n_a)]
    out = {
        "molecular": {
            "n_r_input": int(g.n_r),
            "n_a": int(grid.n_a),
            "angular": grid.angular,
            "angular_points_per_shell": grid.n_ang(grid.n_a),
            "radial_points_by_species": by_element,
            "pruning_regions": regions,
            "points": int(len(grid)),
            "points_unpruned": unpruned,
            "poisson_l_max": int(g.l_max),
        },
        "pair": dict(g.pair),
        "atom": dict(g.atom),
        "guess": dict(g.guess),
        "one_centre": dict(g.one_centre),
        "basis_cutoff": float(g.basis_cutoff),
    }
    if terms is not None and getattr(terms, "sparsity", None):
        out["basis_sparsity"] = {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in terms.sparsity.items()
        }
    return out


def _regions_used(cfg, n_a: int) -> list:
    from ..config import DEFAULT_PRUNING

    regions = cfg.grid.pruning.get("regions", "auto")
    regions = DEFAULT_PRUNING(n_a) if regions == "auto" else regions
    return [(float(b), int(n)) for b, n in regions if int(n) < n_a]


def _angular(kind: str, n_a: int) -> str:
    from .angular import angular_grid, lebedev_order_for

    if kind == "lebedev":
        return f"Lebedev order {lebedev_order_for(n_a)} ({len(angular_grid(kind, n_a)[0])} pts)"
    return f"{n_a} x {2 * n_a}"


def summary_lines(d: dict) -> list[str]:
    """Text block for the reports."""
    m = d["molecular"]
    species = "  ".join(f"{s} {n}" for s, n in m["radial_points_by_species"].items())
    lines = [
        "  INTEGRATION GRIDS",
        "  " + "-" * 62,
        f"  molecular   {m['points']:,} points; radial {species}; "
        f"angular {_angular(m['angular'], m['n_a'])}",
    ]
    if m["pruning_regions"]:
        regs = ", ".join(f"r < {b:g} R: {_angular(m['angular'], n)}"
                         for b, n in m["pruning_regions"])
        lines.append(
            f"              pruned ({regs}); "
            f"{m['points'] / max(1, m['points_unpruned']):.0%} of "
            f"{m['points_unpruned']:,}"
        )
    else:
        lines.append("              not pruned")
    lines.append(f"              Hartree multipoles up to l = {m['poisson_l_max']}")
    lines.append(
        f"  pair        {d['pair']['n_r']} radial x {d['pair']['n_a']} x "
        f"{2 * d['pair']['n_a']} angular per centre (S, T)"
    )
    lines.append(f"  atom        {d['atom']['n_points']} log-grid points (atomic solver)")
    lines.append(
        f"  guess       {d['guess']['n_r']} radial x {d['guess']['n_a']} x "
        f"{2 * d['guess']['n_a']} angular (SAD projection)"
    )
    lines.append(f"  one-centre  {d['one_centre']['n_r']} radial points (non-Slater functions)")
    cut = d["basis_cutoff"]
    if cut > 0:
        sp = d.get("basis_sparsity")
        kept = f"; {sp['kept_fraction']:.1%} of chi kept in {sp['blocks']} blocks" if sp else ""
        lines.append(f"  basis cutoff {cut:.0e}{kept}")
    else:
        lines.append("  basis cutoff off (dense chi)")
    lines.append("")
    return lines
