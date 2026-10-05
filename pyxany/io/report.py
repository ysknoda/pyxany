"""Human-readable result reports."""

from __future__ import annotations

import numpy as np

from ..constants import BOHR_TO_ANG, HARTREE_TO_EV
from ..properties import (
    mulliken,
    mulliken_orbital_populations,
    wiberg_bond_orders,
)


def format_results(result, config, forces=None, dipole=None) -> str:
    """Full text report for a converged SCF."""
    out: list[str] = []
    w = out.append
    mol, basis = result.molecule, result.basis

    w("=" * 78)
    w(f"  RESULTS  {mol.name}   basis {config.basis.set.upper()}   alpha_xc {config.scf.alpha_label}")
    w("=" * 78)
    w("")
    w(f"  {mol.summary()}")
    w(
        f"  SCF: {'converged' if result.converged else 'NOT CONVERGED'} in "
        f"{result.n_iterations} iterations"
    )
    w("")
    from ..grid.summary import grid_summary, summary_lines

    out.extend(summary_lines(grid_summary(config, result.grid, mol, result.terms)))

    w("  TOTAL ENERGY  [Hartree]")
    w("  " + "-" * 62)
    for key in (
        "E_kinetic",
        "E_nuclear_attraction",
        "E_hartree",
        "E_xc",
        "E_nn",
        "E_smearing_TS",
        "E_free",
        "E_total",
    ):
        if key in result.energies:
            v = result.energies[key]
            w(f"  {key:<24s} {v:>18.9f}   ({v * HARTREE_TO_EV:>15.5f} eV)")
    w("")
    if "E_embedding_electrons" in result.energies:
        out.extend(_embedding_lines(result, config))
    w(
        f"  Grid check: int rho dV = {result.energies['n_electrons_integrated']:.8f}   "
        f"Tr(PS) = {result.energies['trace_PS']:.8f}   "
        f"(exact: {mol.n_electrons:g})"
    )
    w("")

    w("  ORBITAL ENERGIES")
    w("  " + "-" * 62)
    w(f"  {'#':>4}  {'occ':>7}  {'Hartree':>13}  {'eV':>12}")
    if result.unrestricted:
        _orbital_table(w, result.eps_a, result.occ_a, "alpha")
        _orbital_table(w, result.eps_b, result.occ_b, "beta")
    else:
        _orbital_table(w, result.eps, result.occupations, None)
    homo, lumo = result.homo_lumo()
    if np.isfinite(homo) and np.isfinite(lumo):
        w(
            f"\n  HOMO {homo * HARTREE_TO_EV:.4f} eV   LUMO {lumo * HARTREE_TO_EV:.4f} eV   "
            f"gap {result.gap_ev():.4f} eV"
        )
        if result.unrestricted:
            w("  (frontier taken over both spin channels)")
            for spin, (h, l) in result.homo_lumo_by_spin().items():
                if np.isfinite(h) and np.isfinite(l):
                    w(
                        f"    {spin:>5}: HOMO {h * HARTREE_TO_EV:9.4f} eV   "
                        f"LUMO {l * HARTREE_TO_EV:9.4f} eV   "
                        f"gap {(l - h) * HARTREE_TO_EV:8.4f} eV"
                    )
    w("")

    if config.properties.population:
        pop = mulliken(result.P, result.S, basis, mol)
        bo = wiberg_bond_orders(result.P, result.S, basis, mol)
        w("  MULLIKEN POPULATION ANALYSIS")
        w("  " + "-" * 62)
        w(f"  {'atom':>8}  {'Z':>7}  {'gross':>10}  {'charge':>9}   position (Ang)")
        for i, atom in enumerate(mol):
            x, y, z = atom.pos_angstrom
            w(
                f"  {atom.symbol + str(i + 1):>8}  {atom.Z:>7.3f}  {pop['gross'][i]:>10.4f}  "
                f"{pop['charges'][i]:>+9.4f}   ({x:8.4f},{y:8.4f},{z:8.4f})"
            )
        w(
            f"  {'sum':>8}  {mol.charges.sum():>7.3f}  {pop['gross'].sum():>10.4f}  "
            f"{pop['charges'].sum():>+9.4f}"
        )
        w("")
        w("  WIBERG BOND ORDERS  (pairs closer than 3.0 A)")
        w("  " + "-" * 62)
        D = mol.distance_matrix() * BOHR_TO_ANG
        for i in range(len(mol)):
            for j in range(i + 1, len(mol)):
                if D[i, j] < 3.0:
                    w(
                        f"  {mol[i].symbol + str(i + 1):>6} -- {mol[j].symbol + str(j + 1):<6} "
                        f"{bo[i, j]:>8.4f}   d = {D[i, j]:.4f} A"
                    )
        w("")
        pops = mulliken_orbital_populations(result.C, result.S, result.occupations, basis, mol)
        if len(pops):
            spin_note = " -- ALPHA CHANNEL ONLY" if result.unrestricted else ""
            w(f"  PER-ORBITAL ATOMIC POPULATIONS (occupied){spin_note}")
            hdr = (
                "  "
                + "MO".rjust(4)
                + "  "
                + "  ".join(f"{a.symbol + str(i + 1):>9}" for i, a in enumerate(mol))
            )
            w(hdr)
            for i, row in enumerate(pops):
                w("  " + str(i + 1).rjust(4) + "  " + "  ".join(f"{p:>9.4f}" for p in row))
            w("")

    if dipole is not None:
        w("  DIPOLE MOMENT  [Debye]")
        w("  " + "-" * 62)
        w(
            f"  mu_x {dipole[0]:>+10.4f}   mu_y {dipole[1]:>+10.4f}   "
            f"mu_z {dipole[2]:>+10.4f}   |mu| {np.linalg.norm(dipole):.4f}"
        )
        w("")

    if forces is not None:
        w(f"  NUCLEAR FORCES  [Hartree/Bohr]   ({config.forces.method})")
        w("  " + "-" * 62)
        w(f"  {'atom':>8}  {'Fx':>13}  {'Fy':>13}  {'Fz':>13}  {'|F|':>11}")
        for i, atom in enumerate(mol):
            fx, fy, fz = forces[i]
            w(
                f"  {atom.symbol + str(i + 1):>8}  {fx:>13.8f}  {fy:>13.8f}  {fz:>13.8f}  "
                f"{np.linalg.norm(forces[i]):>11.8f}"
            )
        w(
            f"  max |F| = {np.abs(forces).max():.8f}   "
            f"|sum F| = {np.linalg.norm(forces.sum(axis=0)):.2e} "
            + ("(net force from the embedding charges)" if config.embedding.charges
               else "(translational invariance)")
        )
        # The line above is ~1e-16 by construction once the net translation has
        # been projected out, so it validates nothing. The pre-projection
        # residual does: it measures the grid-derivative error, and therefore
        # whether a requested force convergence is resolvable on this grid.
        raw = _raw_translation_residual()
        if raw is not None:
            w(
                f"  raw |sum F| before projection = {raw:.2e} "
                f"(grid-derivative error; refine the grid to reduce it)"
            )
        w("")

    if getattr(result, "transition_state", None):
        out.extend(_transition_state_lines(result))

    if config.properties.population:
        from ..properties import bonding

        out.extend(bonding.report_lines(bonding.analyze(result, config), result.molecule))

    passivation = _passivation_check(result, config.properties.terminator_weight_max)
    if passivation:
        out.extend(passivation)
    return "\n".join(out)


def _embedding_lines(result, config) -> list[str]:
    """Point-charge embedding: its energy and whether electrons leak into it."""
    e = result.energies
    emb = config.embedding
    spill = e["embedding_spill_electrons"]
    out = [
        "  POINT-CHARGE EMBEDDING  (included in the totals above)",
        "  " + "-" * 62,
        f"  electrons - charges  {e['E_embedding_electrons']:>18.9f} Ha",
        f"  nuclei - charges     {e['E_embedding_nuclei']:>18.9f} Ha",
        f"  electrons within {emb.spill_radius:g} A of a positive charge: {spill:.4f}",
    ]
    if spill > emb.spill_warn:
        out.append(
            f"  [warning] more than {emb.spill_warn:g} electrons sit on positive point charges, "
            "which have no Pauli repulsion: make the cations bonded to the cluster quantum "
            "atoms (see docs/embedding.md)."
        )
    out.append("")
    return out


def _transition_state_lines(result) -> list[str]:
    """Slater transition state: the orbital energies and the energy they estimate."""
    from ..constants import HARTREE_TO_EV

    t = result.transition_state
    ev = HARTREE_TO_EV
    out = ["  SLATER TRANSITION STATE", "  " + "-" * 62]
    i = t["source"] + 1
    if t["mode"] == "remove":
        out.append(f"  {t['electrons']:g} {t['spin']} electron removed from orbital {i}")
        out.append(
            f"  eps(TS) = {t['eps_source']:.6f} Ha  ->  ionization energy "
            f"-eps(TS) = {-t['eps_source'] * ev:.4f} eV"
        )
    else:
        j = t["dest"] + 1
        out.append(f"  {t['electrons']:g} {t['spin']} electron moved from orbital {i} to {j}")
        out.append(
            f"  eps_to - eps_from = {(t['eps_dest'] - t['eps_source']) * ev:.4f} eV "
            f"(excitation energy)"
        )
    out.append("  (third-order accurate by Janak's theorem; compare Delta-SCF if in doubt)")
    out.append("")
    return out


def _passivation_check(result, max_weight: float = 0.5) -> list[str]:
    """Does the terminator actually stay out of the way?

    A pseudo-hydrogen is only doing its job if it contributes no states in the
    energy range being studied and carries close to no net charge. Orbital
    weights are Loewdin, not Mulliken: Mulliken weights leave [0, 1] with the
    diffuse functions of fractional terminators (1.55 on a MgO6 LUMO, against
    Loewdin 0.72) and overstated the effect about twofold.
    """
    from ..constants import HARTREE_TO_EV
    from ..properties import core_gap, loewdin, loewdin_orbital_weights

    mol, basis = result.molecule, result.basis
    pseudo = [i for i, a in enumerate(mol) if a.is_pseudo]
    if not pseudo:
        return []

    out: list[str] = []
    w = out.append
    mask = np.array([f.atom_idx in set(pseudo) for f in basis], dtype=bool)

    w("  PASSIVATION CHECK  (Loewdin weights on the terminators)")
    w("  " + "-" * 62)

    charges = np.asarray(loewdin(result.P, result.S, basis, mol)["charges"])[pseudo]
    w(
        f"  {len(pseudo)} terminator(s); Loewdin charge "
        f"min {charges.min():+.4f}  max {charges.max():+.4f}  "
        f"mean {charges.mean():+.4f}"
    )
    if np.abs(charges).max() > 0.3:
        w(
            "  [warning] a terminator carries a large net charge; the counting rule "
            "or the coordination number is probably wrong."
        )

    channels = (
        [
            ("alpha", result.eps_a, result.occ_a, result.C_a),
            ("beta", result.eps_b, result.occ_b, result.C_b),
        ]
        if result.unrestricted
        else [("", result.eps, result.occupations, result.C)]
    )
    flagged = False
    for name, eps, occ, C in channels:
        if eps is None:
            continue
        weights = loewdin_orbital_weights(C, result.S, mask)
        g = core_gap(eps, occ, weights, max_weight)
        label = f" ({name})" if name else ""
        if g["homo"] is None or g["lumo"] is None:
            continue
        w(
            f"  HOMO{label} weight {weights[g['homo']]:.3f}   LUMO{label} weight "
            f"{weights[g['lumo']]:.3f}   (terminator state above {max_weight:g})"
        )
        if np.isfinite(g["core_gap"]):
            w(
                f"  gap {g['gap'] * HARTREE_TO_EV:.3f} eV   core gap{label} "
                f"{g['core_gap'] * HARTREE_TO_EV:.3f} eV   "
                f"({g['n_terminator_states_below_core_lumo']} terminator state(s) "
                f"below the core LUMO)"
            )
        if weights[g["homo"]] > max_weight or weights[g["lumo"]] > max_weight:
            flagged = True
    if flagged:
        w(
            "  [warning] a frontier orbital is mostly on the terminators: the plain gap is "
            "not that of the cluster interior -- quote the core gap, relax the terminators "
            "(opt --freeze-core) or enlarge the cluster."
        )
    else:
        w("  frontier orbitals are on the cluster interior -- passivation looks sound.")
    w("")
    return out


def _orbital_table(w, eps, occ, label, n_extra: int = 5) -> None:
    if label:
        w(f"  -- {label} channel --")
    finite = np.where(np.isfinite(eps))[0]
    order = finite[np.argsort(eps[finite])]
    n_occ = int((occ > 1e-8).sum())
    for rank, i in enumerate(order[: n_occ + n_extra]):
        tag = ""
        if rank == n_occ - 1:
            tag = "  <- HOMO"
        elif rank == n_occ:
            tag = "  <- LUMO"
        w(f"  {rank + 1:>4}  {occ[i]:>7.4f}  {eps[i]:>13.6f}  {eps[i] * HARTREE_TO_EV:>12.4f}{tag}")


def _raw_translation_residual():
    """The |sum F| the analytic force routine saw before projecting."""
    from ..forces.analytic import compute_forces

    return getattr(compute_forces, "last_translation_residual", None)
