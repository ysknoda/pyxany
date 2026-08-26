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
    w(f"  RESULTS  {mol.name}   basis {config.basis.set.upper()}   "
      f"alpha_xc {config.scf.alpha}")
    w("=" * 78)
    w("")
    w(f"  {mol.summary()}")
    w(f"  SCF: {'converged' if result.converged else 'NOT CONVERGED'} in "
      f"{result.n_iterations} iterations")
    w("")

    w("  TOTAL ENERGY  [Hartree]")
    w("  " + "-" * 62)
    for key in ("E_kinetic", "E_nuclear_attraction", "E_hartree", "E_xc", "E_nn",
                "E_smearing_TS", "E_free", "E_total"):
        if key in result.energies:
            v = result.energies[key]
            w(f"  {key:<24s} {v:>18.9f}   ({v * HARTREE_TO_EV:>15.5f} eV)")
    w("")
    w(f"  Grid check: int rho dV = {result.energies['n_electrons_integrated']:.8f}   "
      f"Tr(PS) = {result.energies['trace_PS']:.8f}   "
      f"(exact: {mol.n_electrons:g})")
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
        w(f"\n  HOMO {homo * HARTREE_TO_EV:.4f} eV   LUMO {lumo * HARTREE_TO_EV:.4f} eV   "
          f"gap {result.gap_ev():.4f} eV")
    w("")

    if config.properties.population:
        pop = mulliken(result.P, result.S, basis, mol)
        bo = wiberg_bond_orders(result.P, result.S, basis, mol)
        w("  MULLIKEN POPULATION ANALYSIS")
        w("  " + "-" * 62)
        w(f"  {'atom':>8}  {'Z':>7}  {'gross':>10}  {'charge':>9}   position (Ang)")
        for i, atom in enumerate(mol):
            x, y, z = atom.pos_angstrom
            w(f"  {atom.symbol + str(i + 1):>8}  {atom.Z:>7.3f}  {pop['gross'][i]:>10.4f}  "
              f"{pop['charges'][i]:>+9.4f}   ({x:8.4f},{y:8.4f},{z:8.4f})")
        w(f"  {'sum':>8}  {mol.charges.sum():>7.3f}  {pop['gross'].sum():>10.4f}  "
          f"{pop['charges'].sum():>+9.4f}")
        w("")
        w("  WIBERG BOND ORDERS  (pairs closer than 3.0 A)")
        w("  " + "-" * 62)
        D = mol.distance_matrix() * BOHR_TO_ANG
        for i in range(len(mol)):
            for j in range(i + 1, len(mol)):
                if D[i, j] < 3.0:
                    w(f"  {mol[i].symbol + str(i + 1):>6} -- {mol[j].symbol + str(j + 1):<6} "
                      f"{bo[i, j]:>8.4f}   d = {D[i, j]:.4f} A")
        w("")
        pops = mulliken_orbital_populations(result.C, result.S, result.occupations,
                                            basis, mol)
        if len(pops):
            w("  PER-ORBITAL ATOMIC POPULATIONS (occupied)")
            hdr = "  " + "MO".rjust(4) + "  " + "  ".join(
                f"{a.symbol + str(i + 1):>9}" for i, a in enumerate(mol))
            w(hdr)
            for i, row in enumerate(pops):
                w("  " + str(i + 1).rjust(4) + "  " + "  ".join(f"{p:>9.4f}" for p in row))
            w("")

    if dipole is not None:
        w("  DIPOLE MOMENT  [Debye]")
        w("  " + "-" * 62)
        w(f"  mu_x {dipole[0]:>+10.4f}   mu_y {dipole[1]:>+10.4f}   "
          f"mu_z {dipole[2]:>+10.4f}   |mu| {np.linalg.norm(dipole):.4f}")
        w("")

    if forces is not None:
        w(f"  NUCLEAR FORCES  [Hartree/Bohr]   ({config.forces.method})")
        w("  " + "-" * 62)
        w(f"  {'atom':>8}  {'Fx':>13}  {'Fy':>13}  {'Fz':>13}  {'|F|':>11}")
        for i, atom in enumerate(mol):
            fx, fy, fz = forces[i]
            w(f"  {atom.symbol + str(i + 1):>8}  {fx:>13.8f}  {fy:>13.8f}  {fz:>13.8f}  "
              f"{np.linalg.norm(forces[i]):>11.8f}")
        w(f"  max |F| = {np.abs(forces).max():.8f}   "
          f"|sum F| = {np.linalg.norm(forces.sum(axis=0)):.2e} "
          f"(translational invariance)")
        w("")
    return "\n".join(out)


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
        w(f"  {rank + 1:>4}  {occ[i]:>7.4f}  {eps[i]:>13.6f}  "
          f"{eps[i] * HARTREE_TO_EV:>12.4f}{tag}")
