"""Self-consistent field driver (restricted and unrestricted Kohn-Sham X-alpha).

Every energy term in one iteration is built from the *same* density matrix, so
the printed energy is a genuine functional of a single density.  (The original
single-file version built J from the freshly diagonalised density while
evaluating V_xc and the energy from the mixed one.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..basis.evaluate import chi_matrix
from ..constants import HARTREE_TO_EV
from ..grid import build_molecular_grid
from ..integrals import xc
from ..integrals.coulomb import DEFAULT_L_MAX, hartree_potential, nuclear_potential
from ..integrals.overlap import build_overlap_kinetic, condition_number
from ..parallel import allreduce_array, allreduce_scalar, bcast_array, is_root, mpi_print
from ..timing import timer
from .diis import EDIIS, PulayDIIS
from .guess import atomic_superposition, rescale_to_electron_count, spin_split
from .linalg import energy_weighted_density, solve_generalised
from .occupations import (
    density_matrix,
    fermi_occupations,
    smearing_entropy,
    spin_electron_counts,
)


@dataclass
class SCFResult:
    """Everything downstream code needs from a converged (or stalled) SCF."""

    converged: bool
    n_iterations: int
    energies: dict
    eps: np.ndarray
    C: np.ndarray
    P: np.ndarray
    occupations: np.ndarray
    S: np.ndarray
    T: np.ndarray
    V_nuc: np.ndarray
    grid: object
    chi: np.ndarray
    molecule: object
    basis: object
    unrestricted: bool = False
    n_alpha: float = 0.0
    n_beta: float = 0.0
    eps_a: np.ndarray = None
    eps_b: np.ndarray = None
    C_a: np.ndarray = None
    C_b: np.ndarray = None
    P_a: np.ndarray = None
    P_b: np.ndarray = None
    occ_a: np.ndarray = None
    occ_b: np.ndarray = None
    history: list = field(default_factory=list)

    @property
    def total_energy(self) -> float:
        return self.energies["E_total"]

    def homo_lumo(self) -> tuple[float, float]:
        """(HOMO, LUMO) eigenvalues in Hartree; nan when there is no LUMO."""
        occ, eps = self.occupations, self.eps
        filled = np.where(occ > 1e-8)[0]
        empty = np.where((occ <= 1e-8) & np.isfinite(eps))[0]
        homo = float(eps[filled].max()) if len(filled) else float("nan")
        lumo = float(eps[empty].min()) if len(empty) else float("nan")
        return homo, lumo

    def gap_ev(self) -> float:
        h, l = self.homo_lumo()
        return (l - h) * HARTREE_TO_EV


def _density_on_points(chi: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.einsum("km,mn,kn->k", chi, P, chi, optimize=True)


def _total_energy(P, T, V_nuc_mat, rho, V_H, weights, alpha, E_nn,
                  rho_a=None, rho_b=None) -> dict:
    """Assemble the X-alpha total energy from one density."""
    E_kin = float(np.einsum("mn,nm->", P, T))
    E_ne = float(np.einsum("mn,nm->", P, V_nuc_mat))
    E_h = allreduce_scalar(0.5 * float(np.dot(weights, rho * V_H)))
    if rho_a is None:
        E_xc = allreduce_scalar(xc.exchange_energy(rho, weights, alpha))
    else:
        E_xc = allreduce_scalar(xc.exchange_energy_spin(rho_a, rho_b, weights, alpha))
    return {
        "E_kinetic": E_kin,
        "E_nuclear_attraction": E_ne,
        "E_hartree": E_h,
        "E_xc": E_xc,
        "E_nn": E_nn,
        "E_total": E_kin + E_ne + E_h + E_xc + E_nn,
    }


def run_scf(molecule, basis, config, verbose: int = 1) -> SCFResult:
    """Run the SCF to convergence.

    ``config`` is a :class:`dvxa.config.Config`; only the ``grid``, ``scf`` and
    ``basis`` sections are read here.
    """
    scf_cfg, grid_cfg = config.scf, config.grid
    alpha = scf_cfg.alpha
    n_electrons = molecule.n_electrons
    n_basis = len(basis)

    unrestricted = scf_cfg.unrestricted or molecule.multiplicity > 1 or (
        abs(n_electrons - round(n_electrons)) < 1e-8 and int(round(n_electrons)) % 2 == 1
    )
    n_alpha, n_beta = (
        spin_electron_counts(n_electrons, molecule.multiplicity)
        if unrestricted else (n_electrons / 2.0, n_electrons / 2.0)
    )
    max_occ = 1.0 if unrestricted else 2.0

    if verbose:
        _print_header(molecule, basis, config, unrestricted, n_alpha, n_beta)

    # ---- one-time matrices ------------------------------------------------
    S, T = build_overlap_kinetic(molecule, basis,
                                 n_r=grid_cfg.pair_n_r, n_a=grid_cfg.pair_n_a,
                                 screening_tol=grid_cfg.screening_tol)
    cond = condition_number(S)
    if verbose:
        mpi_print(f"  S condition number : {cond:.4g}"
                  + ("   [WARNING: near-linear dependence]" if cond > 1e4 else ""))

    grid = build_molecular_grid(molecule, grid_cfg.n_r, grid_cfg.n_a,
                                partition=grid_cfg.partition)
    with timer("scf.chi"):
        chi = chi_matrix(basis, molecule.positions, grid.local_points)
    weights = grid.local_weights
    if verbose:
        mpi_print(f"  DV grid            : {len(grid)} points "
                  f"({grid.n_r}r x {grid.n_a}th x {2 * grid.n_a}ph per atom), "
                  f"chi {grid.chi_memory_mb(n_basis):.1f} MB/rank")

    with timer("scf.v_nuc"):
        v_nuc = nuclear_potential(grid.local_points, molecule.positions, molecule.charges)
        V_nuc = allreduce_array((chi * (weights * v_nuc)[:, None]).T @ chi)
    E_nn = molecule.nuclear_repulsion()

    # ---- initial density --------------------------------------------------
    P = rescale_to_electron_count(atomic_superposition(molecule, basis), S, n_electrons)
    if unrestricted:
        P_a, P_b = spin_split(P, molecule, basis, n_alpha, n_beta)
        P = P_a + P_b
        accel = {
            "diis_a": PulayDIIS(), "diis_b": PulayDIIS(),
            "ediis_a": EDIIS(), "ediis_b": EDIIS(),
        }
    else:
        P_a = P_b = None
        accel = {"diis": PulayDIIS(), "ediis": EDIIS()}
    bcast_array(P)

    history: list[dict] = []
    energies: dict = {}
    eps = C = occ = None
    eps_a = eps_b = C_a = C_b = occ_a = occ_b = None
    converged = False
    e_prev = float("inf")
    l_max = grid_cfg.poisson_l_max

    if verbose:
        _print_table_header(unrestricted)

    for it in range(1, scf_cfg.max_iter + 1):
        with timer("scf.iteration"):
            # --- density on the grid, from the current P -------------------
            with timer("scf.density"):
                if unrestricted:
                    rho_a = _density_on_points(chi, P_a)
                    rho_b = _density_on_points(chi, P_b)
                    rho = rho_a + rho_b
                else:
                    rho_a = rho_b = None
                    rho = _density_on_points(chi, P)

            # --- Hartree potential (needs the density on the whole grid) ---
            with timer("scf.hartree"):
                rho_full = _gather_full(grid, rho)
                V_H = hartree_potential(grid, rho_full, molecule.positions,
                                        l_max=l_max, eval_points=grid.local_points)
                J = allreduce_array((chi * (weights * V_H)[:, None]).T @ chi)

            # --- exchange-correlation --------------------------------------
            with timer("scf.xc"):
                if unrestricted:
                    Vxc_a = allreduce_array(
                        (chi * (weights * xc.potential_spin(rho_a, alpha))[:, None]).T @ chi)
                    Vxc_b = allreduce_array(
                        (chi * (weights * xc.potential_spin(rho_b, alpha))[:, None]).T @ chi)
                else:
                    Vxc = allreduce_array(
                        (chi * (weights * xc.potential(rho, alpha))[:, None]).T @ chi)

            energies = _total_energy(P, T, V_nuc, rho, V_H, weights, alpha, E_nn,
                                     rho_a, rho_b)
            dE = abs(energies["E_total"] - e_prev)
            e_prev = energies["E_total"]

            # --- Fock matrices and extrapolation ---------------------------
            if unrestricted:
                F_a = T + V_nuc + J + Vxc_a
                F_b = T + V_nuc + J + Vxc_b
                Fa_use, Fb_use, method, err, mix = _accelerate_uks(
                    accel, F_a, F_b, P_a, P_b, S, energies["E_total"], it, scf_cfg, dE)
                eps_a, C_a, eps_b, C_b = _diagonalise_uks(Fa_use, Fb_use, S, n_basis)
                occ_a = fermi_occupations(eps_a, n_alpha, scf_cfg.smearing, 1.0)
                occ_b = fermi_occupations(eps_b, n_beta, scf_cfg.smearing, 1.0)
                Pa_new = density_matrix(C_a, occ_a)
                Pb_new = density_matrix(C_b, occ_b)
                d_rms = float(np.sqrt(np.mean((Pa_new + Pb_new - P) ** 2)))
                P_a = mix * Pa_new + (1.0 - mix) * P_a
                P_b = mix * Pb_new + (1.0 - mix) * P_b
                P = P_a + P_b
                eps, C, occ = eps_a, C_a, occ_a
                extra = (float(eps_a[np.argsort(eps_a)][max(int(round(n_alpha)) - 1, 0)]),
                         float(eps_b[np.argsort(eps_b)][max(int(round(n_beta)) - 1, 0)]))
            else:
                F = T + V_nuc + J + Vxc
                F_use, method, err, mix = _accelerate_rks(
                    accel, F, P, S, energies["E_total"], it, scf_cfg, dE)
                eps, C, _ = solve_generalised(F_use, S, scf_cfg.s_threshold)
                occ = fermi_occupations(eps, n_electrons, scf_cfg.smearing, 2.0)
                P_new = density_matrix(C, occ)
                d_rms = float(np.sqrt(np.mean((P_new - P) ** 2)))
                P = mix * P_new + (1.0 - mix) * P
                homo = float(eps[occ > 1e-8].max()) if np.any(occ > 1e-8) else float("nan")
                extra = (homo * HARTREE_TO_EV,)

            bcast_array(P)
            history.append({"iter": it, "E": energies["E_total"], "dE": dE,
                            "error": err, "d_rms": d_rms, "method": method, "mix": mix})
            if verbose:
                _print_row(it, energies["E_total"], dE, err, d_rms, extra, method)

            if it > 2 and dE < scf_cfg.e_tol and d_rms < scf_cfg.d_tol:
                converged = True
                break

    # ---- final quantities from the converged density ----------------------
    if unrestricted:
        rho_a = _density_on_points(chi, P_a)
        rho_b = _density_on_points(chi, P_b)
        rho = rho_a + rho_b
    else:
        rho_a = rho_b = None
        rho = _density_on_points(chi, P)
    rho_full = _gather_full(grid, rho)
    V_H = hartree_potential(grid, rho_full, molecule.positions, l_max=l_max,
                            eval_points=grid.local_points)
    energies = _total_energy(P, T, V_nuc, rho, V_H, weights, alpha, E_nn, rho_a, rho_b)

    if scf_cfg.smearing > 0:
        if unrestricted:
            ts = (smearing_entropy(occ_a, scf_cfg.smearing, 1.0)
                  + smearing_entropy(occ_b, scf_cfg.smearing, 1.0))
        else:
            ts = smearing_entropy(occ, scf_cfg.smearing, 2.0)
        energies["E_smearing_TS"] = ts
        energies["E_free"] = energies["E_total"] - ts

    energies["n_electrons_integrated"] = allreduce_scalar(float(np.dot(weights, rho)))
    energies["trace_PS"] = float(np.einsum("mn,nm->", P, S))

    if verbose:
        _print_footer(converged, len(history), energies, n_electrons)

    return SCFResult(
        converged=converged, n_iterations=len(history), energies=energies,
        eps=eps, C=C, P=P, occupations=occ, S=S, T=T, V_nuc=V_nuc,
        grid=grid, chi=chi, molecule=molecule, basis=basis,
        unrestricted=unrestricted, n_alpha=n_alpha, n_beta=n_beta,
        eps_a=eps_a, eps_b=eps_b, C_a=C_a, C_b=C_b, P_a=P_a, P_b=P_b,
        occ_a=occ_a, occ_b=occ_b, history=history,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _gather_full(grid, rho_local: np.ndarray) -> np.ndarray:
    """Assemble the full-grid density from the rank-local slices."""
    from ..parallel import PARALLEL

    if not PARALLEL:
        return rho_local
    full = np.zeros(len(grid))
    full[grid.local_slice] = rho_local
    return allreduce_array(full)


def _diis_mixing(cfg, extrapolated: bool, error: float) -> float:
    """How much of the freshly built density to accept.

    A successful Pulay extrapolation already contains the convergence
    acceleration, so the new density is taken *whole* (mix = 1).  Damping on top
    of DIIS simply multiplies the residual by (1 - mix) every iteration and
    throws the acceleration away -- the slow geometric decay that behaviour
    produces is unmistakable in an SCF trace.  Damping is kept only for the
    fallback path where no extrapolation was available.
    """
    if extrapolated and error < 1.0:
        return 1.0
    return cfg.mix_diis


def _accelerate_rks(accel, F, P, S, energy, it, cfg, dE):
    if it <= cfg.ediis_iters:
        accel["ediis"].push(F, P, energy)
        out = accel["ediis"].extrapolate()
        return (out if out is not None else F), "EDIIS", dE, cfg.mix_ediis
    accel["diis"].push(F, P, S)
    out = accel["diis"].extrapolate()
    err = accel["diis"].error
    return (out if out is not None else F), "DIIS", err, _diis_mixing(cfg, out is not None, err)


def _accelerate_uks(accel, F_a, F_b, P_a, P_b, S, energy, it, cfg, dE):
    if it <= cfg.ediis_iters:
        accel["ediis_a"].push(F_a, P_a, energy)
        accel["ediis_b"].push(F_b, P_b, energy)
        a = accel["ediis_a"].extrapolate()
        b = accel["ediis_b"].extrapolate()
        return (a if a is not None else F_a), (b if b is not None else F_b), \
            "EDIIS", dE, cfg.mix_ediis
    accel["diis_a"].push(F_a, P_a, S)
    accel["diis_b"].push(F_b, P_b, S)
    a = accel["diis_a"].extrapolate()
    b = accel["diis_b"].extrapolate()
    err = max(accel["diis_a"].error, accel["diis_b"].error)
    mix = _diis_mixing(cfg, a is not None and b is not None, err)
    return (a if a is not None else F_a), (b if b is not None else F_b), "DIIS", err, mix


def _diagonalise_uks(F_a, F_b, S, n_basis):
    eps_a, C_a, _ = solve_generalised(F_a, S)
    eps_b, C_b, _ = solve_generalised(F_b, S)
    return eps_a, C_a, eps_b, C_b


def _print_header(molecule, basis, config, unrestricted, n_alpha, n_beta):
    mpi_print("=" * 78)
    mpi_print(f"  dvxa  --  DV-Xalpha SCF   [{molecule.name}]")
    mpi_print("=" * 78)
    mpi_print(f"  {molecule.summary()}")
    mpi_print(f"  Basis              : {config.basis.set.upper()}  "
              f"({len(basis)} functions, max l = {max(f.l for f in basis)})")
    mode = (f"UKS  n_alpha={n_alpha:g} n_beta={n_beta:g}" if unrestricted
            else f"RKS  n_occ={n_alpha:g}")
    mpi_print(f"  Method             : {mode}   alpha_xc = {config.scf.alpha}")


def _print_table_header(unrestricted):
    cols = ("HOMO-a (Ha)  HOMO-b (Ha)" if unrestricted else "HOMO (eV)")
    mpi_print("")
    mpi_print(f"  {'it':>3}  {'E_total (Ha)':>17}  {'|dE|':>10}  {'DIIS err':>10}  "
              f"{'dP rms':>10}  {cols:>22}  method")
    mpi_print("  " + "-" * 92)


def _print_row(it, E, dE, err, d_rms, extra, method):
    tail = "  ".join(f"{v:>10.4f}" for v in extra)
    mpi_print(f"  {it:>3}  {E:>17.9f}  {dE:>10.2e}  {err:>10.2e}  "
              f"{d_rms:>10.2e}  {tail:>22}  {method}")


def _print_footer(converged, n_it, energies, n_electrons):
    mpi_print("  " + "-" * 92)
    status = "CONVERGED" if converged else "NOT CONVERGED"
    mpi_print(f"  [{status}] after {n_it} iterations")
    n_int = energies["n_electrons_integrated"]
    mpi_print(f"  int rho dV = {n_int:.6f}   Tr(PS) = {energies['trace_PS']:.6f}   "
              f"(expected {n_electrons:g})")
