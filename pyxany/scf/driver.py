"""Self-consistent field driver (restricted and unrestricted Kohn-Sham X-alpha).

Every energy term in one iteration is built from the *same* density matrix, so
the printed energy is a genuine functional of a single density.  (The original
single-file version built J from the freshly diagonalized density while
evaluating V_xc and the energy from the mixed one.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import ANG_TO_BOHR, HARTREE_TO_EV
from ..grid.molecular import grid_from_config
from ..integrals import xc
from ..integrals.coulomb import hartree_potential, nuclear_potential
from ..integrals.overlap import build_overlap_kinetic, condition_number
from ..parallel import allreduce_array, allreduce_scalar, bcast_array, mpi_print
from ..timing import timer
from .diis import EDIIS, PulayDIIS, SpinPulayDIIS
from .grid_terms import GridTerms, choose_strategy
from .guess import rescale_to_electron_count, sad_density, spin_split
from .linalg import canonical_orthogonalizer, solve_generalized
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
    #: Cached chi matrix, or None when the run was chunked.
    chi: np.ndarray
    #: Grid-term evaluator, valid either way.
    terms: object
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
    #: Slater transition state: orbital indices and energies of the final iteration
    transition_state: dict | None = None

    @property
    def total_energy(self) -> float:
        return self.energies["E_total"]

    @staticmethod
    def _frontier(eps, occ) -> tuple[float, float]:
        filled = np.where(occ > 1e-8)[0]
        empty = np.where((occ <= 1e-8) & np.isfinite(eps))[0]
        return (float(eps[filled].max()) if len(filled) else float("nan"),
                float(eps[empty].min()) if len(empty) else float("nan"))

    def homo_lumo(self) -> tuple[float, float]:
        """(HOMO, LUMO) eigenvalues in Hartree; nan when there is no LUMO.

        For an unrestricted run the frontier is taken over **both** spin
        channels -- the highest occupied level of either and the lowest virtual
        of either.  Reading the alpha channel alone can name the wrong orbital
        entirely: for the CH3 radical the beta LUMO lies below the alpha LUMO,
        so the alpha-only gap is 13.96 eV against a true 3.40 eV.
        """
        if not self.unrestricted or self.eps_a is None or self.eps_b is None:
            return self._frontier(self.eps, self.occupations)
        homo_a, lumo_a = self._frontier(self.eps_a, self.occ_a)
        homo_b, lumo_b = self._frontier(self.eps_b, self.occ_b)
        return (float(np.nanmax([homo_a, homo_b])),
                float(np.nanmin([lumo_a, lumo_b])))

    def homo_lumo_by_spin(self) -> dict:
        """Per-channel frontier levels, for a report that wants both."""
        if not self.unrestricted or self.eps_a is None or self.eps_b is None:
            homo, lumo = self._frontier(self.eps, self.occupations)
            return {"restricted": (homo, lumo)}
        return {"alpha": self._frontier(self.eps_a, self.occ_a),
                "beta": self._frontier(self.eps_b, self.occ_b)}

    def gap_ev(self) -> float:
        h, l = self.homo_lumo()
        return (l - h) * HARTREE_TO_EV


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


def run_scf(molecule, basis, config, verbose: int = 1, *, initial: dict | None = None,
            restart=None, checkpoint_path=None) -> SCFResult:
    """Run the SCF to convergence.

    ``config`` is a :class:`pyxany.config.Config`; only the ``grid``, ``scf`` and
    ``basis`` sections are read here.

    ``initial`` is a starting density ``{"P": ..., "P_a": ..., "P_b": ...}``
    (e.g. the previous point of an optimization); ``restart`` is a checkpoint
    file to resume from or, at another geometry, to start from. With
    ``checkpoint_path`` the state is written every ``scf.checkpoint_every``
    iterations (if > 0) and at the end. See :mod:`pyxany.scf.checkpoint`.
    """
    scf_cfg, grid_cfg = config.scf, config.grid
    alpha = None          # X-alpha parameter on the grid, set once the grid exists
    n_electrons = molecule.n_electrons
    n_basis = len(basis)

    ts_cfg = config.transition_state
    # a half-electron hole in one spin channel makes any system open-shell
    unrestricted = ts_cfg.enabled or scf_cfg.unrestricted or molecule.multiplicity > 1 or (
        abs(n_electrons - round(n_electrons)) < 1e-8 and int(round(n_electrons)) % 2 == 1
    )
    n_alpha, n_beta = (
        spin_electron_counts(n_electrons, molecule.multiplicity)
        if unrestricted else (n_electrons / 2.0, n_electrons / 2.0)
    )
    if verbose:
        _print_header(molecule, basis, config, unrestricted, n_alpha, n_beta)

    # ---- one-time matrices ------------------------------------------------
    S, T = build_overlap_kinetic(molecule, basis,
                                 n_r=grid_cfg.pair["n_r"], n_a=grid_cfg.pair["n_a"],
                                 screening_tol=grid_cfg.screening_tol,
                                 n_r_1c=grid_cfg.one_center["n_r"])
    cond = condition_number(S)
    # Retained (non-linearly-dependent) space, for the self-consistency test.
    X_keep, _ = canonical_orthogonalizer(S, scf_cfg.s_threshold)
    holes = None
    if ts_cfg.enabled:
        from .transition_state import HoleManager

        holes = HoleManager(config, molecule, basis, S)
    if verbose:
        mpi_print(f"  S condition number : {cond:.4g}"
                  + ("   [WARNING: near-linear dependence]" if cond > 1e4 else ""))

    grid = grid_from_config(molecule, grid_cfg)
    from .alpha_field import alpha_on_points

    alpha = alpha_on_points(grid.local_points, molecule, scf_cfg)   # float if uniform
    chunk = choose_strategy(grid, n_basis, grid_cfg.chunk_size,
                            grid_cfg.memory_budget_mb)
    terms = GridTerms(basis, molecule, grid, chunk_size=chunk,
                      cutoff=grid_cfg.basis_cutoff, budget_mb=grid_cfg.memory_budget_mb)
    chi = terms.chi
    weights = grid.local_weights
    if verbose:
        sp = terms.sparsity
        if sp is not None:
            how = (f"block-sparse, {sp['kept_fraction']:.0%} kept, "
                   + ("cached" if sp["cached"] else "rebuilt per pass"))
        else:
            how = ("cached" if terms.cached
                   else f"chunked, {chunk:,} points at a time")
        mpi_print(f"  DV grid            : {len(grid)} points "
                  f"({_radial_summary(grid)}r x {grid.n_ang(grid.n_a)} {grid.angular} "
                  f"angular per atom), "
                  f"chi {sp['chi_mb'] if sp else grid.chi_memory_mb(n_basis, chunk):.1f} "
                  f"MB/rank ({how})")
        if verbose > 1:
            from ..grid.summary import grid_summary, summary_lines

            for line in summary_lines(grid_summary(config, grid, molecule, terms)):
                mpi_print(line)

    with timer("scf.v_nuc"):
        v_nuc = nuclear_potential(grid.local_points, molecule.positions, molecule.charges)
        V_nuc = terms.potential_matrix(v_nuc)
    E_nn = molecule.nuclear_repulsion()

    # ---- point-charge embedding: a fixed one-electron potential (ADR 0010) --
    from .. import embedding as emb_mod

    ext = emb_mod.from_config(config)
    V_ext = None
    E_ext_nuc = 0.0
    if ext is not None:
        with timer("scf.embedding"):
            V_ext = terms.potential_matrix(ext.potential(grid.local_points))
            E_ext_nuc = ext.nuclear_energy(molecule)
        V_nuc = V_nuc + V_ext
        E_nn = E_nn + E_ext_nuc
        if verbose:
            mpi_print(f"  embedding          : {len(ext)} point charges, total "
                      f"{ext.total_charge:+.4f}, sigma {ext.sigma:g} Bohr")

    # ---- initial density --------------------------------------------------
    from . import checkpoint as ckpt_mod

    system_fp = ckpt_mod.system_fingerprint(molecule, basis, config, unrestricted)
    geometry_fp = ckpt_mod.geometry_fingerprint(molecule)
    start_it = 0
    if restart is not None:
        initial, start_it = ckpt_mod.initial_density(
            ckpt_mod.read(restart), system_fp, geometry_fp, n_basis, path=restart,
            verbose=verbose)
    if initial is not None and initial.get("P") is not None:
        P = rescale_to_electron_count(np.array(initial["P"], dtype=float), S, n_electrons)
    else:
        P = rescale_to_electron_count(
            sad_density(molecule, basis, scf_cfg.alpha_for, grid_cfg.atom["n_points"],
                        (grid_cfg.guess["n_r"], grid_cfg.guess["n_a"])),
            S, n_electrons)
    if unrestricted:
        if initial is not None and all(initial.get(k) is not None for k in ("P_a", "P_b")):
            P_a = rescale_to_electron_count(np.array(initial["P_a"], dtype=float), S, n_alpha)
            P_b = rescale_to_electron_count(np.array(initial["P_b"], dtype=float), S, n_beta)
        else:
            P_a, P_b = spin_split(P, molecule, basis, n_alpha, n_beta)
        P = P_a + P_b
        accel = {
            "diis_ab": SpinPulayDIIS(),
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
    # With smearing on, the variational quantity is the free energy
    # F = E_total - sigma*S, not E_total, so that is what convergence is tested
    # on.  `ts_current` is the entropy of the occupations that produced the P
    # being evaluated, and stays 0 when smearing is off.
    ts_current = 0.0
    l_max = grid_cfg.l_max

    if verbose:
        _print_table_header(unrestricted)

    every = int(scf_cfg.checkpoint_every)
    for it in range(start_it + 1, scf_cfg.max_iter + 1):
        with timer("scf.iteration"):
            # --- density on the grid, from the current P -------------------
            with timer("scf.density"):
                if unrestricted:
                    rho_a = terms.density(P_a)
                    rho_b = terms.density(P_b)
                    rho = rho_a + rho_b
                else:
                    rho_a = rho_b = None
                    rho = terms.density(P)

            # --- Hartree potential (needs the density on the whole grid) ---
            with timer("scf.hartree"):
                rho_full = _gather_full(grid, rho)
                V_H = hartree_potential(grid, rho_full, molecule.positions,
                                        l_max=l_max, eval_points=grid.local_points)

            # --- Coulomb plus exchange-correlation, in one pass over the grid.
            # Only J + Vxc enters the Fock matrix (the energy is a grid
            # integral), so each spin channel needs one matrix, not two.
            with timer("scf.xc"):
                if unrestricted:
                    built = terms.potential_matrices({
                        "a": V_H + xc.potential_spin(rho_a, alpha),
                        "b": V_H + xc.potential_spin(rho_b, alpha),
                    })
                    JVxc_a, JVxc_b = built["a"], built["b"]
                else:
                    JVxc = terms.potential_matrices({"v": V_H + xc.potential(rho, alpha)})["v"]

            energies = _total_energy(P, T, V_nuc, rho, V_H, weights, alpha, E_nn,
                                     rho_a, rho_b)
            e_current = energies["E_total"] - ts_current
            dE = abs(e_current - e_prev)
            e_prev = e_current

            # --- Fock matrices and extrapolation ---------------------------
            if unrestricted:
                F_a = T + V_nuc + JVxc_a
                F_b = T + V_nuc + JVxc_b
                commutator = max(_commutator_rms(F_a, P_a, S, X_keep),
                                 _commutator_rms(F_b, P_b, S, X_keep))
                Fa_use, Fb_use, method, err, mix = _accelerate_uks(
                    accel, F_a, F_b, P_a, P_b, S, energies["E_total"], it, scf_cfg, dE)
                eps_a, C_a, eps_b, C_b = _diagonalize_uks(Fa_use, Fb_use, S, n_basis)
                occ_a = fermi_occupations(eps_a, n_alpha, scf_cfg.smearing, 1.0)
                occ_b = fermi_occupations(eps_b, n_beta, scf_cfg.smearing, 1.0)
                if holes is not None:
                    if ts_cfg.spin == "alpha":
                        occ_a = holes.apply(eps_a, occ_a, C_a)
                    else:
                        occ_b = holes.apply(eps_b, occ_b, C_b)
                Pa_new = density_matrix(C_a, occ_a)
                Pb_new = density_matrix(C_b, occ_b)
                if holes is not None and holes.correction is not None:
                    if ts_cfg.spin == "alpha":
                        Pa_new = Pa_new - holes.correction
                    else:
                        Pb_new = Pb_new - holes.correction
                d_rms = float(np.sqrt(np.mean((Pa_new + Pb_new - P) ** 2)))
                P_a = mix * Pa_new + (1.0 - mix) * P_a
                P_b = mix * Pb_new + (1.0 - mix) * P_b
                P = P_a + P_b
                eps, C, occ = eps_a, C_a, occ_a
                if scf_cfg.smearing > 0:
                    ts_current = (smearing_entropy(occ_a, scf_cfg.smearing, 1.0)
                                  + smearing_entropy(occ_b, scf_cfg.smearing, 1.0))
                extra = (float(eps_a[np.argsort(eps_a)][max(int(round(n_alpha)) - 1, 0)]),
                         float(eps_b[np.argsort(eps_b)][max(int(round(n_beta)) - 1, 0)]))
            else:
                F = T + V_nuc + JVxc
                commutator = _commutator_rms(F, P, S, X_keep)
                F_use, method, err, mix = _accelerate_rks(
                    accel, F, P, S, energies["E_total"], it, scf_cfg, dE)
                eps, C, _ = solve_generalized(F_use, S, scf_cfg.s_threshold)
                occ = fermi_occupations(eps, n_electrons, scf_cfg.smearing, 2.0)
                P_new = density_matrix(C, occ)
                d_rms = float(np.sqrt(np.mean((P_new - P) ** 2)))
                P = mix * P_new + (1.0 - mix) * P
                if scf_cfg.smearing > 0:
                    ts_current = smearing_entropy(occ, scf_cfg.smearing, 2.0)
                homo = float(eps[occ > 1e-8].max()) if np.any(occ > 1e-8) else float("nan")
                extra = (homo * HARTREE_TO_EV,)

            bcast_array(P)
            history.append({"iter": it, "E": energies["E_total"], "E_free": e_current,
                            "dE": dE, "error": err, "d_rms": d_rms,
                            "commutator": commutator, "method": method, "mix": mix})
            if verbose:
                _print_row(it, energies["E_total"], dE, err, d_rms, extra, method)

            # The commutator test is what makes this self-consistency: a stalled
            # DIIS can hold the density still while F and P do not commute
            # (ADR 0014; N2 1s transition state, 45 meV off at the defaults).
            if (it > start_it + 2 and dE < scf_cfg.e_tol and d_rms < scf_cfg.d_tol
                    and commutator < scf_cfg.c_tol):
                converged = True
                break
            if checkpoint_path is not None and every > 0 and it % every == 0:
                ckpt_mod.write(checkpoint_path, P=P, P_a=P_a, P_b=P_b, C=C, eps=eps,
                               occ=occ, iteration=it, system=system_fp,
                               geometry=geometry_fp, energy=energies["E_total"],
                               converged=False)

    # ---- final quantities from the converged density ----------------------
    if unrestricted:
        rho_a = terms.density(P_a)
        rho_b = terms.density(P_b)
        rho = rho_a + rho_b
    else:
        rho_a = rho_b = None
        rho = terms.density(P)
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
    if ext is not None:
        # already inside E_nuclear_attraction / E_nn; reported separately
        energies["E_embedding_electrons"] = float(np.einsum("mn,nm->", P, V_ext))
        energies["E_embedding_nuclei"] = E_ext_nuc
        radius = config.embedding.spill_radius * ANG_TO_BOHR
        energies["embedding_spill_electrons"] = allreduce_scalar(
            ext.electrons_near(grid.local_points, weights, rho, radius))
    n_expected = n_electrons - (holes.electrons if holes is not None and holes.mode == "remove"
                                else 0.0)
    if checkpoint_path is not None:
        ckpt_mod.write(checkpoint_path, P=P, P_a=P_a, P_b=P_b, C=C, eps=eps, occ=occ,
                       iteration=start_it + len(history), system=system_fp,
                       geometry=geometry_fp, energy=energies["E_total"], converged=converged)
    energies["trace_PS"] = float(np.einsum("mn,nm->", P, S))

    if verbose:
        _print_footer(converged, len(history), energies, n_expected)
        if not converged:
            for line in diagnose_non_convergence(history, eps, occ, scf_cfg):
                mpi_print(f"  {line}")

    return SCFResult(
        converged=converged, n_iterations=len(history), energies=energies,
        eps=eps, C=C, P=P, occupations=occ, S=S, T=T, V_nuc=V_nuc,
        grid=grid, chi=chi, terms=terms, molecule=molecule, basis=basis,
        unrestricted=unrestricted, n_alpha=n_alpha, n_beta=n_beta,
        eps_a=eps_a, eps_b=eps_b, C_a=C_a, C_b=C_b, P_a=P_a, P_b=P_b,
        occ_a=occ_a, occ_b=occ_b, history=history,
        transition_state=None if holes is None else dict(holes.last, mode=holes.mode,
                                                         spin=ts_cfg.spin,
                                                         electrons=holes.electrons),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _radial_summary(grid) -> str:
    """'30' for a uniform grid, '25-47' when radial sizes differ by element."""
    lo, hi = int(min(grid.n_r_atom)), int(max(grid.n_r_atom))
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _gather_full(grid, rho_local: np.ndarray) -> np.ndarray:
    """Assemble the full-grid density from the rank-local slices."""
    from ..parallel import PARALLEL

    if not PARALLEL:
        return rho_local
    full = np.zeros(len(grid))
    full[grid.local_slice] = rho_local
    return allreduce_array(full)


def _commutator_rms(F: np.ndarray, P: np.ndarray, S: np.ndarray, X: np.ndarray) -> float:
    """rms of X^T (F P S - S P F) X for the F built from P: zero at self-consistency.

    X is the canonical orthogonalizer of S, so this is the commutator in the
    orthonormal space the SCF actually diagonalizes in. Directions dropped for
    near-linear dependence (s_threshold) can carry a commutator the SCF cannot
    remove: the MgO6H example (9 of 132 dropped) sits at 1.4e-4 unprojected
    for ever while the projected value falls to 4e-7 by iteration 15 (ADR 0014).
    F is the unextrapolated Fock matrix, available every iteration (EDIIS too).
    """
    e = F @ P @ S
    e = X.T @ (e - e.T) @ X
    return float(np.sqrt(np.mean(e ** 2)))


def _diis_mixing(cfg, extrapolated: bool, error: float) -> float:
    """How much of the freshly built density to accept.

    A successful Pulay extrapolation already contains the convergence
    acceleration, so the new density is taken *whole* (mix = 1).  Damping on top
    of DIIS simply multiplies the residual by (1 - mix) every iteration and
    throws the acceleration away -- the slow geometric decay that behavior
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
    # one coefficient set for both channels (ADR 0012, F11)
    accel["diis_ab"].push(F_a, F_b, P_a, P_b, S)
    ab = accel["diis_ab"].extrapolate()
    err = accel["diis_ab"].error
    mix = _diis_mixing(cfg, ab is not None, err)
    if ab is None:
        return F_a, F_b, "DIIS", err, mix
    return ab[0], ab[1], "DIIS", err, mix


def _diagonalize_uks(F_a, F_b, S, n_basis):
    eps_a, C_a, _ = solve_generalized(F_a, S)
    eps_b, C_b, _ = solve_generalized(F_b, S)
    return eps_a, C_a, eps_b, C_b


def _print_header(molecule, basis, config, unrestricted, n_alpha, n_beta):
    mpi_print("=" * 78)
    mpi_print(f"  pyxany  --  DV-Xalpha SCF   [{molecule.name}]")
    mpi_print("=" * 78)
    mpi_print(f"  {molecule.summary()}")
    mpi_print(f"  Basis              : {config.basis.set.upper()}  "
              f"({len(basis)} functions, max l = {max(f.l for f in basis)})")
    mode = (f"UKS  n_alpha={n_alpha:g} n_beta={n_beta:g}" if unrestricted
            else f"RKS  n_occ={n_alpha:g}")
    mpi_print(f"  Method             : {mode}   alpha_xc = {config.scf.alpha_label}")


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


def diagnose_non_convergence(history, eps, occ, scf_cfg) -> list[str]:
    """Explain *why* an SCF stalled, with the setting that usually fixes it.

    Three distinguishable situations, checked in order:

    1. the energy is still moving -- it simply ran out of iterations;
    2. the energy has settled but the density has not, and the frontier
       orbitals are nearly degenerate -- occupation swaps between them from
       one iteration to the next, which is what Fermi smearing is for;
    3. the energy has settled but the density has not, with a healthy gap --
       the mixing is too aggressive.

    Blaming degeneracy in case 1 would be wrong, so the energy-stability test
    comes first.
    """
    lines: list[str] = ["", ""]
    recent = np.array([h["E"] for h in history[-20:]])
    spread = float(recent.std()) if len(recent) > 2 else float("inf")
    drift = float(abs(recent[-1] - recent[0])) if len(recent) > 2 else float("inf")
    energy_settled = spread < 1000.0 * scf_cfg.e_tol and drift < 1e4 * scf_cfg.e_tol

    gap = float("nan")
    if eps is not None and occ is not None:
        finite = np.isfinite(eps)
        filled = eps[finite & (occ > 1e-8)]
        empty = eps[finite & (occ <= 1e-8)]
        if len(filled) and len(empty):
            gap = float(empty.min() - filled.max())

    last = history[-1] if history else {}
    density_settled = last.get("d_rms", float("inf")) < scf_cfg.d_tol
    commutator = last.get("commutator", 0.0)
    if energy_settled and density_settled and commutator >= scf_cfg.c_tol:
        lines += [
            "  [diagnosis] the energy and density have settled but F and P do not",
            f"  commute yet (rms {commutator:.1e} > scf.c_tol {scf_cfg.c_tol:g}): the",
            "  SCF is creeping along a slow mode rather than diverging. A core hole on",
            "  one of several equivalent atoms does this (ADR 0014); raise",
            "  scf.max_iter rather than loosening scf.c_tol, which would accept a",
            "  result that is not self-consistent.",
        ]
        return lines
    if not energy_settled:
        lines += [
            f"  [diagnosis] the energy is still moving ({spread:.1e} Ha spread over the",
            f"  last {len(recent)} iterations). It ran out of iterations rather than",
            "  stalling; raise scf.max_iter, or lower scf.mix_ediis if the early",
            "  iterations are swinging.",
        ]
    elif np.isfinite(gap) and gap < 0.02 and scf_cfg.smearing <= 0.0:
        width = float(np.clip(2.0 * max(gap, 1e-3), 0.002, 0.02))
        lines += [
            f"  [diagnosis] the energy is stable to {spread:.1e} Ha but the density is",
            f"  not settling, and the HOMO-LUMO gap is only "
            f"{gap * HARTREE_TO_EV:.4f} eV ({gap:.2e} Ha).",
            "  Near-degenerate frontier orbitals let the occupation swap between",
            "  iterations. Fermi smearing removes the ambiguity:",
            "",
            f"      pyxany run <input.yaml> --set scf.smearing={width:g}",
            "",
            "  Report E_free rather than E_total when smearing is on, and check that",
            "  the result is insensitive to the width you chose.",
        ]
    else:
        gap_note = (f" (gap {gap * HARTREE_TO_EV:.3f} eV)" if np.isfinite(gap) else "")
        lines += [
            f"  [diagnosis] the energy is stable to {spread:.1e} Ha but the density is",
            f"  not settling{gap_note}. Try a smaller scf.mix_diis (currently",
            f"  {scf_cfg.mix_diis:g}), or scf.smearing > 0 if the frontier is crowded.",
        ]
    return lines


# Relative disagreement between int rho dV and Tr(PS) above which the DV grid is
# too coarse for the basis to be integrated consistently.  Water on the default
# grid sits at 6.4e-6.
_GRID_CONSISTENCY_WARN = 1e-4


def _print_footer(converged, n_it, energies, n_electrons):
    mpi_print("  " + "-" * 92)
    status = "CONVERGED" if converged else "NOT CONVERGED"
    mpi_print(f"  [{status}] after {n_it} iterations")
    n_int = energies["n_electrons_integrated"]
    trace = energies["trace_PS"]
    mpi_print(f"  int rho dV = {n_int:.6f}   Tr(PS) = {trace:.6f}   "
              f"(expected {n_electrons:g})")
    # S and T are built on their own two-center grids and with exact one-center
    # formulas, while rho, V_H and V_xc live on the DV molecular grid.  The two
    # therefore disagree slightly about how many electrons there are, and the
    # charge that generates V_H is not exactly the charge the Fock matrix
    # represents.  It is normally far below every other error, but a large gap
    # means the DV grid is too coarse for this basis and the energy is not a
    # clean functional of one density.
    mismatch = abs(n_int - trace) / max(abs(n_electrons), 1.0)
    if mismatch > _GRID_CONSISTENCY_WARN:
        mpi_print(f"  [warning] the DV grid and the overlap matrix disagree on the "
                  f"electron count by {abs(n_int - trace):.2e} "
                  f"({mismatch:.1e} relative).")
        mpi_print("            V_H is built from a density normalized differently "
                  "from Tr(PS); raise grid.n_r / grid.n_a until this shrinks.")
