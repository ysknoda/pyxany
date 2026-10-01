"""Analytic nuclear forces: Hellmann-Feynman plus Pulay basis-derivative terms.

With atom-centred basis functions the total energy depends on a nuclear position
in three ways, and all three must be differentiated:

1. **Hellmann-Feynman** -- the ``-Z_A / |r - R_A|`` operator itself moves.
2. **Pulay** -- the basis functions centred on A move, changing every matrix
   element they enter and the density they build.
3. **Nuclear repulsion** -- classical.

Writing the Kohn-Sham operator as ``h_eff = -1/2 lap + v_nuc + V_H + v_xc`` and
the energy-weighted density matrix as ``W`` (the Lagrange multipliers enforcing
orbital orthonormality), the gradient is::

    dE/dR_A = - Z_A int rho(r) (r - R_A)/|r - R_A|^3 dr          (Hellmann-Feynman)
              - 2 sum_{mu in A, nu} P_{mu nu} int D_mu (h_eff chi_nu) dr   (Pulay)
              + 2 sum_{mu in A, nu} W_{mu nu} int D_mu chi_nu dr           (Pulay, S)
              + dE_nn/dR_A

where ``D_mu = grad chi_mu`` with respect to the *electron* coordinate (the
derivative with respect to the atom position is its negative).

Because ``h_eff`` is applied to ``chi_nu`` directly -- using the analytic
Laplacian rather than a difference formula -- no finite differences appear
anywhere in this routine.

Grid-derivative term
--------------------
The expressions above are the exact gradient of the *continuum* energy
functional.  The computed energy is a quadrature of it on a grid whose points
and Becke weights both move with the nuclei, so the true derivative of the
computed number carries an extra contribution

    sum_k [ dw_k/dR_B * G(r_k) + w_k * grad G(r_k) . dr_k/dR_B ]

with ``G = rho (v_nuc + V_H/2 + eps_xc)``.  For an exact quadrature that sum
vanishes identically -- it *is* the derivative of the quadrature error, and it
shrinks as the grid is refined.

Two things are done about it here rather than hoping it is small:

* its rigid-translation component, which is the dominant part, is removed
  exactly by :func:`project_out_translation` (enabled by default), restoring
  ``sum_A F_A = 0``;
* what remains is measurable -- ``pyxany forces --compare`` prints it against
  central finite differences, and docs/forces.md tabulates how it falls with
  grid density.
"""

from __future__ import annotations

import numpy as np

from ..basis.evaluate import chi_grad_matrix, chi_matrix
from ..integrals import xc
from ..integrals.coulomb import hartree_potential, nuclear_potential
from ..integrals.overlap import overlap_kinetic_gradient
from ..parallel import allreduce_array
from ..scf.linalg import energy_weighted_density
from ..timing import timer


def nuclear_repulsion_gradient(molecule) -> np.ndarray:
    """``dE_nn/dR`` for every atom, shape (n_atoms, 3)."""
    pos = molecule.positions
    Z = molecule.charges
    n = len(Z)
    grad = np.zeros((n, 3))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            d = pos[a] - pos[b]
            r = np.linalg.norm(d)
            if r < 1e-12:
                continue
            grad[a] -= Z[a] * Z[b] * d / r**3
    return grad


def hellmann_feynman_gradient(molecule, points, weights, rho) -> np.ndarray:
    """Explicit ``dE_ne/dR_A`` from the moving nuclear-attraction operator."""
    grad = np.zeros((len(molecule), 3))
    for ai, atom in enumerate(molecule):
        d = points - atom.pos
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-10)
        kernel = -atom.Z * weights * rho / r**3
        grad[ai] = np.einsum("k,kx->x", kernel, d)
    return grad


def compute_forces(result, config, project: bool | None = None) -> np.ndarray:
    """Analytic forces (Hartree/Bohr), shape (n_atoms, 3).

    ``result`` is a converged :class:`~pyxany.scf.SCFResult`.

    ``project`` overrides ``config.forces.project_translation``.  Pass ``False``
    when any atom is held fixed: removing the *all-atom* mean is only legitimate
    while the energy is translation-invariant, which it is not once part of the
    molecule cannot move.
    """
    with timer("forces.analytic"):
        molecule, basis, grid = result.molecule, result.basis, result.grid
        positions = molecule.positions
        pts = grid.local_points
        wts = grid.local_weights
        alpha = config.scf.alpha
        # The gradient block D is (n_points, n_basis, 3) -- three times the size
        # of chi -- so the force evaluation always builds chi explicitly here
        # rather than reusing a cached one, and is the memory high-water mark of
        # a run.  See docs/hpc.md.
        chi = result.chi if result.chi is not None else chi_matrix(basis, positions, pts)

        # -- effective potential on the grid, from the converged density ----
        if result.unrestricted:
            rho_a = np.einsum("km,mn,kn->k", chi, result.P_a, chi, optimize=True)
            rho_b = np.einsum("km,mn,kn->k", chi, result.P_b, chi, optimize=True)
            rho = rho_a + rho_b
        else:
            rho_a = rho_b = None
            rho = np.einsum("km,mn,kn->k", chi, result.P, chi, optimize=True)

        rho_full = _full_density(grid, rho)
        V_H = hartree_potential(
            grid, rho_full, positions, l_max=config.grid.l_max, eval_points=pts
        )
        v_nuc = nuclear_potential(pts, positions, molecule.charges)

        with timer("forces.gradients"):
            D = chi_grad_matrix(basis, positions, pts)  # (npts, nb, 3)

        grad = np.zeros((len(molecule), 3))

        # -- Hellmann-Feynman + nuclear repulsion ---------------------------
        grad += hellmann_feynman_gradient(molecule, pts, wts, rho)

        # -- Pulay: h_eff acting on chi_nu ----------------------------------
        by_atom: dict[int, list[int]] = {}
        for mu, f in enumerate(basis):
            by_atom.setdefault(f.atom_idx, []).append(mu)

        if result.unrestricted:
            spins = (
                (
                    result.P_a,
                    xc.potential_spin(rho_a, alpha),
                    result.eps_a,
                    result.C_a,
                    result.occ_a,
                ),
                (
                    result.P_b,
                    xc.potential_spin(rho_b, alpha),
                    result.eps_b,
                    result.C_b,
                    result.occ_b,
                ),
            )
        else:
            spins = (
                (result.P, xc.potential(rho, alpha), result.eps, result.C, result.occupations),
            )

        W_total = np.zeros_like(result.P)
        for P_s, v_xc_s, eps_s, C_s, occ_s in spins:
            v_eff = v_nuc + V_H + v_xc_s
            # Only the potential matrices are built on the DV grid, so only they
            # are differentiated on it; T and S come from their own builders and
            # are differentiated by overlap_kinetic_gradient below.
            v_chi = v_eff[:, None] * chi  # (npts, nb)
            W_total += energy_weighted_density(C_s, eps_s, occ_s)
            for ai, idx in by_atom.items():
                Pblk = P_s[np.ix_(idx, range(len(basis)))]  # (n_A, nb)
                Dw = D[:, idx, :] * wts[:, None, None]  # (npts, n_A, 3)
                M_v = np.einsum("kax,kn->anx", Dw, v_chi, optimize=True)
                grad[ai] -= 2.0 * np.einsum("an,anx->x", Pblk, M_v, optimize=True)


        # -- grid motion: Becke weights and atom-centred points move with R --
        with timer("forces.grid"):
            grad += _grid_motion_gradient(result, config, chi, D, rho, rho_a, rho_b, V_H, v_nuc)

        allreduce_array(grad)
        # -- Pulay terms of T and S, from the matrices the SCF used. Added after
        # the reduction: overlap_kinetic_gradient already returns the global sum.
        grad += overlap_kinetic_gradient(
            molecule,
            basis,
            result.P,
            W_total,
            n_r=config.grid.pair_n_r,
            n_a=config.grid.pair_n_a,
            screening_tol=config.grid.screening_tol,
        )
        grad += nuclear_repulsion_gradient(molecule)
        forces = -grad
        # Keep the pre-projection residual: it is a direct measure of the
        # grid-derivative error, and therefore of whether a requested force
        # convergence threshold is resolvable at all.  After projection
        # |sum F| is ~1e-16 by construction and says nothing.
        compute_forces.last_translation_residual = float(np.linalg.norm(forces.sum(axis=0)))
        wanted = getattr(config.forces, "project_translation", True) if project is None else project
        if wanted:
            forces = project_out_translation(forces)
        return forces


def _grid_motion_gradient(result, config, chi, D, rho, rho_a, rho_b, V_H, v_nuc):
    """Derivative of the quadrature itself: ``sum_k d(w_k G(r_k)) / d R_B``.

    The grid-sampled energy is ``sum_k w_k G(r_k)`` with
    ``G = rho (v_nuc + V_H) + rho eps_xc`` -- V_H enters without the factor 1/2
    because the Hartree energy is a symmetric double sum over grid points, so
    moving one point changes both its source and its evaluation role. Every
    point is rigidly attached to its owner atom and its Becke weight depends on
    all nuclei, giving two contributions:

    * weight term  ``sum_k raw_k G_k d cell_k / d R_B``, with the owner's own
      derivative from translational invariance (``-sum_{B != A}``);
    * point term   ``sum_{k owned by B} w_k grad G(r_k)``, where
      ``grad G = grad rho (v_nuc + V_H + v_xc) + rho (grad v_nuc + grad V_H)``.

    Rank-local; the caller reduces.
    """
    from ..grid.becke import own_cell_weight_gradient

    molecule, grid = result.molecule, result.grid
    alpha = config.scf.alpha
    n_atoms = len(molecule)
    grad = np.zeros((n_atoms, 3))
    if n_atoms == 1 or grid.partition != "becke":
        return grad

    sl = grid.local_slice
    pts, wts = grid.local_points, grid.local_weights
    owner = grid.owner[sl]

    # energy density and its gradient on the grid
    if result.unrestricted:
        e_xc = 2.0 * rho_a * xc.energy_density(
            2.0 * rho_a, alpha
        ) + 2.0 * rho_b * xc.energy_density(2.0 * rho_b, alpha)
        grad_rho_a = 2.0 * np.einsum("kmx,mn,kn->kx", D, result.P_a, chi, optimize=True)
        grad_rho_b = 2.0 * np.einsum("kmx,mn,kn->kx", D, result.P_b, chi, optimize=True)
        grad_rho = grad_rho_a + grad_rho_b
        xc_part = (
            xc.potential_spin(rho_a, alpha)[:, None] * grad_rho_a
            + xc.potential_spin(rho_b, alpha)[:, None] * grad_rho_b
        )
    else:
        e_xc = rho * xc.energy_density(rho, alpha)
        grad_rho = 2.0 * np.einsum("kmx,mn,kn->kx", D, result.P, chi, optimize=True)
        xc_part = xc.potential(rho, alpha)[:, None] * grad_rho
    G = rho * (v_nuc + V_H) + e_xc

    grad_v_nuc = np.zeros_like(pts)
    for pos, Z in zip(molecule.positions, molecule.charges):
        d = pts - pos
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        grad_v_nuc += Z * d / r[:, None] ** 3

    # grad V_H by central differences of the Poisson solution at shifted points
    rho_full = _full_density(grid, rho)
    h = 1e-4
    grad_V_H = np.zeros_like(pts)
    for x in range(3):
        e = np.zeros(3)
        e[x] = h
        vp = hartree_potential(
            grid, rho_full, molecule.positions, l_max=config.grid.l_max, eval_points=pts + e
        )
        vm = hartree_potential(
            grid, rho_full, molecule.positions, l_max=config.grid.l_max, eval_points=pts - e
        )
        grad_V_H[:, x] = (vp - vm) / (2.0 * h)

    grad_G = grad_rho * (v_nuc + V_H)[:, None] + xc_part + rho[:, None] * (grad_v_nuc + grad_V_H)
    for A in range(n_atoms):
        sel = owner == A
        grad[A] += wts[sel] @ grad_G[sel]

    radii = np.array([a.grid_radius() for a in molecule])
    M = own_cell_weight_gradient(pts, owner, grid.raw_weights[sl] * G, molecule.positions, radii)
    for A in range(n_atoms):
        for B in range(n_atoms):
            if B != A:
                grad[B] += M[A, B]
                grad[A] -= M[A, B]
    return grad


def _full_density(grid, rho_local):
    from ..parallel import PARALLEL

    if not PARALLEL:
        return rho_local
    full = np.zeros(len(grid))
    full[grid.local_slice] = rho_local
    return allreduce_array(full)


def project_out_translation(forces: np.ndarray) -> np.ndarray:
    """Remove the net translational component, which must vanish exactly.

    Translational invariance requires ``sum_A F_A = 0``.  Any residual is
    numerical -- here, the rigid-shift part of the grid-derivative term -- so
    subtracting the mean is not a cosmetic fix but the removal of a known
    error component.

    .. warning::

       This is only valid when **every** atom is free.  Translating a subset of
       a molecule does change the energy, so with a freeze mask in force the
       all-atom mean is not an error component and subtracting it injects a
       uniform spurious force into the moving atoms -- measured at 1.8e-3
       Ha/Bohr for water on the default grid, against a default
       ``optimise.f_max`` of 1e-3.  :func:`pyxany.forces.optimise_geometry`
       therefore turns it off whenever anything is frozen.
    """
    return forces - forces.mean(axis=0)


def translation_residual(forces: np.ndarray) -> float:
    """``|sum_A F_A|`` -- should vanish by translational invariance."""
    return float(np.linalg.norm(forces.sum(axis=0)))
