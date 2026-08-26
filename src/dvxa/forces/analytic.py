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
* what remains is measurable -- ``dvxa forces --compare`` prints it against
  central finite differences, and docs/forces.md tabulates how it falls with
  grid density.
"""

from __future__ import annotations

import numpy as np

from ..basis.evaluate import chi_grad_matrix, chi_laplacian_matrix, chi_matrix
from ..integrals import xc
from ..integrals.coulomb import hartree_potential, nuclear_potential
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


def compute_forces(result, config) -> np.ndarray:
    """Analytic forces (Hartree/Bohr), shape (n_atoms, 3).

    ``result`` is a converged :class:`~dvxa.scf.SCFResult`.
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
        V_H = hartree_potential(grid, rho_full, positions,
                                l_max=config.grid.poisson_l_max, eval_points=pts)
        v_nuc = nuclear_potential(pts, positions, molecule.charges)

        with timer("forces.gradients"):
            D = chi_grad_matrix(basis, positions, pts)        # (npts, nb, 3)
            L = chi_laplacian_matrix(basis, positions, pts)   # (npts, nb)

        grad = np.zeros((len(molecule), 3))

        # -- Hellmann-Feynman + nuclear repulsion ---------------------------
        grad += hellmann_feynman_gradient(molecule, pts, wts, rho)

        # -- Pulay: h_eff acting on chi_nu ----------------------------------
        by_atom: dict[int, list[int]] = {}
        for mu, f in enumerate(basis):
            by_atom.setdefault(f.atom_idx, []).append(mu)

        if result.unrestricted:
            spins = ((result.P_a, xc.potential_spin(rho_a, alpha), result.eps_a,
                      result.C_a, result.occ_a),
                     (result.P_b, xc.potential_spin(rho_b, alpha), result.eps_b,
                      result.C_b, result.occ_b))
        else:
            spins = ((result.P, xc.potential(rho, alpha), result.eps,
                      result.C, result.occupations),)

        for P_s, v_xc_s, eps_s, C_s, occ_s in spins:
            v_eff = v_nuc + V_H + v_xc_s
            # h_eff chi_nu evaluated on the grid
            h_chi = -0.5 * L + v_eff[:, None] * chi                      # (npts, nb)
            W = energy_weighted_density(C_s, eps_s, occ_s)
            for ai, idx in by_atom.items():
                Pblk = P_s[np.ix_(idx, range(len(basis)))]               # (n_A, nb)
                Wblk = W[np.ix_(idx, range(len(basis)))]
                # int D_mu (h_eff chi_nu) and int D_mu chi_nu
                Dw = D[:, idx, :] * wts[:, None, None]                   # (npts, n_A, 3)
                M_h = np.einsum("kax,kn->anx", Dw, h_chi, optimize=True)
                M_s = np.einsum("kax,kn->anx", Dw, chi, optimize=True)
                grad[ai] -= 2.0 * np.einsum("an,anx->x", Pblk, M_h, optimize=True)
                grad[ai] += 2.0 * np.einsum("an,anx->x", Wblk, M_s, optimize=True)

        allreduce_array(grad)
        grad += nuclear_repulsion_gradient(molecule)
        forces = -grad
        if getattr(config.forces, "project_translation", True):
            forces = project_out_translation(forces)
        return forces


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
    """
    return forces - forces.mean(axis=0)


def translation_residual(forces: np.ndarray) -> float:
    """``|sum_A F_A|`` -- should vanish by translational invariance."""
    return float(np.linalg.norm(forces.sum(axis=0)))
