"""Element-specific X-alpha parameters on the grid (ADR 0011).

Each atom carries its own alpha; at a point the parameter is the Becke-weighted
mean

    alpha(r) = sum_A w_A(r) alpha_A,     sum_A w_A = 1,

so it equals alpha_A deep inside atom A's cell and changes smoothly between
atoms. A uniform setting returns a plain float and leaves every code path
exactly as before.

The weights move with the nuclei, so the exchange energy picks up a force term;
:func:`alpha_mixture_gradient` gives ``sum_k c_k d alpha(r_k) / d R_B`` for grid
points that move rigidly with their owner atom.
"""

from __future__ import annotations

import numpy as np

from ..grid.becke import DEFAULT_STIFFNESS, becke_weights, size_adjustment


def atom_alphas(molecule, scf_cfg) -> np.ndarray:
    """alpha of every atom; pseudo-hydrogens take hydrogen's value."""
    return np.array([scf_cfg.alpha_for("H" if a.is_pseudo else a.element) for a in molecule])


def alpha_on_points(points, molecule, scf_cfg, block: int = 100_000):
    """``alpha(r)`` at ``points``: a float when uniform, else an array."""
    uniform = scf_cfg.uniform_alpha
    if uniform is not None:
        return uniform
    alphas = atom_alphas(molecule, scf_cfg)
    if np.all(alphas == alphas[0]):
        return float(alphas[0])
    radii = np.array([a.grid_radius() for a in molecule])
    pts = np.asarray(points, dtype=float)
    from ..kernels import alpha_mix

    compiled = alpha_mix(pts, np.asarray(molecule.positions, dtype=float),
                         size_adjustment(radii), alphas, DEFAULT_STIFFNESS)
    if compiled is not None:
        return compiled
    out = np.empty(len(pts))
    for lo in range(0, len(pts), block):
        w = becke_weights(pts[lo : lo + block], molecule.positions, radii)
        out[lo : lo + block] = w @ alphas
    return out


def _smooth_step_and_derivative(nu, stiffness):
    f = np.clip(nu, -1.0, 1.0)
    df = np.ones_like(f)
    for _ in range(stiffness):
        df = df * 1.5 * (1.0 - f * f)
        f = 1.5 * f - 0.5 * f * f * f
    return 0.5 * (1.0 - f), -0.5 * df


def alpha_point_gradient(points, molecule, alphas, stiffness: int = DEFAULT_STIFFNESS):
    """``g[k, B] = d alpha(r_k) / d R_B`` with the points held fixed, (n, n_atoms, 3).

    With ``P_C = prod_{D != C} s(nu_CD)``, ``Z = sum_C P_C`` and
    ``alpha = sum_C alpha_C P_C / Z``:

        d alpha / d R_X = sum_C (alpha_C - alpha) / Z * dP_C / dR_X,
        dP_C / dR_X = P_C sum_{D != C} (s'/s)_CD dnu_CD / dR_X,

    ``nu_CD = mu + a_CD (1 - mu^2)``, ``mu = (d_C - d_D) / R_CD``. Where a factor
    s vanishes so does s' (the smoothing polynomial is flat at nu = +-1), so the
    ratio is taken as zero there.
    """
    pts = np.asarray(points, dtype=float)
    C = np.asarray(molecule.positions, dtype=float)
    n, n_at = len(pts), len(C)
    radii = np.array([a.grid_radius() for a in molecule])
    a = size_adjustment(radii)
    diff = pts[:, None, :] - C[None, :, :]  # r - R_X
    d = np.linalg.norm(diff, axis=2)
    u = diff / np.maximum(d, 1e-300)[:, :, None]
    P = np.ones((n, n_at))
    terms = []  # (C, D, ratio * dnu/dmu, mu)
    for c in range(n_at):
        for e in range(n_at):
            if c == e:
                continue
            R = np.linalg.norm(C[c] - C[e])
            mu = (d[:, c] - d[:, e]) / R
            nu = mu + a[c, e] * (1.0 - mu * mu)
            s, ds = _smooth_step_and_derivative(nu, stiffness)
            P[:, c] *= s
            ratio = np.where(s > 1e-300, ds / np.maximum(s, 1e-300), 0.0)
            terms.append((c, e, R, ratio * (1.0 - 2.0 * a[c, e] * mu), mu))
    Z = P.sum(axis=1)
    Z = np.where(Z > 0, Z, 1.0)
    abar = (P @ alphas) / Z
    g = np.zeros((n, n_at, 3))
    for c, e, R, r_dnu, mu in terms:
        coef = ((alphas[c] - abar) / Z * P[:, c] * r_dnu / R)[:, None]  # (n, 1)
        e_ce = (C[c] - C[e]) / R
        # dmu/dR_C = (-u_C - mu e_CE) / R ; dmu/dR_E = (u_E + mu e_CE) / R  (R folded in coef)
        g[:, c, :] += coef * (-u[:, c, :] - mu[:, None] * e_ce)
        g[:, e, :] += coef * (u[:, e, :] + mu[:, None] * e_ce)
    return g


def alpha_mixture_gradient(points, owner, coeff, molecule, alphas, block: int = 20_000):
    """``sum_k coeff_k d alpha(r_k) / d R_B`` for points moving with their owner.

    A point owned by atom O moves with R_O, so its total derivative is
    ``g[k, B] + delta_{B O} grad_r alpha(r_k)`` and, by translational invariance,
    ``grad_r alpha = -sum_X g[k, X]``. Returns (n_atoms, 3).
    """
    n_at = len(molecule)
    pts = np.asarray(points, dtype=float)
    owner = np.asarray(owner)
    from ..kernels import alpha_mixture_gradient as compiled

    radii = np.array([a.grid_radius() for a in molecule])
    fast = compiled(pts, owner, np.asarray(coeff, dtype=float),
                    np.asarray(molecule.positions, dtype=float), size_adjustment(radii),
                    np.asarray(alphas, dtype=float), DEFAULT_STIFFNESS)
    if fast is not None:
        return fast
    out = np.zeros((n_at, 3))
    for lo in range(0, len(pts), block):
        sl = slice(lo, lo + block)
        g = alpha_point_gradient(pts[sl], molecule, alphas)  # (m, n_at, 3)
        cg = coeff[sl][:, None, None] * g
        out += cg.sum(axis=0)
        np.add.at(out, owner[sl], -cg.sum(axis=1))
    return out


__all__ = [
    "atom_alphas",
    "alpha_on_points",
    "alpha_point_gradient",
    "alpha_mixture_gradient",
    "becke_weights",
]
