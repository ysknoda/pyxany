"""Evaluate basis functions and their gradients on a set of grid points.

Two entry points:

``chi_matrix(basis, positions, points)``
    ``(n_points, n_basis)`` values of every basis function.

``chi_grad_matrix(basis, positions, points)``
    ``(n_points, n_basis, 3)`` gradients with respect to the *electron*
    coordinate.  The derivative with respect to the position of the atom the
    function is centred on is the negative of this, which is what the Pulay
    force terms need.

Both group functions by their parent atom so that the radius and the Cartesian
offsets are computed once per atom rather than once per basis function.
"""

from __future__ import annotations

import numpy as np

from ..kernels import (
    chi_all_kernel,
    chi_block_kernel,
    chi_grad_block_kernel,
    chi_hess_block_kernel,
)
from .harmonics import solid_harmonic, solid_harmonic_grad

_SAFE_R = 1e-14


def _atom_groups(basis) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for mu, f in enumerate(basis):
        groups.setdefault(f.atom_idx, []).append(mu)
    return groups


def chi_matrix(basis, positions, points, out: np.ndarray | None = None) -> np.ndarray:
    """(n_points, n_basis) values of the basis on ``points`` (Bohr)."""
    pts = np.ascontiguousarray(points, dtype=float)
    n_pts, n_bas = len(pts), len(basis)
    X = np.empty((n_pts, n_bas), dtype=float) if out is None else out

    every = chi_all_kernel()
    if every is not None and n_bas and all(f.kind == "sto" for f in basis):
        pos = np.asarray(positions, dtype=float)
        every(pos[[f.atom_idx for f in basis]],
              np.array([f.norm for f in basis]), np.array([f.zeta for f in basis]),
              np.array([f.n - 1 - f.l for f in basis], np.int64),
              np.array([f.l for f in basis], np.int64),
              np.array([f.m for f in basis], np.int64), pts, X)
        return X

    fast = chi_block_kernel()
    for ai, idx in _atom_groups(basis).items():
        d = pts - np.asarray(positions[ai], dtype=float)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        if fast is not None and all(basis[mu].kind == "sto" for mu in idx):
            fast(
                np.array([basis[mu].norm for mu in idx]),
                np.array([basis[mu].zeta for mu in idx]),
                np.array([basis[mu].n - 1 - basis[mu].l for mu in idx], np.int64),
                np.array([basis[mu].l for mu in idx], np.int64),
                np.array([basis[mu].m for mu in idx], np.int64),
                d, r, X, np.array(idx, np.int64),
            )
            continue
        for mu in idx:
            f = basis[mu]
            X[:, mu] = f.g(r) * solid_harmonic(f.l, f.m, d[:, 0], d[:, 1], d[:, 2])
    return X


def chi_grad_matrix(basis, positions, points, out: np.ndarray | None = None) -> np.ndarray:
    """(n_points, n_basis, 3) gradient of each basis function w.r.t. the electron.

    ``grad chi = g'(r) rhat S_lm + g(r) grad S_lm``. ``out`` may be a strided
    view, e.g. ``np.empty((3, n, nb)).transpose(1, 2, 0)`` to make each
    Cartesian component contiguous.
    """
    pts = np.ascontiguousarray(points, dtype=float)
    G = np.empty((len(pts), len(basis), 3), dtype=float) if out is None else out

    fast = chi_grad_block_kernel()
    for ai, idx in _atom_groups(basis).items():
        d = pts - np.asarray(positions[ai], dtype=float)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        if fast is not None and all(basis[mu].kind == "sto" for mu in idx):
            fast(
                np.array([basis[mu].norm for mu in idx]),
                np.array([basis[mu].zeta for mu in idx]),
                np.array([basis[mu].n - 1 - basis[mu].l for mu in idx], np.int64),
                np.array([basis[mu].l for mu in idx], np.int64),
                np.array([basis[mu].m for mu in idx], np.int64),
                d, r, G, np.array(idx, np.int64),
            )
            continue
        safe = np.maximum(r, _SAFE_R)
        rhat = d / safe[:, None]
        for mu in idx:
            f = basis[mu]
            S = solid_harmonic(f.l, f.m, d[:, 0], d[:, 1], d[:, 2])
            gS = solid_harmonic_grad(f.l, f.m, d[:, 0], d[:, 1], d[:, 2])
            gr = f.g(r)
            dg = f.dg_dr(r)
            for ax in range(3):
                G[:, mu, ax] = dg * rhat[:, ax] * S + gr * gS[ax]
    return G


def chi_hess_matrix(basis, positions, points) -> np.ndarray | None:
    """(n_points, n_basis, 3, 3) Hessian of each basis function w.r.t. the electron.

    Compiled STO path only; returns None when numba is unavailable or the
    basis contains other function types, so callers keep a fallback.
    """
    fast = chi_hess_block_kernel()
    if fast is None or any(f.kind != "sto" for f in basis):
        return None
    pts = np.ascontiguousarray(points, dtype=float)
    H = np.empty((len(pts), len(basis), 3, 3), dtype=float)
    for ai, idx in _atom_groups(basis).items():
        d = pts - np.asarray(positions[ai], dtype=float)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        fast(
            np.array([basis[mu].norm for mu in idx]),
            np.array([basis[mu].zeta for mu in idx]),
            np.array([basis[mu].n - 1 - basis[mu].l for mu in idx], np.int64),
            np.array([basis[mu].l for mu in idx], np.int64),
            np.array([basis[mu].m for mu in idx], np.int64),
            d, r, H, np.array(idx, np.int64),
        )
    return H


def density_on_grid(chi: np.ndarray, P: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """rho(r_k) = sum_{mu nu} P_{mu nu} chi_{k mu} chi_{k nu}."""
    rho = np.einsum("km,mn,kn->k", chi, P, chi, optimize=True)
    return np.maximum(rho, floor) if floor else rho


def potential_matrix(chi: np.ndarray, weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    """<mu| v |nu> = sum_k w_k v(r_k) chi_{k mu} chi_{k nu}."""
    return (chi * (weights * v)[:, None]).T @ chi


def chi_laplacian_matrix(basis, positions, points) -> np.ndarray:
    """(n_points, n_basis) values of ``laplacian chi``.

    Uses ``lap chi = [g'' + 2(l+1) g'/r] S_lm``, exact for every basis type
    because a real solid harmonic is harmonic and homogeneous of degree l.
    """
    pts = np.ascontiguousarray(points, dtype=float)
    L = np.empty((len(pts), len(basis)), dtype=float)
    for ai, idx in _atom_groups(basis).items():
        d = pts - np.asarray(positions[ai], dtype=float)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        for mu in idx:
            f = basis[mu]
            L[:, mu] = f.laplacian_radial(r) * solid_harmonic(
                f.l, f.m, d[:, 0], d[:, 1], d[:, 2])
    return L
