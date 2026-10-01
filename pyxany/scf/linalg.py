"""Generalised eigenproblem with canonical orthogonalisation."""

from __future__ import annotations

import numpy as np

DEFAULT_S_THRESHOLD = 5e-4


def canonical_orthogonaliser(S: np.ndarray, threshold: float = DEFAULT_S_THRESHOLD):
    """X with ``X^T S X = I``, dropping near-linearly-dependent combinations.

    Returns ``(X, n_dropped)``.
    """
    s_vals, s_vecs = np.linalg.eigh(S)
    keep = s_vals > threshold
    X = s_vecs[:, keep] * (s_vals[keep] ** -0.5)[np.newaxis, :]
    return X, int((~keep).sum())


def solve_generalised(F: np.ndarray, S: np.ndarray,
                      threshold: float = DEFAULT_S_THRESHOLD):
    """Solve ``F C = S C eps``.

    Eigenvalues of discarded directions are returned as ``+inf`` and their
    coefficient columns as zero, so that occupying orbitals in energy order can
    never pick up a linearly dependent combination.
    """
    n = F.shape[0]
    X, dropped = canonical_orthogonaliser(S, threshold)
    n_keep = X.shape[1]
    eps_k, C_k = np.linalg.eigh(X.T @ F @ X)

    eps = np.full(n, np.inf)
    eps[:n_keep] = eps_k
    C = np.zeros((n, n))
    C[:, :n_keep] = X @ C_k
    return eps, C, dropped


def energy_weighted_density(C: np.ndarray, eps: np.ndarray,
                            occ: np.ndarray) -> np.ndarray:
    """W = sum_i occ_i eps_i C_i C_i^T, the Lagrange multiplier matrix.

    Used by the Pulay force term ``- sum_{mu nu} W_{mu nu} dS_{mu nu}/dR``.
    """
    active = (occ > 0) & np.isfinite(eps)
    return (C[:, active] * (occ[active] * eps[active])) @ C[:, active].T
