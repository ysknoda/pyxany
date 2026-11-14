"""Convergence accelerators: energy-DIIS then Pulay DIIS.

EDIIS (Kudin, Scuseria & Cances, *J. Chem. Phys.* **116**, 8255 (2002)) is robust
far from convergence but stalls near it; Pulay DIIS (*Chem. Phys. Lett.* **73**,
393 (1980)) is the opposite.  The driver runs EDIIS for the first few
iterations and Pulay DIIS afterwards.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve

#: Drop the oldest vectors once the DIIS matrix is worse conditioned than this.
MAX_CONDITION = 1e12


class PulayDIIS:
    """Extrapolate F to minimize the commutator error ``e = FPS - SPF``."""

    def __init__(self, max_vectors: int = 10):
        self.max_vectors = max_vectors
        self._F: list[np.ndarray] = []
        self._e: list[np.ndarray] = []

    def push(self, F: np.ndarray, P: np.ndarray, S: np.ndarray) -> None:
        e = (F @ P @ S - S @ P @ F).ravel()
        self._F.append(np.array(F, copy=True))
        self._e.append(e)
        if len(self._F) > self.max_vectors:
            self._F.pop(0)
            self._e.pop(0)

    def extrapolate(self) -> np.ndarray | None:
        """Solve the Pulay equations, discarding the oldest vectors if needed.

        As the SCF converges the error vectors become nearly parallel and the
        B matrix turns singular.  Rather than letting the solver return noise,
        the oldest vectors are dropped until the system is well conditioned.
        """
        while len(self._F) >= 2:
            n = len(self._F)
            E = np.array(self._e)
            B = np.zeros((n + 1, n + 1))
            B[:n, :n] = E @ E.T
            scale = max(np.abs(B[:n, :n]).max(), 1e-300)
            B[:n, :n] /= scale
            B[n, :n] = B[:n, n] = -1.0
            rhs = np.zeros(n + 1)
            rhs[n] = -1.0
            if np.linalg.cond(B) > MAX_CONDITION:
                self._F.pop(0)
                self._e.pop(0)
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                try:
                    c = solve(B, rhs, assume_a="sym")
                except Exception:
                    return None
            if not np.all(np.isfinite(c)):
                return None
            return sum(c[i] * self._F[i] for i in range(n))
        return None

    @property
    def error(self) -> float:
        return float(np.sqrt(np.mean(self._e[-1] ** 2))) if self._e else float("inf")

    @property
    def n_vectors(self) -> int:
        return len(self._F)


class SpinPulayDIIS(PulayDIIS):
    """Pulay DIIS for UKS: one coefficient set for both spin channels.

    F_alpha depends on P_beta through the Coulomb term, so the two channels are
    one fixed-point problem. Extrapolating them with independent coefficient
    sets, as two PulayDIIS objects did, converged UKS runs slowly or not at
    all: O2 triplet 134 iterations against 17 with the stacked error, the N2 1s
    transition state 321 against 20 (ADR 0012, F11). The error vector is the
    two commutators stacked; F is stored as the (2, n, n) pair.
    """

    def push(self, F_a, F_b, P_a, P_b, S) -> None:  # noqa: D102
        e_a = (F_a @ P_a @ S - S @ P_a @ F_a).ravel()
        e_b = (F_b @ P_b @ S - S @ P_b @ F_b).ravel()
        self._F.append(np.stack([F_a, F_b]))
        self._e.append(np.concatenate([e_a, e_b]))
        if len(self._F) > self.max_vectors:
            self._F.pop(0)
            self._e.pop(0)


class EDIIS:
    """Energy-DIIS of Kudin, Scuseria & Cances, J. Chem. Phys. 116, 8255 (2002).

    Minimizes the interpolated energy

        E(c) = sum_i c_i E_i - 1/4 sum_ij c_i c_j tr[(F_i - F_j)(P_i - P_j)]

    over c_i >= 0, sum c_i = 1, and returns sum_i c_i F_i. The paper's 1/2 is
    for its convention dE/dD = 2F (D the spin density of RHF); here F = dE/dP
    for the total density P, and 1/4 is what makes E(c) equal E(sum c_i P_i)
    exactly for a quadratic functional E(P) = tr(hP) + 1/2 tr(P G P): checked
    against brute force on such a model in tests/test_scf.py. F and
    P may be stacked (2, n, n) arrays for UKS, the trace then running over both
    channels with one coefficient set (as for SpinPulayDIIS).

    The previous version solved the Lagrange system of c^T B c alone and never
    read the energies (any two vectors gave the midpoint, even for E = -10 and
    +10 Ha), then clipped negative coefficients, which is not the constrained
    minimum (ADR 0012 P0).
    """

    def __init__(self, max_vectors: int = 8):
        self.max_vectors = max_vectors
        self._F: list[np.ndarray] = []
        self._P: list[np.ndarray] = []
        self._E: list[float] = []

    def push(self, F: np.ndarray, P: np.ndarray, energy: float) -> None:
        self._F.append(np.array(F, copy=True))
        self._P.append(np.array(P, copy=True))
        self._E.append(float(energy))
        if len(self._F) > self.max_vectors:
            self._F.pop(0)
            self._P.pop(0)
            self._E.pop(0)

    def coefficients(self) -> np.ndarray | None:
        """The constrained minimizer c of the interpolated energy."""
        from scipy.optimize import minimize

        n = len(self._F)
        if n < 2:
            return None
        E = np.array(self._E)
        # tr[(F_i - F_j)(P_i - P_j)] = d_ii + d_jj - d_ij - d_ji, d_ij = tr(F_i P_j)
        Fs = np.array(self._F).reshape(n, -1)
        Ps = np.array([np.swapaxes(P, -1, -2) for P in self._P]).reshape(n, -1)
        d = Fs @ Ps.T
        diag = np.diag(d)
        B = diag[:, None] + diag[None, :] - d - d.T
        shift = E.min()                      # conditioning only; c sums to 1

        def f(c):
            return float(c @ (E - shift) - 0.25 * c @ B @ c)

        def grad(c):
            return (E - shift) - 0.5 * (B @ c)

        best = None
        # the objective need not be convex: start from every vertex and the centre
        starts = [np.eye(n)[i] for i in range(n)] + [np.full(n, 1.0 / n)]
        for c0 in starts:
            res = minimize(f, c0, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                           constraints=[{"type": "eq", "fun": lambda c: c.sum() - 1.0,
                                         "jac": lambda c: np.ones_like(c)}],
                           options={"ftol": 1e-12, "maxiter": 200})
            if np.all(np.isfinite(res.x)) and (best is None or f(res.x) < f(best)):
                best = res.x
        if best is None:
            return None
        c = np.clip(best, 0.0, None)
        total = c.sum()
        return c / total if total > 1e-15 else None

    def extrapolate(self) -> np.ndarray | None:
        c = self.coefficients()
        if c is None:
            return None
        return sum(c[i] * self._F[i] for i in range(len(c)))

    @property
    def n_vectors(self) -> int:
        return len(self._F)
