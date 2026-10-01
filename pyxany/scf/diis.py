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
    """Extrapolate F to minimise the commutator error ``e = FPS - SPF``."""

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


class EDIIS:
    """Energy-DIIS: minimise a quadratic model of E over density mixings."""

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

    def extrapolate(self) -> np.ndarray | None:
        n = len(self._F)
        if n < 2:
            return None
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                B[i, j] = np.einsum("mn,nm->", self._F[i] - self._F[j],
                                    self._P[i] - self._P[j])
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = B
        A[n, :n] = A[:n, n] = 1.0
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            try:
                c = np.linalg.lstsq(A, rhs, rcond=None)[0][:n]
            except Exception:
                return None
        c = np.maximum(c, 0.0)
        total = c.sum()
        if not np.isfinite(total) or total < 1e-15:
            return None
        c /= total
        return sum(c[i] * self._F[i] for i in range(n))

    @property
    def n_vectors(self) -> int:
        return len(self._F)
