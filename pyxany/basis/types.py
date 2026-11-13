"""Basis-function types: Slater (STO), contracted Gaussian (GTO), numerical (NAO).

All three share one representation::

    chi(r) = R(r) * Y_lm(theta, phi)
           = g(r) * S_lm(x, y, z)          with g(r) = R(r) / r^l

and every type supplies ``g`` and ``dg/dr``.  The gradient then follows from a
single formula that works for any basis type::

    grad chi = g'(r) * rhat * S_lm  +  g(r) * grad S_lm

which is what the Pulay force terms need.  Normalization is always
``int |chi|^2 dV = 1``, i.e. ``int R(r)^2 r^2 dr = 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gamma as _gamma

_SMALL_R = 1e-12


class BasisFunction:
    """Common interface.  Subclasses implement :meth:`g` and :meth:`dg_dr`."""

    l: int
    m: int
    atom_idx: int
    label: str
    kind: str

    def g(self, r: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def dg_dr(self, r: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def d2g_dr2(self, r: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def laplacian_radial(self, r: np.ndarray) -> np.ndarray:
        """The bracket in ``laplacian chi = [g'' + 2(l+1) g'/r] * S_lm``.

        Uses the fact that a real solid harmonic is harmonic (``lap S = 0``) and
        homogeneous of degree l (``r . grad S = l S``), so no finite differences
        are needed for the kinetic-energy integrand.
        """
        r = np.maximum(np.asarray(r, dtype=float), _SMALL_R)
        return self.d2g_dr2(r) + 2.0 * (self.l + 1) * self.dg_dr(r) / r

    def radial(self, r: np.ndarray) -> np.ndarray:
        """R(r) = g(r) * r^l, the conventionally normalized radial function."""
        return self.g(r) * np.asarray(r) ** self.l

    def exponent_scale(self) -> float:
        """A representative decay constant, used for distance screening."""
        raise NotImplementedError  # pragma: no cover


@dataclass
class STO(BasisFunction):
    """Slater-type orbital ``chi = N r^(n-1) exp(-zeta r) Y_lm``.

    ``N = (2 zeta)^n sqrt(2 zeta / (2n)!)`` makes ``int |chi|^2 dV = 1``.
    """

    n: int
    l: int
    m: int
    zeta: float
    atom_idx: int
    label: str = ""
    kind: str = field(default="sto", init=False)

    def __post_init__(self) -> None:
        if self.n - 1 - self.l < 0:
            raise ValueError(f"invalid STO: n={self.n} must be > l={self.l}")
        self.norm = (2.0 * self.zeta) ** self.n * math.sqrt(
            2.0 * self.zeta / math.factorial(2 * self.n)
        )
        self._p = self.n - 1 - self.l

    def g(self, r):
        r = np.asarray(r, dtype=float)
        base = self.norm * np.exp(-self.zeta * r)
        return base if self._p == 0 else base * r**self._p

    def dg_dr(self, r):
        r = np.asarray(r, dtype=float)
        e = self.norm * np.exp(-self.zeta * r)
        if self._p == 0:
            return -self.zeta * e
        return e * (self._p * r ** (self._p - 1) - self.zeta * r**self._p)

    def d2g_dr2(self, r):
        r = np.asarray(r, dtype=float)
        e = self.norm * np.exp(-self.zeta * r)
        p, z = self._p, self.zeta
        out = z * z * (r**p if p else np.ones_like(r))
        if p >= 1:
            out = out - 2.0 * z * p * r ** (p - 1)
        if p >= 2:
            out = out + p * (p - 1) * r ** (p - 2)
        return e * out

    def exponent_scale(self) -> float:
        return float(self.zeta)


@dataclass
class GTO(BasisFunction):
    """Contracted spherical Gaussian.

    ``chi = norm * sum_k d_k N_k r^l exp(-alpha_k r^2) * Y_lm``

    The tabulated ``d_k`` are contraction coefficients for **normalized**
    primitives -- the standard EMSL / Gaussian / Basis Set Exchange convention.
    Each primitive therefore carries

        N_k = [ Gamma(l + 3/2) / (2 (2 alpha_k)^(l + 3/2)) ]^(-1/2)

    and ``norm`` is a residual renormalization that is 1.0 to ~7 digits for
    every shell in the shipped tables (a useful self-check: a table read with
    the wrong convention gives ``norm`` far from 1).

    The products ``d_k * N_k`` are folded into ``coeffs`` once at construction
    so the hot evaluation loop does no extra work.
    """

    l: int
    m: int
    prims: tuple  # ((alpha, d_table), ...) as tabulated
    atom_idx: int
    label: str = ""
    norm: float = 1.0
    coeffs: tuple = ()  # ((alpha, d_table * N_prim * norm), ...) ready to evaluate
    kind: str = field(default="gto", init=False)

    @staticmethod
    def primitive_norm(l: int, alpha: float) -> float:
        """N such that ``int (N r^l e^(-alpha r^2))^2 r^2 dr = 1``."""
        return 1.0 / math.sqrt(_gamma(l + 1.5) / (2.0 * (2.0 * alpha) ** (l + 1.5)))

    @classmethod
    def contracted(cls, l: int, m: int, prims, atom_idx: int, label: str = "") -> GTO:
        """Build from tabulated (alpha, coefficient) pairs on normalized primitives."""
        prims = tuple((float(a), float(d)) for a, d in prims)
        scaled = [(a, d * cls.primitive_norm(l, a)) for a, d in prims]
        # int_0^inf r^(2l+2) exp(-s r^2) dr = Gamma(l + 3/2) / (2 s^(l + 3/2))
        s_int = 0.0
        for a_j, c_j in scaled:
            for a_k, c_k in scaled:
                s_int += c_j * c_k * _gamma(l + 1.5) / (2.0 * (a_j + a_k) ** (l + 1.5))
        norm = 1.0 / math.sqrt(s_int) if s_int > 1e-30 else 1.0
        coeffs = tuple((a, c * norm) for a, c in scaled)
        return cls(l=l, m=m, prims=prims, atom_idx=atom_idx, label=label,
                   norm=norm, coeffs=coeffs)

    def g(self, r):
        r = np.asarray(r, dtype=float)
        r2 = r * r
        out = np.zeros_like(r)
        for a, c in self.coeffs:
            out += c * np.exp(-a * r2)
        return out

    def dg_dr(self, r):
        r = np.asarray(r, dtype=float)
        r2 = r * r
        out = np.zeros_like(r)
        for a, c in self.coeffs:
            out += c * (-2.0 * a * r) * np.exp(-a * r2)
        return out

    def d2g_dr2(self, r):
        r = np.asarray(r, dtype=float)
        r2 = r * r
        out = np.zeros_like(r)
        for a, c in self.coeffs:
            out += c * (4.0 * a * a * r2 - 2.0 * a) * np.exp(-a * r2)
        return out

    def exponent_scale(self) -> float:
        # sqrt(alpha) has units of inverse length, comparable to an STO zeta
        return float(math.sqrt(min(a for a, _ in self.prims)))


@dataclass
class NAO(BasisFunction):
    """Numerical atomic orbital: R(r) tabulated on a radial grid and splined."""

    l: int
    m: int
    r_grid: np.ndarray
    radial_grid: np.ndarray
    spline_tck: object
    atom_idx: int
    label: str = ""
    kind: str = field(default="nao", init=False)

    def _R(self, r, der: int = 0):
        from scipy.interpolate import splev

        return splev(np.maximum(np.asarray(r, dtype=float), _SMALL_R),
                     self.spline_tck, der=der, ext=3)

    def g(self, r):
        r = np.maximum(np.asarray(r, dtype=float), _SMALL_R)
        return self._R(r) / r**self.l if self.l else self._R(r)

    def dg_dr(self, r):
        r = np.maximum(np.asarray(r, dtype=float), _SMALL_R)
        R = self._R(r)
        dR = self._R(r, der=1)
        if self.l == 0:
            return dR
        return (dR - self.l * R / r) / r**self.l

    def d2g_dr2(self, r):
        r = np.maximum(np.asarray(r, dtype=float), _SMALL_R)
        R, dR, d2R = self._R(r), self._R(r, der=1), self._R(r, der=2)
        if self.l == 0:
            return d2R
        l = self.l
        return (d2R - 2.0 * l * dR / r + l * (l + 1) * R / (r * r)) / r**l

    def radial(self, r):
        return self._R(r)

    def exponent_scale(self) -> float:
        """Effective decay rate from where the tail falls to 1/e of its peak."""
        R = np.abs(self.radial_grid)
        peak = int(np.argmax(R))
        tail = R[peak:]
        if len(tail) < 2 or R[peak] <= 0:
            return 1.0
        idx = int(np.argmax(tail < R[peak] / math.e))
        if idx == 0:
            return 1.0
        return float(1.0 / max(self.r_grid[peak + idx] - self.r_grid[peak], 1e-3))


class BasisSet(list):
    """A molecule's basis functions, with a few convenience views."""

    def __init__(self, functions=(), n_atoms: int = 0):
        super().__init__(functions)
        self.n_atoms = n_atoms or (max((f.atom_idx for f in self), default=-1) + 1)

    @property
    def n_basis(self) -> int:
        return len(self)

    @property
    def kinds(self) -> set:
        return {f.kind for f in self}

    def by_atom(self) -> list[list[int]]:
        """``[[mu, ...], ...]`` -- basis-function indices grouped by atom."""
        groups: list[list[int]] = [[] for _ in range(self.n_atoms)]
        for mu, f in enumerate(self):
            groups[f.atom_idx].append(mu)
        return groups

    def angular_arrays(self):
        """``(l, m, atom_idx)`` as int arrays, for vectorized code."""
        return (
            np.array([f.l for f in self], dtype=np.int64),
            np.array([f.m for f in self], dtype=np.int64),
            np.array([f.atom_idx for f in self], dtype=np.int64),
        )

    def max_l(self) -> int:
        return max((f.l for f in self), default=0)
