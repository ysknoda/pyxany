"""Spherical Kohn-Sham X-alpha solver for a single (possibly fractional) atom.

Solves the radial equation

    [ -1/2 d^2/dr^2 + l(l+1)/(2 r^2) + V_eff(r) ] u_nl = eps_nl u_nl

with ``u = r R`` and ``V_eff = -Z/r + V_H[rho] + V_xc[rho]``, iterated to
self-consistency.

Grid
----
A **logarithmic** grid ``r = r_min exp(x)`` is used, as in every standard atomic
code: the nuclear cusp of a 1s orbital is unresolvable on a uniform mesh at any
affordable point count (a uniform grid puts the hydrogen 1s eigenvalue ~4 % off
and can even invert the 2s/2p ordering of neon).  Substituting ``v = u / sqrt(r)``
turns the equation into

    -1/2 d^2 v/dx^2 + [ (l + 1/2)^2 / 2 + r^2 V_eff ] v = eps r^2 v

Bound states are found by **Numerov integration with node counting**, the
standard method for this equation.  (Reducing the generalised tridiagonal
eigenproblem ``A v = eps diag(r^2) v`` to standard form instead is exact on
paper but useless in practice: on a mesh reaching r ~ 1e-6 the reduction
multiplies the matrix by ``1/r^2``, giving a condition number around 1e16 and
eigenvalues that collapse into a degenerate cluster.)  Two uses:

* generating numerical atomic orbitals (the ``nao`` basis);
* deriving the 1s function of a **pseudo-hydrogen** with fractional nuclear
  charge ``alpha`` and ``alpha`` electrons -- see :mod:`pyxany.basis.pseudo_h`.

Nothing here is fitted: the pseudo-hydrogen orbital is simply the solution of
the atomic problem for that charge, so the terminator introduces no adjustable
parameter into a calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import splrep

from ..constants import PI
from ..data import elements as elem
from ..integrals import xc
from ..kernels import numerov_inward, numerov_outward, poisson_radial


@dataclass
class AtomicOrbital:
    n: int
    l: int
    occupation: float
    eigenvalue: float
    r: np.ndarray
    radial: np.ndarray          # R(r), normalised so int R^2 r^2 dr = 1
    spline: object


@dataclass
class AtomicSolution:
    symbol: str
    Z: float
    orbitals: dict               # {(n, l): AtomicOrbital}
    density: np.ndarray
    r: np.ndarray
    converged: bool
    n_iterations: int

    def total_electrons(self) -> float:
        return sum(o.occupation for o in self.orbitals.values())


def _shell_occupations(symbol: str) -> dict[tuple[int, int], float]:
    """{(n, l): electrons} for a neutral atom, from the tabulated configuration."""
    totals: dict[tuple[int, int], float] = {}
    for label, ne in elem.occupations(symbol).items():
        l = elem._subshell_l(label)
        n = next((int(c) for c in label if c.isdigit()), 1)
        totals[(n, l)] = totals.get((n, l), 0.0) + float(ne)
    return {k: v for k, v in totals.items() if v > 0}


def solve_atom(symbol: str | None = None, Z: float | None = None,
               occupations: dict | None = None, n_points: int = 1500,
               r_max: float | None = None, alpha: float = 0.7,
               max_iter: int = 300, tol: float = 1e-9,
               mixing: float = 0.2) -> AtomicSolution:
    """Self-consistent spherical X-alpha solution for one atom.

    Parameters
    ----------
    symbol       element symbol (sets Z and the shell occupations by default)
    Z            nuclear charge; may be fractional (pseudo-hydrogen)
    occupations  ``{(n, l): electrons}``; defaults to the neutral configuration
    """
    if symbol is not None:
        Z = float(elem.atomic_number(symbol)) if Z is None else float(Z)
        occupations = occupations or _shell_occupations(symbol)
        r_max = r_max or max(30.0, 6.0 * elem.bragg_slater_radius(symbol))
    else:
        if Z is None or occupations is None:
            raise ValueError("provide either a symbol or both Z and occupations")
        Z = float(Z)
        r_max = r_max or max(40.0, 30.0 / max(Z, 0.2))

    # logarithmic grid: dense at the nucleus, sparse in the tail
    r_min = 1e-6 / max(Z, 0.2)
    h = np.log(r_max / r_min) / (n_points - 1)
    r = r_min * np.exp(h * np.arange(n_points))
    w_r = h * r                                   # weights for int f(r) dr

    rho = _initial_density(r, Z, occupations, w_r)

    orbitals: dict = {}
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        V_h = poisson_radial(r, w_r, rho)
        V_xc = xc.potential(rho, alpha)
        V_eff = -Z / r + V_h + V_xc

        new_rho = np.zeros_like(r)
        orbitals = {}
        for (n, l), ne in sorted(occupations.items()):
            if ne <= 0:
                continue
            eps, R = _solve_channel(r, h, V_eff, l, n_nodes=n - l - 1)
            if R is None:
                continue
            norm = np.sqrt(np.dot(w_r, R * R * r * r))
            if norm < 1e-12:
                continue
            R = R / norm
            new_rho += ne * R * R / (4.0 * PI)
            orbitals[(n, l)] = AtomicOrbital(n=n, l=l, occupation=ne, eigenvalue=eps,
                                             r=r, radial=R, spline=None)

        # Scale-free convergence measure: the fraction of an electron that moved.
        # An absolute tolerance on rho is unusable here -- the density spans ten
        # orders of magnitude between the nucleus and the tail.
        n_e = max(sum(occupations.values()), 1e-12)
        delta = float(np.dot(w_r, np.abs(new_rho - rho) * 4.0 * PI * r * r)) / n_e
        rho = mixing * new_rho + (1.0 - mixing) * rho
        if delta < tol and it > 3:
            converged = True
            break

    for orb in orbitals.values():
        try:
            orb.spline = splrep(orb.r, orb.radial, s=0, k=3)
        except Exception:  # pragma: no cover - degenerate grid
            orb.spline = None

    return AtomicSolution(symbol=symbol or f"Z={Z:g}", Z=Z, orbitals=orbitals,
                          density=rho, r=r, converged=converged, n_iterations=it)


def _initial_density(r, Z, occupations, w_r) -> np.ndarray:
    """Hydrogenic starting density, screened by a crude effective charge."""
    rho = np.zeros_like(r)
    n_e = sum(occupations.values())
    for (n, _l), ne in occupations.items():
        z_eff = max((Z - 0.5 * (n_e - ne)) / n, 0.3)
        R = 2.0 * z_eff**1.5 * np.exp(-z_eff * r)
        norm = float(np.dot(w_r, R * R * r * r))
        rho += ne * R * R / (4.0 * PI * max(norm, 1e-30))
    return rho


@lru_cache(maxsize=4)
def _scratch(n: int):
    """Reusable work arrays for the Numerov integrations."""
    return np.empty(n), np.empty(n)


def _count_nodes(v: np.ndarray) -> int:
    scale = max(np.abs(v[1:-1]).max(), 1e-300)
    sig = v[1:-1][np.abs(v[1:-1]) > 1e-10 * scale]
    if len(sig) < 2:
        return 0
    return int(np.sum(np.sign(sig[1:]) != np.sign(sig[:-1])))


def _matching_index(g: np.ndarray) -> int:
    """Outermost classical turning point, where the solution starts decaying."""
    allowed = np.where(g < 0)[0]
    if len(allowed) == 0:
        return len(g) // 2
    return int(min(max(allowed[-1], 3), len(g) - 4))


def _trial(r, h, V_eff, l, eps):
    """Match outward and inward Numerov solutions at the turning point.

    Returns ``(n_nodes, log-derivative mismatch, wavefunction)``.  The mismatch
    changes sign exactly at an eigenvalue, and the node count of the matched
    function identifies which state that is.
    """
    g = (l + 0.5) ** 2 + 2.0 * r * r * (V_eff - eps)
    m = _matching_index(g)
    scratch = _scratch(len(r))
    v_out = numerov_outward(g, h, l, m, scratch[0])
    v_in = numerov_inward(g, h, r, np.sqrt(max(-2.0 * eps, 1e-12)), m, scratch[1])
    if abs(v_out[m]) < 1e-300 or abs(v_in[m]) < 1e-300:
        return -1, 0.0, None
    v = np.empty(len(r))
    v[: m + 1] = v_out[: m + 1]
    v[m + 1:] = v_in[m + 1:] * (v_out[m] / v_in[m])
    d_out = (v_out[m + 1] - v_out[m - 1]) / (2.0 * h * v_out[m])
    d_in = (v_in[m + 1] - v_in[m - 1]) / (2.0 * h * v_in[m])
    return _count_nodes(v), d_out - d_in, v


def _solve_channel(r, h: float, V_eff, l: int, n_nodes: int,
                   max_bisections: int = 200):
    """Bound state with exactly ``n_nodes`` radial nodes, by Numerov shooting.

    A single bisection suffices once the two diagnostics are combined into one
    monotone predicate.  For a trial energy ``eps`` the matched solution has a
    node count and a log-derivative mismatch at the turning point; ``eps`` is
    above the wanted eigenvalue exactly when

        nodes > n_nodes            (already past the state), or
        nodes == n_nodes and mismatch < 0   (past the root inside this window)

    and below it otherwise.  Node count alone is constant across the whole gap
    between neighbouring states, and the mismatch alone vanishes at every state,
    so neither is sufficient on its own.
    """
    Z_eff = max(-float(V_eff[0]) * r[0], 1.0)
    lo = -2.0 * Z_eff * Z_eff - 10.0
    hi = -1e-12

    for _ in range(max_bisections):
        mid = 0.5 * (lo + hi)
        nodes, mismatch, _ = _trial(r, h, V_eff, l, mid)
        if nodes < 0:
            return 0.0, None
        too_high = nodes > n_nodes or (nodes == n_nodes and mismatch < 0.0)
        if too_high:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14 * max(abs(hi), 1.0):
            break

    eps = 0.5 * (lo + hi)
    nodes, _, v = _trial(r, h, V_eff, l, eps)
    if v is None or nodes != n_nodes or eps >= -1e-12:
        return 0.0, None
    # v = u / sqrt(r) and u = r R, so R = v / sqrt(r).
    R = v / np.sqrt(r)
    if not np.all(np.isfinite(R)):
        return 0.0, None
    # The Numerov integration starts from an arbitrary seed (1e-30) and is
    # rescaled on overflow, so the returned amplitude is meaningless.  Fix the
    # scale here; the caller normalises properly against the radial measure.
    peak = float(np.abs(R).max())
    if peak < 1e-300:
        return 0.0, None
    R = R / peak
    if R[np.argmax(np.abs(R))] < 0:
        R = -R
    return float(eps), R
