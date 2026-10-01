"""Pure-numpy reference implementations of the JIT-accelerated inner loops.

These are always available and define the *correct* answer; the numba versions
in :mod:`dvxa.kernels.numba_impl` are pinned to them by tests/test_backends.py.
"""

from __future__ import annotations

import numpy as np


def poisson_radial(r: np.ndarray, w: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Spherically symmetric Poisson solve on a radial grid.

        V(r) = (1/r) int_0^r 4 pi s^2 rho(s) ds  +  int_r^inf 4 pi s rho(s) ds

    Implemented with cumulative sums over the quadrature nodes, which requires
    the nodes to be sorted in increasing r.
    """
    integrand = 4.0 * np.pi * r * r * rho
    order = np.argsort(r)
    rs, ws, fs = r[order], w[order], integrand[order]

    inner = np.cumsum(ws * fs)                     # int_0^r
    outer = np.cumsum((ws * fs / np.maximum(rs, 1e-20))[::-1])[::-1] - \
        ws * fs / np.maximum(rs, 1e-20)            # int_r^inf, exclusive

    V = np.empty_like(rs)
    V[:] = inner / np.maximum(rs, 1e-20) + outer
    out = np.empty_like(V)
    out[order] = V
    return out


def numerov_outward(g, h, l, m, out):
    """Pure-python Numerov, outward.  Same contract as the numba kernel."""
    n = len(g)
    out[:] = 0.0
    f = 1.0 - h * h * g / 12.0
    out[0] = 1e-30
    out[1] = out[0] * np.exp((l + 0.5) * h)
    top = min(m + 1, n - 2)
    for i in range(1, top + 1):
        out[i + 1] = ((12.0 - 10.0 * f[i]) * out[i] - f[i - 1] * out[i - 1]) / f[i + 1]
        if abs(out[i + 1]) > 1e200:
            out[: i + 2] *= 1e-200
    return out


def numerov_inward(g, h, r, kappa, m, out):
    """Pure-python Numerov, inward.  Same contract as the numba kernel."""
    n = len(g)
    out[:] = 0.0
    f = 1.0 - h * h * g / 12.0
    out[n - 1] = 1e-30
    # v = u/sqrt(r) ~ exp(-kappa r)/sqrt(r), so v[n-1]/v[n-2] carries
    # sqrt(r[n-2]/r[n-1]) -- not its reciprocal.
    decay = np.exp(-kappa * (r[n - 1] - r[n - 2])) * np.sqrt(r[n - 2] / r[n - 1])
    out[n - 2] = out[n - 1] / max(decay, 1e-300)
    bottom = max(m - 1, 1)
    for i in range(n - 2, bottom - 1, -1):
        out[i - 1] = ((12.0 - 10.0 * f[i]) * out[i] - f[i + 1] * out[i + 1]) / f[i - 1]
        if abs(out[i - 1]) > 1e200:
            out[i - 1:] *= 1e-200
    return out
