"""Numba-accelerated inner loops.

Only imported when numba is installed and not disabled through
``DVXA_DISABLE_NUMBA``.  Every kernel here must reproduce its numpy counterpart
to machine precision -- tests/test_backends.py runs the whole SCF both ways and
compares total energies.

Compilation is lazy (``cache=True``, no import-time warm-up) so that importing
``dvxa`` stays fast for the CLI subcommands that never touch a grid.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from ..basis.harmonics import flat_coefficient_table, index_of

_OFFSETS, _COUNTS, _COEFFS, _POWERS = flat_coefficient_table()


@nb.njit(cache=True, fastmath=True)
def _solid_harmonic_terms(idx, dx, dy, dz, offsets, counts, coeffs, powers):
    """Evaluate one real solid harmonic from the packed monomial table."""
    acc = 0.0
    o = offsets[idx]
    for t in range(counts[idx]):
        term = coeffs[o + t]
        px = powers[o + t, 0]
        py = powers[o + t, 1]
        pz = powers[o + t, 2]
        for _ in range(px):
            term *= dx
        for _ in range(py):
            term *= dy
        for _ in range(pz):
            term *= dz
        acc += term
    return acc


@nb.njit(cache=True, parallel=True, fastmath=True)
def _chi_block(norms, zetas, powers_n, lm_index, d, r, out, cols,
               offsets, counts, coeffs, powers):
    """Fill ``out[:, cols]`` with STO values for one atom's basis functions."""
    n_pts = d.shape[0]
    n_f = norms.shape[0]
    for k in nb.prange(n_pts):
        dx = d[k, 0]
        dy = d[k, 1]
        dz = d[k, 2]
        rk = r[k]
        for j in range(n_f):
            rad = norms[j] * np.exp(-zetas[j] * rk)
            p = powers_n[j]
            for _ in range(p):
                rad *= rk
            ang = _solid_harmonic_terms(lm_index[j], dx, dy, dz,
                                        offsets, counts, coeffs, powers)
            out[k, cols[j]] = rad * ang


def chi_block(norms, zetas, powers_n, l_arr, m_arr, d, r, out, cols):
    """Wrapper translating (l, m) pairs into flat-table indices."""
    lm_index = np.array([index_of(int(l), int(m)) for l, m in zip(l_arr, m_arr)], np.int64)
    _chi_block(np.ascontiguousarray(norms, np.float64),
               np.ascontiguousarray(zetas, np.float64),
               np.ascontiguousarray(powers_n, np.int64),
               lm_index,
               np.ascontiguousarray(d, np.float64),
               np.ascontiguousarray(r, np.float64),
               out, np.ascontiguousarray(cols, np.int64),
               _OFFSETS, _COUNTS, _COEFFS, _POWERS)


@nb.njit(cache=True, fastmath=True)
def _poisson_radial_sorted(rs, ws, fs):
    n = rs.shape[0]
    V = np.empty(n)
    inner = 0.0
    for i in range(n):
        inner += ws[i] * fs[i]
        V[i] = inner / (rs[i] if rs[i] > 1e-20 else 1e-20)
    outer = 0.0
    for i in range(n - 1, -1, -1):
        V[i] += outer
        outer += ws[i] * fs[i] / (rs[i] if rs[i] > 1e-20 else 1e-20)
    return V


def poisson_radial(r, w, rho):
    """O(n) two-sweep spherical Poisson solve (same contract as the numpy one)."""
    r = np.ascontiguousarray(r, np.float64)
    w = np.ascontiguousarray(w, np.float64)
    integrand = 4.0 * np.pi * r * r * np.ascontiguousarray(rho, np.float64)
    order = np.argsort(r)
    V = _poisson_radial_sorted(r[order], w[order], integrand[order])
    out = np.empty_like(V)
    out[order] = V
    return out


@nb.njit(cache=True, parallel=True, fastmath=True)
def density_from_chi(chi, P, out):
    """rho_k = sum_{mn} P_{mn} chi_{km} chi_{kn}, without an (n_pts, nb) temporary."""
    n_pts = chi.shape[0]
    nb_ = chi.shape[1]
    for k in nb.prange(n_pts):
        acc = 0.0
        for m in range(nb_):
            cm = chi[k, m]
            if cm == 0.0:
                continue
            row = 0.0
            for n in range(nb_):
                row += P[m, n] * chi[k, n]
            acc += cm * row
        out[k] = acc


@nb.njit(cache=True)
def numerov_outward(g, h, l, m, out):
    """Integrate v'' = g v outward from the inner boundary through index m."""
    n = g.shape[0]
    for i in range(n):
        out[i] = 0.0
    inv12 = h * h / 12.0
    out[0] = 1e-30
    out[1] = out[0] * np.exp((l + 0.5) * h)
    top = m + 1 if m + 1 < n - 1 else n - 2
    for i in range(1, top + 1):
        fi = 1.0 - inv12 * g[i]
        fp = 1.0 - inv12 * g[i + 1]
        fm = 1.0 - inv12 * g[i - 1]
        out[i + 1] = ((12.0 - 10.0 * fi) * out[i] - fm * out[i - 1]) / fp
        if out[i + 1] > 1e200 or out[i + 1] < -1e200:
            for k in range(i + 2):
                out[k] *= 1e-200
    return out


@nb.njit(cache=True)
def numerov_inward(g, h, r, kappa, m, out):
    """Integrate v'' = g v inward from the outer boundary down through index m."""
    n = g.shape[0]
    for i in range(n):
        out[i] = 0.0
    inv12 = h * h / 12.0
    out[n - 1] = 1e-30
    decay = np.exp(-kappa * (r[n - 1] - r[n - 2])) * np.sqrt(r[n - 1] / r[n - 2])
    out[n - 2] = out[n - 1] / (decay if decay > 1e-300 else 1e-300)
    bottom = m - 1 if m - 1 > 1 else 1
    for i in range(n - 2, bottom - 1, -1):
        fi = 1.0 - inv12 * g[i]
        fp = 1.0 - inv12 * g[i + 1]
        fm = 1.0 - inv12 * g[i - 1]
        out[i - 1] = ((12.0 - 10.0 * fi) * out[i] - fp * out[i + 1]) / fm
        if out[i - 1] > 1e200 or out[i - 1] < -1e200:
            for k in range(i - 1, n):
                out[k] *= 1e-200
    return out
