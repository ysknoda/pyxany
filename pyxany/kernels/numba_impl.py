"""Numba-accelerated inner loops.

Only imported when numba is installed and not disabled through
``PYXANY_DISABLE_NUMBA``.  Every kernel here must reproduce its numpy counterpart
to machine precision -- tests/test_backends.py runs the whole SCF both ways and
compares total energies.

Compilation is lazy (``cache=True``, no import-time warm-up) so that importing
``pyxany`` stays fast for the CLI subcommands that never touch a grid.
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
def _ipow(x, p):
    v = 1.0
    for _ in range(p):
        v *= x
    return v


@nb.njit(cache=True, parallel=True, fastmath=True)
def _chi_grad_block(norms, zetas, powers_n, lm_index, d, r, out, cols,
                    offsets, counts, coeffs, powers):
    """Fill ``out[:, cols, :]`` with STO gradients w.r.t. the electron.

    ``grad chi = g'(r) rhat S_lm + g(r) grad S_lm`` with
    ``g = N r^p e^(-zeta r)``; ``grad S_lm`` differentiates the monomial table.
    """
    n_pts = d.shape[0]
    n_f = norms.shape[0]
    for k in nb.prange(n_pts):
        dx = d[k, 0]
        dy = d[k, 1]
        dz = d[k, 2]
        rk = r[k]
        rs = rk if rk > 1e-14 else 1e-14
        for j in range(n_f):
            e = norms[j] * np.exp(-zetas[j] * rk)
            p = powers_n[j]
            g = e * _ipow(rk, p)
            if p == 0:
                dg = -zetas[j] * e
            else:
                dg = e * (p * _ipow(rk, p - 1) - zetas[j] * _ipow(rk, p))
            idx = lm_index[j]
            o = offsets[idx]
            s = 0.0
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for t in range(counts[idx]):
                c = coeffs[o + t]
                px = powers[o + t, 0]
                py = powers[o + t, 1]
                pz = powers[o + t, 2]
                mx = _ipow(dx, px)
                my = _ipow(dy, py)
                mz = _ipow(dz, pz)
                s += c * mx * my * mz
                if px > 0:
                    sx += c * px * _ipow(dx, px - 1) * my * mz
                if py > 0:
                    sy += c * py * mx * _ipow(dy, py - 1) * mz
                if pz > 0:
                    sz += c * pz * mx * my * _ipow(dz, pz - 1)
            radial = dg * s / rs
            col = cols[j]
            out[k, col, 0] = radial * dx + g * sx
            out[k, col, 1] = radial * dy + g * sy
            out[k, col, 2] = radial * dz + g * sz


def chi_grad_block(norms, zetas, powers_n, l_arr, m_arr, d, r, out, cols):
    """Gradient counterpart of :func:`chi_block` (``out`` is (n_pts, nb, 3))."""
    lm_index = np.array([index_of(int(l), int(m)) for l, m in zip(l_arr, m_arr)], np.int64)
    _chi_grad_block(np.ascontiguousarray(norms, np.float64),
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
    # v = u/sqrt(r) ~ exp(-kappa r)/sqrt(r), so v[n-1]/v[n-2] carries
    # sqrt(r[n-2]/r[n-1]) -- not its reciprocal.
    decay = np.exp(-kappa * (r[n - 1] - r[n - 2])) * np.sqrt(r[n - 2] / r[n - 1])
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


# --------------------------------------------------------------------------
# Becke cell weights: the O(N_atoms^2) product per point, compiled
# --------------------------------------------------------------------------
@nb.njit(cache=True, fastmath=False)
def _smooth_step(nu, stiffness):
    f = min(max(nu, -1.0), 1.0)
    for _ in range(stiffness):
        f = 1.5 * f - 0.5 * f * f * f
    return 0.5 * (1.0 - f)


@nb.njit(parallel=True, cache=True)
def becke_own_cell(points, owner, centres, a, inv_r, stiffness, out):
    """Own-cell Becke weight ``P_owner / sum_C P_C`` of every point (see grid/becke.py)."""
    n_pts = points.shape[0]
    n_at = centres.shape[0]
    for k in nb.prange(n_pts):
        dist = np.empty(n_at)
        for i in range(n_at):
            dx = points[k, 0] - centres[i, 0]
            dy = points[k, 1] - centres[i, 1]
            dz = points[k, 2] - centres[i, 2]
            dist[i] = np.sqrt(dx * dx + dy * dy + dz * dz)
        total = 0.0
        own = 0.0
        for i in range(n_at):
            p = 1.0
            for j in range(n_at):
                if i == j:
                    continue
                mu = (dist[i] - dist[j]) * inv_r[i, j]
                p *= _smooth_step(mu + a[i, j] * (1.0 - mu * mu), stiffness)
                if p == 0.0:
                    break
            total += p
            if i == owner[k]:
                own = p
        if total > 1e-300:
            out[k] = own / total
        else:  # far from everything: nearest atom takes the point
            best = 0
            for i in range(1, n_at):
                if dist[i] < dist[best]:
                    best = i
            out[k] = 1.0 if best == owner[k] else 0.0


# --------------------------------------------------------------------------
# Hartree potential: sum of every atom's multipole expansion at each point
# --------------------------------------------------------------------------
@nb.njit(parallel=True, cache=True)
def hartree_evaluate(points, centres, knots, n_knots, coefs, v_first, v_last, l_max, norms,
                     out):
    """``V(r_k) = sum_A sum_lm v^A_lm(|r_k - R_A|) Y_lm`` (see integrals/coulomb.py).

    ``knots[A, :n_knots[A]]`` and ``coefs[A, :, :n_knots[A] - 1]`` are the
    cubic-spline breakpoints and coefficients (scipy's ``CubicSpline.c``
    layout) of atom A's radial multipole potentials, zero-padded to a common
    length; inside the first node V_l ~ r^l, beyond the last V_l ~ r^-(l+1). Solid harmonics are built by the same recursion as
    basis/solid_harmonics.py, one point at a time, so no (n_lm, n_points)
    array is ever formed.
    """
    n_pts = points.shape[0]
    n_at = centres.shape[0]
    n_lm = (l_max + 1) * (l_max + 1)
    for k in nb.prange(n_pts):
        S = np.empty(n_lm)
        v = np.empty(n_lm)
        acc = 0.0
        for A in range(n_at):
            x = points[k, 0] - centres[A, 0]
            y = points[k, 1] - centres[A, 1]
            z = points[k, 2] - centres[A, 2]
            r2 = x * x + y * y + z * z
            r = np.sqrt(r2)
            nk = n_knots[A]
            r0 = knots[A, 0]
            rl_ = knots[A, nk - 1]
            # -- radial multipole potentials at r --------------------------
            if r < r0:
                ratio = r / r0
                for l in range(l_max + 1):
                    p = ratio ** l
                    for c in range(l * l, (l + 1) * (l + 1)):
                        v[c] = v_first[A, c] * p
            elif r > rl_:
                for l in range(l_max + 1):
                    p = (rl_ / r) ** (l + 1)
                    for c in range(l * l, (l + 1) * (l + 1)):
                        v[c] = v_last[A, c] * p
            else:
                lo = 0
                hi = nk - 1
                while hi - lo > 1:             # knots[A, lo] <= r < knots[A, hi]
                    mid = (lo + hi) // 2
                    if knots[A, mid] <= r:
                        lo = mid
                    else:
                        hi = mid
                seg = lo
                t = r - knots[A, lo]
                for c in range(n_lm):
                    v[c] = ((coefs[A, 0, seg, c] * t + coefs[A, 1, seg, c]) * t
                            + coefs[A, 2, seg, c]) * t + coefs[A, 3, seg, c]
            # -- solid harmonics, then V += sum v_lm Y_lm --------------------
            S[0] = 1.0
            for l in range(1, l_max + 1):
                f = np.sqrt((2.0 * l - 1.0) / (2.0 * l))
                cc = (l - 1) * (l - 1) + 2 * (l - 1)     # column of (l-1, l-1)
                cs = (l - 1) * (l - 1)                   # column of (l-1, -(l-1))
                pc = S[cc]
                ps = S[cs] if l > 1 else 0.0
                S[l * l + 2 * l] = f * (x * pc - y * ps)
                S[l * l] = f * (y * pc + x * ps)
                for m in range(-(l - 1), l):
                    num = (2.0 * l - 1.0) * z * S[(l - 1) * (l - 1) + m + l - 1]
                    if l >= 2 and abs(m) <= l - 2:
                        num -= np.sqrt((l - 1.0) ** 2 - m * m) * r2 * S[(l - 2) * (l - 2) + m + l - 2]
                    S[l * l + m + l] = num / np.sqrt(l * l - m * m)
            acc += v[0] * norms[0]
            if r > 1e-300:
                inv = 1.0 / r
                rl = 1.0
                for l in range(1, l_max + 1):
                    rl *= inv
                    for c in range(l * l, (l + 1) * (l + 1)):
                        acc += v[c] * S[c] * rl * norms[c]
        out[k] = acc


@nb.njit(parallel=True, cache=True)
def hartree_evaluate_grad(points, centres, knots, n_knots, coefs, v_first, v_last, l_max,
                          norms, out):
    """Exact gradient of the potential :func:`hartree_evaluate` returns.

    Differentiates the same piecewise radial functions (cubic spline, r^l
    inside the first node, r^-(l+1) beyond the last) and carries the solid
    harmonic recursion forward with its x, y, z derivatives. Each term is
    ``q_l(r) S_lm(x)`` with ``q_l = v_l r^-l``, so
    ``grad = q_l'(r) rhat S_lm + q_l grad S_lm``.
    """
    n_pts = points.shape[0]
    n_at = centres.shape[0]
    n_lm = (l_max + 1) * (l_max + 1)
    for k in nb.prange(n_pts):
        S = np.empty(n_lm)
        Sx = np.empty(n_lm)
        Sy = np.empty(n_lm)
        Sz = np.empty(n_lm)
        q = np.empty(n_lm)
        dq = np.empty(n_lm)
        gx = 0.0
        gy = 0.0
        gz = 0.0
        for A in range(n_at):
            x = points[k, 0] - centres[A, 0]
            y = points[k, 1] - centres[A, 1]
            z = points[k, 2] - centres[A, 2]
            r2 = x * x + y * y + z * z
            r = np.sqrt(r2)
            nk = n_knots[A]
            r0 = knots[A, 0]
            rl_ = knots[A, nk - 1]
            # -- q_l = v_l r^-l and dq_l/dr ------------------------------------
            if r < r0:
                # v_l = v_first (r/r0)^l, so q_l is constant
                for l in range(l_max + 1):
                    p = 1.0 / r0 ** l
                    for c in range(l * l, (l + 1) * (l + 1)):
                        q[c] = v_first[A, c] * p
                        dq[c] = 0.0
            else:
                inv = 1.0 / r
                if r > rl_:
                    for l in range(l_max + 1):
                        p = (rl_ * inv) ** (l + 1) * inv ** l
                        for c in range(l * l, (l + 1) * (l + 1)):
                            q[c] = v_last[A, c] * p
                            dq[c] = -(2 * l + 1) * q[c] * inv
                else:
                    lo = 0
                    hi = nk - 1
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if knots[A, mid] <= r:
                            lo = mid
                        else:
                            hi = mid
                    seg = lo
                    t = r - knots[A, lo]
                    for l in range(l_max + 1):
                        p = inv ** l
                        for c in range(l * l, (l + 1) * (l + 1)):
                            c0 = coefs[A, 0, seg, c]
                            c1 = coefs[A, 1, seg, c]
                            c2 = coefs[A, 2, seg, c]
                            v = ((c0 * t + c1) * t + c2) * t + coefs[A, 3, seg, c]
                            dv = (3.0 * c0 * t + 2.0 * c1) * t + c2
                            q[c] = v * p
                            dq[c] = (dv - l * v * inv) * p
            # -- solid harmonics and their Cartesian derivatives ---------------
            S[0] = 1.0
            Sx[0] = 0.0
            Sy[0] = 0.0
            Sz[0] = 0.0
            for l in range(1, l_max + 1):
                f = np.sqrt((2.0 * l - 1.0) / (2.0 * l))
                cc = (l - 1) * (l - 1) + 2 * (l - 1)
                cs = (l - 1) * (l - 1)
                pc = S[cc]
                pcx = Sx[cc]
                pcy = Sy[cc]
                pcz = Sz[cc]
                if l > 1:
                    ps = S[cs]
                    psx = Sx[cs]
                    psy = Sy[cs]
                    psz = Sz[cs]
                else:
                    ps = 0.0
                    psx = 0.0
                    psy = 0.0
                    psz = 0.0
                i = l * l + 2 * l
                S[i] = f * (x * pc - y * ps)
                Sx[i] = f * (pc + x * pcx - y * psx)
                Sy[i] = f * (x * pcy - ps - y * psy)
                Sz[i] = f * (x * pcz - y * psz)
                i = l * l
                S[i] = f * (y * pc + x * ps)
                Sx[i] = f * (y * pcx + ps + x * psx)
                Sy[i] = f * (pc + y * pcy + x * psy)
                Sz[i] = f * (y * pcz + x * psz)
                for m in range(-(l - 1), l):
                    a1 = (l - 1) * (l - 1) + m + l - 1
                    c1_ = 2.0 * l - 1.0
                    num = c1_ * z * S[a1]
                    nx = c1_ * z * Sx[a1]
                    ny = c1_ * z * Sy[a1]
                    nz = c1_ * (S[a1] + z * Sz[a1])
                    if l >= 2 and abs(m) <= l - 2:
                        a2 = (l - 2) * (l - 2) + m + l - 2
                        c2_ = np.sqrt((l - 1.0) ** 2 - m * m)
                        num -= c2_ * r2 * S[a2]
                        nx -= c2_ * (2.0 * x * S[a2] + r2 * Sx[a2])
                        ny -= c2_ * (2.0 * y * S[a2] + r2 * Sy[a2])
                        nz -= c2_ * (2.0 * z * S[a2] + r2 * Sz[a2])
                    den = 1.0 / np.sqrt(l * l - m * m)
                    i = l * l + m + l
                    S[i] = num * den
                    Sx[i] = nx * den
                    Sy[i] = ny * den
                    Sz[i] = nz * den
            # -- accumulate q' rhat S + q grad S -------------------------------
            radial = 0.0
            for c in range(n_lm):
                radial += dq[c] * S[c] * norms[c]
                gx += q[c] * Sx[c] * norms[c]
                gy += q[c] * Sy[c] * norms[c]
                gz += q[c] * Sz[c] * norms[c]
            if r > 1e-300:
                radial /= r
                gx += radial * x
                gy += radial * y
                gz += radial * z
        out[k, 0] = gx
        out[k, 1] = gy
        out[k, 2] = gz


@nb.njit(cache=True)
def _smooth_step_and_derivative(nu, stiffness):
    f = min(max(nu, -1.0), 1.0)
    df = 1.0
    for _ in range(stiffness):
        df *= 1.5 * (1.0 - f * f)
        f = 1.5 * f - 0.5 * f * f * f
    return 0.5 * (1.0 - f), -0.5 * df


@nb.njit(parallel=True, cache=True)
def own_cell_weight_gradient(points, order, starts, coeff, centres, a, stiffness, out):
    """``out[A, B] = sum_{k owned by A} coeff_k d w_A(r_k) / d R_B`` (points fixed).

    ``order[starts[A]:starts[A+1]]`` lists the points owned by atom A; each
    thread handles whole atoms, so it writes only its own ``out[A]`` rows.
    Same expressions as grid/becke.py:own_cell_weight_gradient.
    """
    n_at = centres.shape[0]
    for A in nb.prange(n_at):
        dist = np.empty(n_at)
        unit = np.empty((n_at, 3))
        P = np.empty(n_at)
        s = np.empty((n_at, n_at))
        dln = np.empty((n_at, n_at))
        for q in range(starts[A], starts[A + 1]):
            k = order[q]
            for i in range(n_at):
                d0 = points[k, 0] - centres[i, 0]
                d1 = points[k, 1] - centres[i, 1]
                d2 = points[k, 2] - centres[i, 2]
                dist[i] = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                inv = 1.0 / max(dist[i], 1e-300)
                unit[i, 0] = d0 * inv
                unit[i, 1] = d1 * inv
                unit[i, 2] = d2 * inv
            for i in range(n_at):
                P[i] = 1.0
                for j in range(n_at):
                    if i == j:
                        continue
                    R0 = centres[i, 0] - centres[j, 0]
                    R1 = centres[i, 1] - centres[j, 1]
                    R2 = centres[i, 2] - centres[j, 2]
                    Rij = np.sqrt(R0 * R0 + R1 * R1 + R2 * R2)
                    mu = (dist[i] - dist[j]) / Rij
                    sv, dsv = _smooth_step_and_derivative(mu + a[i, j] * (1.0 - mu * mu),
                                                          stiffness)
                    s[i, j] = sv
                    dln[i, j] = (dsv / sv if sv > 1e-300 else 0.0) * (1.0 - 2.0 * a[i, j] * mu)
                    P[i] *= sv
            Z = 0.0
            for i in range(n_at):
                Z += P[i]
            if Z <= 1e-300:
                Z = 1.0
            w_own = P[A] / Z
            for i in range(n_at):
                pre = coeff[k] * ((1.0 if i == A else 0.0) - w_own) / Z * P[i]
                if pre == 0.0:
                    continue
                for j in range(n_at):
                    if i == j:
                        continue
                    R0 = centres[i, 0] - centres[j, 0]
                    R1 = centres[i, 1] - centres[j, 1]
                    R2 = centres[i, 2] - centres[j, 2]
                    Rij2 = R0 * R0 + R1 * R1 + R2 * R2
                    Rij = np.sqrt(Rij2)
                    mu = (dist[i] - dist[j]) / Rij
                    alpha = pre * dln[i, j]
                    # d mu / d R_i and d mu / d R_j
                    out[A, i, 0] += alpha * (-unit[i, 0] / Rij - mu * R0 / Rij2)
                    out[A, i, 1] += alpha * (-unit[i, 1] / Rij - mu * R1 / Rij2)
                    out[A, i, 2] += alpha * (-unit[i, 2] / Rij - mu * R2 / Rij2)
                    out[A, j, 0] += alpha * (unit[j, 0] / Rij + mu * R0 / Rij2)
                    out[A, j, 1] += alpha * (unit[j, 1] / Rij + mu * R1 / Rij2)
                    out[A, j, 2] += alpha * (unit[j, 2] / Rij + mu * R2 / Rij2)
