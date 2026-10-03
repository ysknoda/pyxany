"""Angular quadrature on the unit sphere.

Gauss-Legendre in ``cos(theta)`` crossed with a uniform (trapezoidal, exact for
trigonometric polynomials) rule in ``phi``.

.. warning::

   The volume element is ``dV = r^2 dr d(cos theta) d phi``.  Because the polar
   nodes are Gauss-Legendre nodes *in cos(theta)*, the weights must **not**
   carry an extra ``sin(theta)`` factor.  The original single-file version of
   this code did include one, which scaled every grid integral by
   ``(pi/2)/2 = pi/4 = 0.7854``; see docs/legacy_comparison.md.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def spherical_product(n_theta: int, n_phi: int | None = None):
    """Unit-sphere directions and weights with ``sum(w) == 4 pi``.

    Returns
    -------
    dirs : (n, 3) unit vectors
    w    : (n,) weights integrating ``f`` over the solid angle
    """
    n_phi = 2 * n_theta if n_phi is None else n_phi
    ct, wt = np.polynomial.legendre.leggauss(n_theta)
    st = np.sqrt(np.maximum(1.0 - ct**2, 0.0))
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    wp = np.full(n_phi, 2.0 * np.pi / n_phi)

    ct_g, phi_g = np.meshgrid(ct, phi, indexing="ij")
    st_g, _ = np.meshgrid(st, phi, indexing="ij")
    wt_g, wp_g = np.meshgrid(wt, wp, indexing="ij")

    dirs = np.column_stack([
        (st_g * np.cos(phi_g)).ravel(),
        (st_g * np.sin(phi_g)).ravel(),
        ct_g.ravel(),
    ])
    # d(cos theta) d phi  -- no sin(theta): the nodes already live in cos(theta)
    w = (wt_g * wp_g).ravel()
    return dirs, w


#: Orders (polynomial degree integrated exactly) scipy's Lebedev rule supports.
LEBEDEV_ORDERS = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35,
                  41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131)
ANGULAR_KINDS = ("product", "lebedev")


def lebedev_order_for(n_a: int) -> int:
    """Smallest Lebedev order matching an ``n_a x 2 n_a`` product grid.

    The product grid integrates spherical harmonics exactly up to degree
    ``2 n_a - 1``; the Lebedev rule of at least that order does the same.
    """
    need = 2 * int(n_a) - 1
    for order in LEBEDEV_ORDERS:
        if order >= need:
            return order
    raise ValueError(f"n_a = {n_a} needs Lebedev order {need}; the largest available is "
                     f"{LEBEDEV_ORDERS[-1]} (n_a <= {(LEBEDEV_ORDERS[-1] + 1) // 2})")


@lru_cache(maxsize=64)
def angular_grid(kind: str, n_a: int):
    """Directions and weights (summing to 4 pi) for accuracy level ``n_a``.

    ``product``: Gauss-Legendre x uniform, ``n_a x 2 n_a`` points.
    ``lebedev``: the Lebedev rule of the same polynomial degree (Lebedev &
    Laikov, Dokl. Math. 59, 477 (1999), via ``scipy.integrate.lebedev_rule``),
    about 2/3 of the points. Both resolve multipoles up to ``l = n_a - 1``.
    """
    if kind == "product":
        return spherical_product(n_a)
    if kind == "lebedev":
        from scipy.integrate import lebedev_rule

        x, w = lebedev_rule(lebedev_order_for(n_a))
        return np.ascontiguousarray(x.T), np.asarray(w, dtype=float)
    raise ValueError(f"unknown angular grid {kind!r}; use one of {ANGULAR_KINDS}")


def n_angular_points(n_theta: int, n_phi: int | None = None) -> int:
    return n_theta * (2 * n_theta if n_phi is None else n_phi)
