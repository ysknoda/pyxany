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


def n_angular_points(n_theta: int, n_phi: int | None = None) -> int:
    return n_theta * (2 * n_theta if n_phi is None else n_phi)
