"""The molecular DV integration grid.

A ``Grid`` owns the full set of quadrature points and weights, knows which
contiguous slice belongs to the current MPI rank, and can hand out chunks so
that large (n_points x n_basis) intermediates never have to be materialised.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ..parallel import local_slice
from ..timing import timer
from .angular import spherical_product
from .becke import DEFAULT_STIFFNESS, becke_weights
from .radial import log_squared_map, treutler_ahlrichs_m4

#: Bytes per float64 entry of the chi matrix, used for the memory estimate.
_BYTES = 8


@dataclass
class Grid:
    """Molecular quadrature grid.

    Attributes
    ----------
    points   : (n_points, 3) Bohr
    weights  : (n_points,) volume-element weights, Becke-partitioned
    owner    : (n_points,) index of the atom whose sub-grid produced the point
    """

    points: np.ndarray
    weights: np.ndarray
    owner: np.ndarray
    n_r: int = 0
    n_a: int = 0
    partition: str = "becke"
    #: Quadrature weights *before* the Becke cell factor.
    raw_weights: np.ndarray = None
    #: Becke cell weight of each point's own parent atom (1.0 without partitioning).
    cell_weights: np.ndarray = None
    #: Unit direction vectors of one atomic sub-grid, shape (n_ang, 3).
    directions: np.ndarray = None
    #: Angular quadrature weights of one atomic sub-grid, shape (n_ang,).
    angular_weights: np.ndarray = None
    #: Radial nodes per atom, shape (n_atoms, n_r).
    radii: np.ndarray = None
    n_atoms: int = 0
    _slice: slice = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._slice is None:
            self._slice = local_slice(len(self.points))

    def __len__(self) -> int:
        return len(self.points)

    # -- per-atom block structure -----------------------------------------
    @property
    def n_angular(self) -> int:
        return len(self.directions)

    @property
    def block_size(self) -> int:
        """Points contributed by one atom's sub-grid."""
        return self.n_r * self.n_angular

    def atom_block(self, ai: int) -> slice:
        """Rows of the full grid produced by atom ``ai``'s sub-grid."""
        nb = self.block_size
        return slice(ai * nb, (ai + 1) * nb)

    def as_radial_angular(self, values: np.ndarray, ai: int) -> np.ndarray:
        """Reshape a per-point array's atom-``ai`` block to ``(n_r, n_angular)``."""
        return values[self.atom_block(ai)].reshape(self.n_r, self.n_angular)

    # -- MPI-local view ----------------------------------------------------
    @property
    def local_slice(self) -> slice:
        return self._slice

    @property
    def local_points(self) -> np.ndarray:
        return self.points[self._slice]

    @property
    def local_weights(self) -> np.ndarray:
        return self.weights[self._slice]

    @property
    def n_local(self) -> int:
        return self._slice.stop - self._slice.start

    # -- chunking ----------------------------------------------------------
    def chunks(self, chunk_size: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(points, weights)`` blocks of this rank's slice.

        ``chunk_size=None`` yields the whole local slice in one go.
        """
        pts = self.local_points
        wts = self.local_weights
        n = len(pts)
        if not chunk_size or chunk_size >= n:
            yield pts, wts
            return
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            yield pts[lo:hi], wts[lo:hi]

    def chi_memory_mb(self, n_basis: int, chunk_size: int | None = None) -> float:
        """Memory a chi matrix would take on this rank, in MB."""
        rows = self.n_local if not chunk_size else min(chunk_size, self.n_local)
        return rows * n_basis * _BYTES / (1024.0**2)

    def choose_chunk_size(self, n_basis: int, budget_mb: float) -> int | None:
        """Largest chunk fitting ``budget_mb``; ``None`` means 'no chunking needed'.

        A few working copies of the chi block are needed at once (chi, chi*w,
        and a gradient block during force evaluation), so the budget is divided
        by a safety factor of 4.
        """
        if self.chi_memory_mb(n_basis) <= budget_mb / 4.0:
            return None
        rows = int(max(1, (budget_mb / 4.0) * 1024**2 / (n_basis * _BYTES)))
        return min(rows, self.n_local)


def build_molecular_grid(
    molecule,
    n_r: int,
    n_a: int,
    partition: str = "becke",
    stiffness: int = DEFAULT_STIFFNESS,
) -> Grid:
    """Superposition of atom-centred TA-radial x Gauss-Legendre-angular grids.

    The volume element is ``dV = r^2 dr d(cos theta) d phi`` -- the angular
    weights carry no ``sin(theta)`` factor (see :mod:`pyxany.grid.angular`).

    With ``partition="becke"`` each point's weight is multiplied by the Becke
    cell weight of its parent atom, which removes the ``N_atoms``-fold
    overcounting of a bare superposition.  ``partition="none"`` is provided only
    so that tests can demonstrate the difference; it is not a supported mode for
    production calculations.
    """
    with timer("grid.build"):
        dirs, w_ang = spherical_product(n_a)
        n_ang = len(dirs)

        all_pts, all_wts, all_owner, all_radii = [], [], [], []
        for ai, atom in enumerate(molecule):
            r, w_r = treutler_ahlrichs_m4(n_r, atom.grid_radius())
            all_radii.append(r)
            # outer product of radial and angular grids
            pts = (r[:, None, None] * dirs[None, :, :]).reshape(-1, 3) + atom.pos
            wts = (w_r[:, None] * r[:, None] ** 2 * w_ang[None, :]).reshape(-1)
            all_pts.append(pts)
            all_wts.append(wts)
            all_owner.append(np.full(n_r * n_ang, ai, dtype=np.int32))

        points = np.vstack(all_pts)
        raw_weights = np.concatenate(all_wts)
        owner = np.concatenate(all_owner)

        if partition not in ("becke", "none"):
            raise ValueError(f"unknown grid partition scheme '{partition}'")

        if partition == "becke" and len(molecule) > 1:
            with timer("grid.becke"):
                radii = np.array([a.grid_radius() for a in molecule])
                w_cell = becke_weights(points, molecule.positions, radii, stiffness)
                cell = w_cell[np.arange(len(points)), owner]
        else:
            cell = np.ones(len(points))

        return Grid(points=points, weights=raw_weights * cell, owner=owner,
                    n_r=n_r, n_a=n_a, partition=partition,
                    raw_weights=raw_weights, cell_weights=cell,
                    directions=dirs, angular_weights=w_ang,
                    radii=np.array(all_radii), n_atoms=len(molecule))


def two_centre_grid(centre, n_r: int, n_a: int, scale: float = 1.0):
    """Single-centre Gauss-Legendre 3-D grid, used for two-centre S/T integrals.

    Returns ``(points, weights)`` with ``dV = r^2 dr d(cos theta) d phi``.
    """
    dirs, w_ang = spherical_product(n_a)
    r, w_r = log_squared_map(n_r, scale)
    pts = (r[:, None, None] * dirs[None, :, :]).reshape(-1, 3) + np.asarray(centre, dtype=float)
    wts = (w_r[:, None] * r[:, None] ** 2 * w_ang[None, :]).reshape(-1)
    return pts, wts
