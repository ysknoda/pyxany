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
    #: Input (second-row) radial size; atoms may carry more, see ``n_r_atom``.
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
    #: Radial nodes of each atom (one array per atom; lengths may differ).
    radii: list = None
    n_atoms: int = 0
    #: Radial points of each atom's sub-grid.
    n_r_atom: np.ndarray = None
    #: Angular pruning: per atom, ``[(shell_lo, shell_hi, n_a), ...]`` contiguous
    #: runs of radial shells sharing one angular grid (None: every shell uses n_a).
    segments: list = None
    _slice: slice = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._slice is None:
            self._slice = local_slice(len(self.points))
        if self.n_r_atom is None:
            self.n_r_atom = np.full(self.n_atoms, self.n_r, dtype=int)
        if self.directions is None:
            self._starts = None
            return
        if self.segments is None:
            self.segments = [[(0, int(n), self.n_a)] for n in self.n_r_atom]
        sizes = [sum((hi - lo) * 2 * na * na for lo, hi, na in seg) for seg in self.segments]
        self._starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    @property
    def pruned(self) -> bool:
        return any(na != self.n_a for seg in (self.segments or []) for _, _, na in seg)

    def shell_blocks(self, ai: int):
        """Yield ``(shell_lo, shell_hi, n_a, rows)`` for atom ``ai``'s angular runs.

        ``rows`` is the slice of the full grid holding those shells, ordered
        shell-major, so ``values[rows].reshape(shell_hi - shell_lo, 2 n_a^2)``.
        """
        start = int(self._starts[ai])
        for lo, hi, na in self.segments[ai]:
            n = (hi - lo) * 2 * na * na
            yield lo, hi, na, slice(start, start + n)
            start += n

    def __len__(self) -> int:
        return len(self.points)

    # -- per-atom block structure -----------------------------------------
    @property
    def n_angular(self) -> int:
        return len(self.directions)

    def atom_block(self, ai: int) -> slice:
        """Rows of the full grid produced by atom ``ai``'s sub-grid."""
        return slice(int(self._starts[ai]), int(self._starts[ai + 1]))

    def as_radial_angular(self, values: np.ndarray, ai: int) -> np.ndarray:
        """Reshape a per-point array's atom-``ai`` block to ``(n_r_atom[ai], n_angular)``.

        Only for an unpruned grid; a pruned one is walked with :meth:`shell_blocks`.
        """
        if self.pruned:
            raise ValueError("as_radial_angular needs an unpruned grid; use shell_blocks")
        return values[self.atom_block(ai)].reshape(int(self.n_r_atom[ai]), self.n_angular)

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
    radial_by_element: bool = True,
    n_r_by_element: dict | None = None,
    pruning: list | None = None,
) -> Grid:
    """Superposition of atom-centred TA-radial x Gauss-Legendre-angular grids.

    ``n_r`` is the radial size for second-row atoms. With ``radial_by_element``
    each atom gets ``ceil(n_r * ratio)`` points, the ratio measured per element
    (data/tables/radial_points.json) so that every element reaches the
    ionisation-energy accuracy C-F reach at ``n_r``: Ga and As get 47 at
    ``n_r = 30``, Cd 60. Pseudo-hydrogens use hydrogen's ratio.

    The volume element is ``dV = r^2 dr d(cos theta) d phi`` -- the angular
    weights carry no ``sin(theta)`` factor (see :mod:`pyxany.grid.angular`).

    ``n_r_by_element`` (``{"Si": 40}``) overrides the radial size of the
    elements it names. ``pruning`` (``[(r/R upper bound, n_a), ...]``) gives
    shells with ``r < bound * R_Bragg`` the smaller angular grid of the first
    region that contains them (ADR 0009).

    With ``partition="becke"`` each point's weight is multiplied by the Becke
    cell weight of its parent atom, which removes the ``N_atoms``-fold
    overcounting of a bare superposition.  ``partition="none"`` is provided only
    so that tests can demonstrate the difference; it is not a supported mode for
    production calculations.
    """
    with timer("grid.build"):
        dirs, w_ang = spherical_product(n_a)

        n_r_atom = radial_sizes(molecule, n_r) if radial_by_element \
            else np.full(len(molecule), n_r, dtype=int)
        if n_r_by_element:
            n_r_atom = np.array([n_r_by_element.get(a.element, n) for a, n in
                                 zip(molecule, n_r_atom, strict=True)], dtype=int)
        regions = sorted(pruning or [])
        angular = {n_a: (dirs, w_ang)}
        all_pts, all_wts, all_owner, all_radii, segments = [], [], [], [], []
        for ai, atom in enumerate(molecule):
            r, w_r = treutler_ahlrichs_m4(int(n_r_atom[ai]), atom.grid_radius())
            all_radii.append(r)
            shell_na = np.full(len(r), n_a, dtype=int)
            bragg = atom.grid_radius()
            for bound, na in reversed(regions):     # innermost region wins
                shell_na[r < bound * bragg] = na
            seg = []
            for k in range(len(r)):
                if seg and seg[-1][2] == shell_na[k]:
                    seg[-1][1] = k + 1
                else:
                    seg.append([k, k + 1, int(shell_na[k])])
            segments.append([tuple(x) for x in seg])
            for lo, hi, na in seg:
                if na not in angular:
                    angular[na] = spherical_product(na)
                d, wa = angular[na]
                rr, ww = r[lo:hi], w_r[lo:hi]
                # outer product of radial and angular grids
                all_pts.append((rr[:, None, None] * d[None, :, :]).reshape(-1, 3) + atom.pos)
                all_wts.append((ww[:, None] * rr[:, None] ** 2 * wa[None, :]).reshape(-1))
                all_owner.append(np.full((hi - lo) * len(d), ai, dtype=np.int32))

        points = np.vstack(all_pts)
        raw_weights = np.concatenate(all_wts)
        owner = np.concatenate(all_owner)

        if partition not in ("becke", "none"):
            raise ValueError(f"unknown grid partition scheme '{partition}'")

        if partition == "becke" and len(molecule) > 1:
            with timer("grid.becke"):
                radii = np.array([a.grid_radius() for a in molecule])
                from ..kernels import becke_own_cell
                from .becke import size_adjustment

                cell = becke_own_cell(points, owner, molecule.positions,
                                      size_adjustment(radii), stiffness)
                if cell is None:  # numpy reference
                    w_cell = becke_weights(points, molecule.positions, radii, stiffness)
                    cell = w_cell[np.arange(len(points)), owner]
        else:
            cell = np.ones(len(points))

        return Grid(points=points, weights=raw_weights * cell, owner=owner,
                    n_r=n_r, n_a=n_a, partition=partition,
                    raw_weights=raw_weights, cell_weights=cell,
                    directions=dirs, angular_weights=w_ang,
                    radii=all_radii, n_atoms=len(molecule), n_r_atom=n_r_atom,
                    segments=segments)


def grid_from_config(molecule, grid_cfg, n_r: int | None = None, n_a: int | None = None):
    """The molecular grid an input file asks for (``grid:`` section).

    ``n_r`` / ``n_a`` override the configured sizes (used by ``gridcheck``);
    automatic pruning regions follow the n_a actually used.
    """
    from ..config import DEFAULT_PRUNING

    n_r = grid_cfg.n_r if n_r is None else n_r
    n_a = grid_cfg.n_a if n_a is None else n_a
    pruning = []
    if grid_cfg.pruning.get("enabled", True):
        regions = grid_cfg.pruning.get("regions", "auto")
        regions = DEFAULT_PRUNING(n_a) if regions == "auto" else regions
        pruning = [(float(b), min(int(na), n_a)) for b, na in regions if int(na) < n_a]
    return build_molecular_grid(molecule, n_r, n_a, partition=grid_cfg.partition,
                                radial_by_element=grid_cfg.radial_by_element,
                                n_r_by_element=grid_cfg.n_r_by_element or None,
                                pruning=pruning)


def radial_sizes(molecule, n_r: int) -> np.ndarray:
    """Per-atom radial points: ``ceil(n_r * ratio(element))``, at least 5."""
    from ..data.loaders import radial_point_ratios

    ratios = radial_point_ratios()
    # ratios are stored to 4 decimals (47/30 -> 1.5667), so 30 * ratio may land a
    # hair above the integer it encodes; 0.01 absorbs that rounding only
    return np.array([max(5, int(np.ceil(n_r * ratios.get(a.element, 1.0) - 0.01)))
                     for a in molecule], dtype=int)


def two_centre_grid(centre, n_r: int, n_a: int, scale: float = 1.0):
    """Single-centre Gauss-Legendre 3-D grid, used for two-centre S/T integrals.

    Returns ``(points, weights)`` with ``dV = r^2 dr d(cos theta) d phi``.
    """
    dirs, w_ang = spherical_product(n_a)
    r, w_r = log_squared_map(n_r, scale)
    pts = (r[:, None, None] * dirs[None, :, :]).reshape(-1, 3) + np.asarray(centre, dtype=float)
    wts = (w_r[:, None] * r[:, None] ** 2 * w_ang[None, :]).reshape(-1)
    return pts, wts
