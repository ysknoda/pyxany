"""Grid-integrated matrices, either from a cached chi or in bounded-memory chunks.

The chi matrix is ``(n_grid_points x n_basis)``.  For a few hundred atoms in a
polarized basis that is tens of gigabytes on one rank, so it cannot always be
held.  Two strategies, chosen from the memory budget:

**cached**  build chi once and reuse it for every term.  Fastest, and what a
molecule-sized job should do.

**chunked** walk the grid in blocks, accumulating the ``n_basis x n_basis``
matrices as we go.  Memory drops from ``O(N_grid n_basis)`` to
``O(chunk n_basis)``; chi is evaluated twice per SCF iteration (once to build
the density the Hartree solver needs, once to contract the potentials) instead
of once, which is the price of the bound.

Both paths must produce identical numbers -- ``tests/test_chunking.py`` asserts
it on a real SCF.

**block-sparse** (``grid.basis_cutoff > 0``, ADR 0009) splits the grid into
spatially compact blocks (median bisection) and keeps, per block, only the
atoms whose basis functions can exceed the cutoff anywhere in it. Each block's
products then cost ``n_active^2`` instead of ``n_basis^2``; the dropped values
are below the cutoff by construction. Blocks are cached when their total size
fits the budget, otherwise rebuilt on every pass.
"""

from __future__ import annotations

import numpy as np

from ..basis.evaluate import chi_matrix
from ..parallel import allreduce_array
from ..timing import timer


class GridTerms:
    """Evaluates density and potential matrices on the grid.

    ``chunk_size=None`` caches chi; an integer processes that many points at a
    time and never materializes the whole thing.
    """

    def __init__(self, basis, molecule, grid, chunk_size: int | None = None,
                 cutoff: float = 0.0, budget_mb: float = 4096.0):
        self.basis = basis
        self.molecule = molecule
        self.grid = grid
        self.chunk_size = chunk_size
        self.n_basis = len(basis)
        self.cutoff = float(cutoff)
        self._chi = None
        self._sparse = None
        if self.cutoff > 0:
            with timer("scf.chi"):
                self._sparse = SparseBlocks(basis, molecule, grid.local_points, self.cutoff,
                                            budget_mb)
            return
        if chunk_size is None:
            with timer("scf.chi"):
                self._chi = chi_matrix(basis, molecule.positions, grid.local_points)

    # -- introspection -----------------------------------------------------
    @property
    def cached(self) -> bool:
        if self._sparse is not None:
            return self._sparse.cached
        return self._chi is not None

    @property
    def sparsity(self) -> dict | None:
        """Block-sparse statistics, or None for the dense paths."""
        return None if self._sparse is None else self._sparse.stats()

    @property
    def chi(self) -> np.ndarray | None:
        """The cached chi matrix, or None when running chunked."""
        return self._chi

    def blocks(self):
        """Yield ``(chi_block, weights_block)`` covering this rank's grid slice."""
        if self._chi is not None:
            yield self._chi, self.grid.local_weights
            return
        positions = self.molecule.positions
        for points, weights in self.grid.chunks(self.chunk_size):
            yield chi_matrix(self.basis, positions, points), weights

    # -- quantities --------------------------------------------------------
    def density(self, P: np.ndarray) -> np.ndarray:
        """rho on this rank's grid slice."""
        if self._sparse is not None:
            rho = np.zeros(self.grid.n_local)     # points no function reaches stay 0
            for idx, act, chi in self._sparse.blocks():
                rho[idx] = np.einsum("km,km->k", chi @ P[np.ix_(act, act)], chi)
            return rho
        rho = np.empty(self.grid.n_local)
        offset = 0
        for chi, _w in self.blocks():
            n = len(chi)
            rho[offset:offset + n] = np.einsum("km,mn,kn->k", chi, P, chi,
                                               optimize=True)
            offset += n
        return rho

    def potential_matrix(self, v: np.ndarray, reduce: bool = True) -> np.ndarray:
        """``<mu| v |nu>`` for a potential sampled on this rank's grid slice."""
        if self._sparse is not None:
            return self.potential_matrices({"v": v}, reduce=reduce)["v"]
        M = np.zeros((self.n_basis, self.n_basis))
        offset = 0
        for chi, w in self.blocks():
            n = len(chi)
            M += (chi * (w * v[offset:offset + n])[:, None]).T @ chi
            offset += n
        return allreduce_array(M) if reduce else M

    def potential_matrices(self, potentials: dict, reduce: bool = True) -> dict:
        """Several potential matrices in one pass over the grid.

        Under chunking this is what keeps the extra chi evaluations to one per
        iteration rather than one per term.
        """
        out = {k: np.zeros((self.n_basis, self.n_basis)) for k in potentials}
        if self._sparse is not None:
            w_all = self.grid.local_weights
            for idx, act, chi in self._sparse.blocks():
                ix = np.ix_(act, act)
                for key, v in potentials.items():
                    out[key][ix] += (chi * (w_all[idx] * v[idx])[:, None]).T @ chi
            if reduce:
                for m in out.values():
                    allreduce_array(m)
            return out
        offset = 0
        for chi, w in self.blocks():
            n = len(chi)
            for key, v in potentials.items():
                out[key] += (chi * (w * v[offset:offset + n])[:, None]).T @ chi
            offset += n
        if reduce:
            for m in out.values():
                allreduce_array(m)
        return out


def choose_strategy(grid, n_basis: int, setting, budget_mb: float):
    """Resolve ``grid.chunk_size`` to an explicit chunk size or None.

    ``'auto'`` caches chi when it fits a quarter of the budget (several working
    copies are alive at once) and otherwise picks the largest block that does.
    """
    if setting == "auto" or setting is None:
        return grid.choose_chunk_size(n_basis, budget_mb)
    size = int(setting)
    return None if size <= 0 or size >= grid.n_local else size


def basis_reach(basis, molecule, tol: float) -> np.ndarray:
    """Per atom, the distance beyond which all its basis functions are < ``tol``.

    Sampled per species on a lone atom (radial step 0.1 Bohr out to 120 Bohr,
    a 10 x 20 angular product grid), plus one step of margin.
    """
    import copy

    from ..grid.angular import spherical_product

    dirs, _ = spherical_product(10)
    r = np.arange(0.05, 120.0, 0.1)
    pts = (r[:, None, None] * dirs[None]).reshape(-1, 3)
    by_species: dict[str, float] = {}
    reach = np.zeros(len(molecule))
    for ai, atom in enumerate(molecule):
        if atom.symbol not in by_species:
            local = []
            for f in basis:
                if f.atom_idx == ai:
                    g = copy.copy(f)
                    g.atom_idx = 0
                    local.append(g)
            if not local:
                by_species[atom.symbol] = 0.0
            else:
                peak = np.abs(chi_matrix(local, np.zeros((1, 3)), pts)).reshape(
                    len(r), -1).max(axis=1)
                above = np.where(peak >= tol)[0]
                by_species[atom.symbol] = (r[above.max()] + 0.1) if len(above) else 0.0
        reach[ai] = by_species[atom.symbol]
    return reach


def spatial_blocks(points: np.ndarray, leaf: int = 2048) -> list[np.ndarray]:
    """Index sets of compact point blocks: bisect at the median of the longest axis."""
    out = []
    stack = [np.arange(len(points))]
    while stack:
        idx = stack.pop()
        if len(idx) <= leaf:
            out.append(np.sort(idx))
            continue
        p = points[idx]
        axis = int(np.argmax(p.max(axis=0) - p.min(axis=0)))
        order = np.argsort(p[:, axis], kind="stable")
        half = len(idx) // 2
        stack.append(idx[order[half:]])
        stack.append(idx[order[:half]])
    return out


class SparseBlocks:
    """Grid blocks with the basis functions that matter on each (ADR 0009)."""

    def __init__(self, basis, molecule, points, tol: float, budget_mb: float,
                 leaf: int = 2048):
        self.basis = basis
        self.positions = molecule.positions
        self.points = points
        reach = basis_reach(basis, molecule, tol)
        by_atom: dict[int, list[int]] = {}
        for mu, f in enumerate(basis):
            by_atom.setdefault(f.atom_idx, []).append(mu)
        self.layout = []                       # (point indices, active functions)
        for idx in spatial_blocks(points, leaf):
            lo, hi = points[idx].min(axis=0), points[idx].max(axis=0)
            # distance from each nucleus to the block's bounding box
            gap = np.maximum(np.maximum(lo - self.positions, self.positions - hi), 0.0)
            near = np.where(np.linalg.norm(gap, axis=1) < reach)[0]
            act = np.array([mu for a in near for mu in by_atom.get(int(a), [])], dtype=int)
            if len(act):
                self.layout.append((idx, act))
        size_mb = sum(len(i) * len(a) for i, a in self.layout) * 8 / 1024**2
        self.size_mb = size_mb
        self.cached = size_mb <= budget_mb / 4.0
        self._chi = [self._evaluate(i, a) for i, a in self.layout] if self.cached else None

    def _evaluate(self, idx, act):
        return chi_matrix([self.basis[m] for m in act], self.positions, self.points[idx])

    def blocks(self):
        """Yield ``(point indices, active function indices, chi block)``."""
        for k, (idx, act) in enumerate(self.layout):
            yield idx, act, (self._chi[k] if self._chi is not None else self._evaluate(idx, act))

    def stats(self) -> dict:
        n = sum(len(i) for i, _ in self.layout)
        kept = sum(len(i) * len(a) for i, a in self.layout)
        return {
            "blocks": len(self.layout),
            "kept_fraction": kept / max(1, n * len(self.basis)),
            "chi_mb": self.size_mb,
            "cached": self.cached,
        }
