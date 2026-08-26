"""Grid-integrated matrices, either from a cached chi or in bounded-memory chunks.

The chi matrix is ``(n_grid_points x n_basis)``.  For a few hundred atoms in a
polarised basis that is tens of gigabytes on one rank, so it cannot always be
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
"""

from __future__ import annotations

import numpy as np

from ..basis.evaluate import chi_matrix
from ..parallel import allreduce_array
from ..timing import timer


class GridTerms:
    """Evaluates density and potential matrices on the grid.

    ``chunk_size=None`` caches chi; an integer processes that many points at a
    time and never materialises the whole thing.
    """

    def __init__(self, basis, molecule, grid, chunk_size: int | None = None):
        self.basis = basis
        self.molecule = molecule
        self.grid = grid
        self.chunk_size = chunk_size
        self.n_basis = len(basis)
        self._chi = None
        if chunk_size is None:
            with timer("scf.chi"):
                self._chi = chi_matrix(basis, molecule.positions, grid.local_points)

    # -- introspection -----------------------------------------------------
    @property
    def cached(self) -> bool:
        return self._chi is not None

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
