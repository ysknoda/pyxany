"""Electric dipole moment from the DV grid."""

from __future__ import annotations

import numpy as np

from ..constants import AU_TO_DEBYE
from ..parallel import allreduce_array


def dipole_moment(molecule, grid, rho_local: np.ndarray) -> np.ndarray:
    """Dipole vector in Debye.

    ``mu = sum_A Z_A R_A - int r rho(r) dV``.  The electronic part is summed
    over the rank-local grid slice and reduced, so the result is the same in
    serial and parallel runs.
    """
    electronic = -np.einsum("k,kx->x", grid.local_weights * rho_local,
                            grid.local_points)
    electronic = allreduce_array(np.ascontiguousarray(electronic))
    nuclear = np.einsum("a,ax->x", molecule.charges, molecule.positions)
    return (nuclear + electronic) * AU_TO_DEBYE
