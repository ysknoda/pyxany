"""Electric dipole moment from the DV grid."""

from __future__ import annotations

import numpy as np

from ..constants import AU_TO_DEBYE
from ..parallel import allreduce_array


def dipole_moment(molecule, grid, rho_local: np.ndarray,
                  origin: np.ndarray | None = None) -> np.ndarray:
    """Dipole vector in Debye, measured about ``origin``.

    ``mu = sum_A Z_A (R_A - o) - int (r - o) rho(r) dV``.  The electronic part
    is summed over the rank-local grid slice and reduced, so the result is the
    same in serial and parallel runs.

    For a **neutral** molecule the dipole does not depend on ``o`` and the
    choice is immaterial.  For a **charged** one it does -- the two terms no
    longer cancel -- so a cluster cut from an ionic crystal, which frequently
    carries a net charge, gives a different answer for every origin: water
    translated by 3 A keeps |mu| = 1.85 D when neutral but goes from 2.98 D to
    16.41 D as a cation.  ``origin`` therefore defaults to the centre of
    nuclear charge rather than to the coordinate origin, which at least makes
    the value translation-invariant and reproducible; callers that need a
    particular convention should pass it explicitly.
    """
    if origin is None:
        origin = centre_of_nuclear_charge(molecule)
    origin = np.asarray(origin, dtype=float)

    electronic = -np.einsum("k,kx->x", grid.local_weights * rho_local,
                            grid.local_points - origin[None, :])
    electronic = allreduce_array(np.ascontiguousarray(electronic))
    nuclear = np.einsum("a,ax->x", molecule.charges,
                        molecule.positions - origin[None, :])
    return (nuclear + electronic) * AU_TO_DEBYE


def centre_of_nuclear_charge(molecule) -> np.ndarray:
    """``sum_A Z_A R_A / sum_A Z_A`` -- the default dipole origin."""
    charges = np.asarray(molecule.charges, dtype=float)
    total = float(charges.sum())
    if abs(total) < 1e-12:  # pragma: no cover - a molecule with no nuclei
        return np.zeros(3)
    return np.einsum("a,ax->x", charges, molecule.positions) / total
