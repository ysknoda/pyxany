"""SCF checkpoint and restart (ADR 0004).

A checkpoint is a NumPy ``.npz`` holding the density matrix (both spin
channels for UKS), the orbitals, eigenvalues and occupations of the last
completed iteration, the iteration count, and two fingerprints:

* ``system`` -- everything that must match for the stored density to mean the
  same thing: nuclear charges, charge, multiplicity, basis-function list,
  restricted/unrestricted, X-alpha alpha. A mismatch is an error.
* ``geometry`` -- the nuclear positions. A mismatch with an otherwise
  identical system (the next point of a scan or optimization) downgrades the
  restart to "use this density as the initial guess", with a warning.

DIIS history is deliberately not stored: it is up to ten Fock/error pairs, so
it would multiply the file size about tenfold for a 200-atom cluster, and it
is rebuilt within a few iterations of a restart.

Files are written by rank 0 to a temporary name in the same directory and
moved into place with ``os.replace``, which is atomic on POSIX: a job killed
mid-write leaves the previous checkpoint intact.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ConfigError
from ..parallel import is_root, mpi_print

FORMAT_VERSION = 1


def _digest(*arrays_or_text) -> str:
    h = hashlib.sha256()
    for item in arrays_or_text:
        if isinstance(item, str):
            h.update(item.encode())
        else:
            h.update(np.ascontiguousarray(np.asarray(item, dtype=float)).tobytes())
    return h.hexdigest()[:24]


def system_fingerprint(molecule, basis, config, unrestricted: bool) -> str:
    """Identity of the system a checkpoint belongs to.

    Hashes the alpha *values* per atom (not the repr of the setting, whose
    mapping order and table name hid equal or different physics) and the
    parameters of every function (NAO/GTO hashed zeta = 0, so another NAO
    basis with the same labels passed; ADR 0012 P0).
    """
    labels = "|".join(f"{f.atom_idx}:{f.label}:{type(f).__name__}" for f in basis)
    alphas = [round(float(config.scf.alpha_for(a.element if not a.is_pseudo else "H")), 12)
              for a in molecule]
    return _digest(
        np.asarray(molecule.charges),
        f"q={molecule.charge};m={molecule.multiplicity};u={unrestricted};"
        f"ts={_ts_key(config)};emb={_embedding_key(config)};{labels}",
        np.asarray(alphas),
        np.concatenate([_function_parameters(f) for f in basis]) if len(basis) else [],
    )


def _function_parameters(f) -> np.ndarray:
    """Numbers that define a basis function's radial shape."""
    if hasattr(f, "zeta"):
        return np.atleast_1d(np.asarray(f.zeta, dtype=float))
    if hasattr(f, "prims"):
        return np.round(np.asarray(f.prims, dtype=float).ravel(), 12)
    if hasattr(f, "radial_grid"):
        # a few samples of R(r) pin the solved orbital
        return np.round(np.asarray(f.radial_grid, dtype=float)[::97], 10)
    return np.zeros(1)


def _embedding_key(config) -> str:
    """Different point charges make a different system."""
    from ..embedding import from_config

    ext = from_config(config)
    if ext is None:
        return "none"
    return _digest(np.round(ext.positions, 10), ext.charges, f"s={ext.sigma!r}")


def _ts_key(config) -> str:
    """A transition-state run is a different system from the ground state."""
    ts = getattr(config, "transition_state", None)
    if ts is None or not ts.enabled:
        return "none"
    return f"{ts.remove!r}|{ts.excite!r}|{ts.spin}|{ts.electrons!r}"


def geometry_fingerprint(molecule) -> str:
    return _digest(np.round(np.asarray(molecule.positions), 10))


@dataclass
class Checkpoint:
    P: np.ndarray
    P_a: np.ndarray | None
    P_b: np.ndarray | None
    iteration: int
    system: str
    geometry: str
    energy: float
    converged: bool


def write(
    path,
    *,
    P,
    P_a=None,
    P_b=None,
    C=None,
    eps=None,
    occ=None,
    iteration: int,
    system: str,
    geometry: str,
    energy: float,
    converged: bool,
) -> None:
    """Atomically write a checkpoint (rank 0 only)."""
    if not is_root():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arrays = {
        "P": P,
        "format_version": np.array(FORMAT_VERSION),
        "iteration": np.array(iteration),
        "energy": np.array(energy),
        "converged": np.array(converged),
        "system": np.array(system),
        "geometry": np.array(geometry),
    }
    for name, value in (("P_a", P_a), ("P_b", P_b), ("C", C), ("eps", eps), ("occ", occ)):
        if value is not None:
            arrays[name] = value
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read(path) -> Checkpoint:
    with np.load(path, allow_pickle=False) as d:
        version = int(d["format_version"])
        if version != FORMAT_VERSION:
            raise ConfigError(f"{path}: checkpoint format {version}, expected {FORMAT_VERSION}")
        return Checkpoint(
            P=d["P"],
            P_a=d["P_a"] if "P_a" in d else None,
            P_b=d["P_b"] if "P_b" in d else None,
            iteration=int(d["iteration"]),
            system=str(d["system"]),
            geometry=str(d["geometry"]),
            energy=float(d["energy"]),
            converged=bool(d["converged"]),
        )


def initial_density(
    ckpt: Checkpoint, system: str, geometry: str, n_basis: int, path="checkpoint", verbose: int = 1
) -> tuple[dict, int]:
    """Validate a checkpoint against the run and return ``(guess, start_iteration)``.

    Same system and geometry: resume, continuing the iteration count. Same
    system, different geometry: initial guess only, count restarts at 0.
    Different system: error, since the density would mean something else.
    """
    if ckpt.P.shape != (n_basis, n_basis):
        raise ConfigError(
            f"{path}: density is {ckpt.P.shape[0]} x {ckpt.P.shape[1]} but this "
            f"basis has {n_basis} functions; it is a different system"
        )
    if ckpt.system != system:
        raise ConfigError(
            f"{path}: written for a different system (atoms, charge, "
            f"multiplicity, basis or alpha differ); refusing to restart from it"
        )
    guess = {"P": ckpt.P, "P_a": ckpt.P_a, "P_b": ckpt.P_b}
    if ckpt.geometry != geometry:
        if verbose:
            mpi_print(
                f"  [restart] {path}: geometry differs -- using its density as the "
                f"initial guess only"
            )
        return guess, 0
    if verbose:
        mpi_print(
            f"  [restart] resuming from {path} at iteration {ckpt.iteration} "
            f"(E = {ckpt.energy:.9f} Ha; DIIS history rebuilt)"
        )
    return guess, ckpt.iteration
