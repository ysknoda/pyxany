# 0004. SCF checkpoint and restart

- Status: Accepted
- Date: 2026-10-01

## Context

`scf.checkpoint_every` is currently rejected by the validator because no
restart exists: a job killed at a wall-clock limit loses all work. Scans and
optimizations also restart every point from an atomic-superposition guess.

## Decision

- Format: NumPy `.npz` (no new dependency).
- Contents: P (P_alpha, P_beta for UKS), C, eps, occ, iteration number, and
  two hashes: the *system* (nuclear charges, charge, multiplicity, basis
  functions, restricted/unrestricted, alpha) and the *geometry*.
- DIIS history is **not** stored (decided 2026-10-02, Q25): it holds up to ten
  Fock/error pairs, multiplying the file about tenfold for a 200-atom cluster,
  and is rebuilt within a few iterations; a restart only has to avoid losing
  the work, not reproduce the next iterate bit for bit.
- Written every N iterations to a temporary file and moved into place with an
  atomic rename, so a crash mid-write never corrupts the previous checkpoint.
- `--restart file.npz`: if both hashes match, resume from the stored density
  and iteration count. If only the geometry differs (next scan or
  optimization point), downgrade to "use as initial density" with a warning.
  A different system is an error.
- `scan` and `opt` reuse the previous point's density automatically.

## Consequences

`scf.checkpoint_every` becomes a real option; `docs/hpc.md` is updated.
