# 0004. SCF checkpoint and restart

- Status: Accepted
- Date: 2026-10-01

## Context

`scf.checkpoint_every` is currently rejected by the validator because no
restart exists: a job killed at a wall-clock limit loses all work. Scans and
optimisations also restart every point from an atomic-superposition guess.

## Decision

- Format: NumPy `.npz` (no new dependency).
- Contents: P (P_alpha, P_beta for UKS), C, eps, occ, iteration number, DIIS
  history, and hashes of geometry, basis and settings.
- Written every N iterations to a temporary file and moved into place with an
  atomic rename, so a crash mid-write never corrupts the previous checkpoint.
- `--restart file.npz`: if geometry and basis hashes match, resume exactly.
  If only the geometry differs and the basis dimension is unchanged (next scan
  or optimisation point), downgrade to "use as initial density" with a warning.
- `scan` and `opt` reuse the previous point's density automatically.

## Consequences

`scf.checkpoint_every` becomes a real option; `docs/hpc.md` is updated.
