# 0015. Default radial grid n_r = 45

- Status: Accepted
- Date: 2026-11-14
- Decisions: Q86 (geometry optimization is a routine use)

## Context

The grid's force error is set by its radial quadrature (docs/forces.md).
H2O (DZ) against an (n_r 80, n_a 26) reference, measured 2026-11-14:

| n_r, n_a | abs(dE) (Ha) | max abs(dF) (Ha/Bohr) |
|---|---|---|
| 30, 12 | 2.0e-3 | 2.2e-3 |
| 45, 12 | 3.4e-4 | 1.7e-4 |
| 60, 12 | 8.7e-5 | 2.3e-5 |
| 30, 18 | 2.0e-3 | 2.1e-3 |

At the old default n_r 30 the force error exceeds the default
`optimize.f_max` of 1e-3 Ha/Bohr, so an optimization at the defaults could stop
on, or chase, grid noise. Options were: raise n_r to 45; keep 30 and tell
users to refine for optimizations; loosen f_max. The user expects geometry
optimization to be routine, so the default is raised.

## Decision

`grid.n_r` defaults to **45** (radial points for C-F; per element
ceil(45 x ratio): H 54, C 38, O 44, Si 50, Ga/As 71, Cd 90, Pb 107). The
force error at the default grid, 1.7e-4 Ha/Bohr for H2O, is a factor six
below `optimize.f_max`. `pyxany init`, `passivate --emit-yaml`, the shipped
examples, README and docs/input_format.md use 45. Tests that set their own
grid are unchanged. `forces --compare` keeps estimating the grid error from
n_r + 15 and warns when f_max is below it.

## Consequences

Tier C examples, n_r 30 -> 45 (SCF + forces, one machine, 2026-11-14):

| example | dE (Ha) | d gap (eV) | time |
|---|---|---|---|
| si5h12 | -0.093 | -0.0004 | 29 s -> 33 s |
| ga4ash | -0.129 | +0.0013 | 49 s -> 57 s |
| cdse4h | -0.144 | +0.0024 | 51 s -> 58 s |
| mgo6h | +0.003 | +0.0204 | 101 s -> 133 s |

- Absolute total energies move by ~0.1 Ha for the heavy-atom clusters
  (radial quadrature error of Ga, As, Cd, Se at n_r 30); results computed at
  the old default are not comparable with new ones at the 1e-1 Ha level. Gaps
  move by at most 0.02 eV.
- Cost: +14 to +32 % here, less than the 1.5x point count, since grid work
  is only part of the run. The ADR 0007/0009 timings were measured at n_r 30.
- Tier C re-pinned; the n_r 30 values were first reproduced to 3e-8 Ha, so
  the P0 fixes of ADR 0012 did not move them.
