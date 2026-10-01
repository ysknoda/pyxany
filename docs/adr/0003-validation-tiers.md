# 0003. Three-tier validation against real molecules

- Status: Accepted
- Date: 2026-10-01

## Context

X-alpha without correlation is less accurate in absolute terms than standard
DFT. A test that compares total energies or gaps with experiment would fail for
physical reasons, not bugs, and would hide real bugs behind a loose tolerance.

## Decision

**Tier A -- same functional, other code (strict; detects implementation bugs).**
PySCF with the same Gaussian basis (STO-3G, 6-31G), same geometry and
exchange `1.05 * Slater` (= alpha 0.7 / (2/3)), both on converged grids.
Tolerances, set from measurement on 2026-10-01 (PySCF 2.14.0, grid level 9;
pyxany test grid n_r=50, n_a=18, poisson_l_max=8, pair 60/20), on 10 closed-shell
molecules, one 6-31G(d) case and one UKS radical (CH3):

| quantity | worst measured | tolerance |
|---|---|---|
| total energy | 2.45 mHa (CO) | \|dE\| <= 3 mHa |
| occupied orbital energies | MAE 0.33 mHa (H2S) | MAE <= 0.5 mHa |
| Mulliken charges | 3e-4 e | <= 2e-3 e |
| dipole | 6e-4 D | <= 5e-3 D |

The energy gap is quadrature error on the pyxany side and is asserted to
shrink monotonically with the grid, reaching <= 0.5 mHa at n_r=90, n_a=30
(CO -2.45 / -0.45 / -0.30 mHa; H2S -2.14 / -0.93 / -0.34 mHa at n_r = 50 / 70 / 90).
The originally proposed 1 mHa energy tolerance is therefore met only on a finer
grid than the test grid; 3 mHa at the test grid keeps the run time to ~5 min.

PySCF is installed on the development machine only and
is **not** listed in `pyproject.toml`; these tests skip when it is absent.

**Tier B -- experiment (loose; physical sanity).** Experimental geometries and
values from NIST CCCBDB, fetched rate-limited and cached on disk.
Pass/fail uses only quantities X-alpha should get right qualitatively:

- equilibrium bond lengths within +-3 % (UNCERTAIN, ~70 % confidence; to be
  backed by a literature value for alpha = 0.7 before finalising);
- HOMO-LUMO gap **ordering** within a series, not absolute values;
- sign and ordering of Mulliken charges;
- dipole magnitude order and direction (CO excluded: its sign is delicate).

Not used for pass/fail: total energies, absolute gaps, Koopmans-like orbital
ionisation energies.

**Tier C -- regression.** Current values pinned.

### Molecule set (ordered by atomic number of heaviest element)

- Small molecules (Tier A/B, run in CI): H2, LiH, CH4, NH3, H2O, HF, N2, CO,
  SiH4, H2S.
- Clusters (Tier B/C, `slow`): C6H6, Si5H12, Ga4AsH, Cd4Se4H, MgO6H. The
  cluster series checks the experimental gap trend Si < GaAs < CdSe < MgO and
  that passivation removes in-gap states.

## Consequences

Tier A becomes the safety net for all later feature work (ADR 0007), so it is
built before any new feature.
