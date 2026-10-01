# 0003. Three-tier validation against real molecules

- Status: Accepted (Tier A tolerances to be confirmed by measurement)
- Date: 2026-10-01

## Context

X-alpha without correlation is less accurate in absolute terms than standard
DFT. A test that compares total energies or gaps with experiment would fail for
physical reasons, not bugs, and would hide real bugs behind a loose tolerance.

## Decision

**Tier A -- same functional, other code (strict; detects implementation bugs).**
PySCF with the same Gaussian basis (STO-3G, 6-31G), same geometry and
exchange `1.05 * Slater` (= alpha 0.7 / (2/3)), both on converged grids.
Proposed tolerances:

| quantity | tolerance |
|---|---|
| total energy | \|dE\| <= 1 mHa |
| occupied orbital energies | MAE <= 1 mHa |
| Mulliken charges | <= 0.01 e |
| dipole | <= 0.02 D |

Achievable agreement is measured on a few systems first; any loosening is
reported with its cause. PySCF is installed on the development machine only and
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
