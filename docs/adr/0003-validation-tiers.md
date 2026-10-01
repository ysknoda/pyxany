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
pyxany test grid n_r=50, n_a=18, poisson_l_max=8, pair 60/20; re-measured after
the ratio-damped Poisson solver), on 10 closed-shell molecules, one 6-31G(d)
case and one UKS radical (CH3):

| quantity | worst measured | tolerance |
|---|---|---|
| total energy | 3.03 mHa (H2S) | \|dE\| <= 4 mHa |
| occupied orbital energies | MAE 0.26 mHa (CO) | MAE <= 0.5 mHa |
| Mulliken charges | 2e-4 e | <= 2e-3 e |
| dipole | 5e-4 D | <= 5e-3 D |

The energy gap is quadrature error on the pyxany side and vanishes with the
grid (CO -1.16 / -0.12 / -0.18, SiH4 +2.61 / +0.34 / +0.08, H2S +3.03 / +0.42 /
+0.15 mHa at n_r = 50 / 70 / 90); the finest grid must agree to 0.5 mHa.

PySCF is installed on the development machine only and
is **not** listed in `pyproject.toml`; these tests skip when it is absent.

**Tier B -- experiment (loose; physical sanity).** Experimental geometries and
values from NIST CCCBDB (`tools/fetch_cccbdb.py`, rate-limited, raw pages
cached; parsed values with primary references in
`tests/data/cccbdb_experimental.json`). Tolerances come from the functional's
own measured error, obtained with PySCF at a large basis so that basis error is
removed, plus the measured STO basis error -- not from guesses:

- **bond lengths** (geometry optimised, DZP): intrinsic X-alpha error -0.2 %
  (N2) to +4.0 % (H2), X-H bonds +1.2 to +1.8 % (cc-pVTZ); pyxany DZP measured
  -0.3 % to +4.2 %. Tolerance **5 %**. LiH excluded (STO Li basis: +5.8 %
  against intrinsic +1.8 %). The originally proposed 3 % would have failed H2
  on the functional alone.
- **dipoles** (experimental geometry, DZP): intrinsic X-alpha / aug-cc-pVTZ
  0.94-1.07 x experiment; pyxany DZP 0.74-1.07 (no diffuse functions).
  Tolerance: **ratio in [0.65, 1.35] and correct direction** (LiH, NH3, H2O,
  HF, H2S); CO excluded (0.11 D, small difference of large terms).
- **Mulliken charges**: signs (LiH, NH3, H2O, HF) and the ordering
  q_H(NH3) < q_H(H2O) < q_H(HF). Excluded: CH4 (C at -1.1 e, Mulliken
  artefact), SiH4 (Si sign flips between DZ and DZP), H2S.
- **clusters**: anion charge ordering MgO < CdSe < GaAs < 0 (electronegativity
  difference 2.13 > 0.86 > 0.37), and pseudo-hydrogen opening the gap that
  plain hydrogen closes in GaAs4 (0.07 -> 5.87 eV), the effect of Deng et al.
  (2012).

**Withdrawn:** the HOMO-LUMO gap ordering Si < GaAs < CdSe < MgO. Measured
6.05 / 5.87 / 3.34 / 5.98 eV: these 5-7 heavy-atom clusters are dominated by
quantum confinement and differ in size, so there is no physical reason for them
to follow the bulk order.

Not used for pass/fail: total energies, absolute gaps, orbital ionisation
energies.

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
