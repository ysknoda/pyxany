# 0006. Slater transition-state method

- Status: Accepted
- Date: 2026-10-01

## Context

Ionisation and excitation energies from the Slater transition state are a
principal use of DV-Xalpha (XPS, XANES edge positions). Koopmans-like orbital
energies from X-alpha carry large self-interaction errors and are not
ionisation energies.

## Decision

In scope now:

- (a) valence ionisation: remove 0.5 electron from a chosen MO;
- (b) core ionisation (XPS): remove 0.5 electron from 1s/2p of a chosen atom,
  with the hole localised by distinguishing that one atom among equivalent
  ones;
- (c) transition energies: move 0.5 electron from MO i to MO j.

Input syntax:

```yaml
transition_state:
  # (a) valence ionisation: MO index (1-based, by energy) or 'homo'
  remove: {orbital: homo, spin: alpha, electrons: 0.5}
  # (b) core ionisation: atom index (1-based, as in the .xyz) + nl
  # remove: {atom: 3, shell: 1s, electrons: 0.5}
  # (c) excitation: from -> to
  # excite: {from: homo, to: lumo, electrons: 0.5}
```

- One transition per run; several transitions are separate jobs.
- A core hole makes the system open-shell, so the run switches to UKS
  automatically.
- Atom and MO indices are **1-based** here, matching the .xyz file as read by
  eye. Note that the existing CLI coordinate syntax (`bond:0,1`) is 0-based;
  the docs must state this difference explicitly.

Deferred: (d) XANES/XES spectra from transition-state dipole intensities
(ADR 0008).

## Implementation notes (2026-10-02)

- Validated by Janak's theorem: dE/dx by central differences in the number of
  electrons removed equals -eps(TS) to 1.4e-5 Ha (H2O HOMO, l_max = 13); the
  residual is set by the multipole truncation (5.8e-3 Ha at l_max = 2), i.e.
  by the small asymmetry of the truncated Hartree operator.
- TS vs Delta-SCF for the H2O HOMO: 11.16 vs 11.12 eV (third-order term).
  X-alpha itself underestimates this IP: PySCF X-alpha/aug-cc-pVTZ Delta-SCF
  12.04 eV against 12.62 eV measured (CCCBDB), so experiment is not a
  pass/fail criterion.
- Core holes on equivalent atoms do **not** localise by themselves from a
  symmetric start (N2 stayed 0.50/0.50 and gave 399.6 eV). The hole is first
  taken as the maximally localised combination of the element's core
  orbitals (a density correction that breaks the symmetry), then from the
  localised eigenvector's occupation once one exists (> 0.9 on the atom), so
  the density commutes with F and the SCF converges: N2 1s 408.1 eV, H2O
  O 1s 536.9 eV (DZ, non-relativistic). Measured XPS values of roughly 409.9
  and 539.9 eV (Jolly et al. 1984, quoted from memory: UNCERTAIN, ~85 %) are
  shown for orientation only.

## Consequences

Requires fractional, user-fixed occupations in the SCF that override aufbau
and smearing.
