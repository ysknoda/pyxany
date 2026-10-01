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

## Consequences

Requires fractional, user-fixed occupations in the SCF that override aufbau
and smearing.
