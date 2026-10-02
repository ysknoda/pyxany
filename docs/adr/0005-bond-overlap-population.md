# 0005. Bond overlap population and its plots

- Status: Accepted
- Date: 2026-10-01

## Context

Mulliken bond overlap population is a standard DV-Xalpha output for discussing
chemical bonding. The code has Mayer and Wiberg bond orders but not overlap
populations.

## Decision

- Output (a): total overlap population per atom pair, as a matrix.
- Output (b): MO-resolved overlap population per atom pair, distinguishing
  bonding (> 0) and antibonding (< 0) contributions.
- Both written to JSON and the text report.
- Plots with matplotlib (optional extra, like the existing summary figure),
  written as PDF and PNG and checked by rendering to PNG before release:
  1. MO-resolved overlap population (COOP-like): energy on x, overlap
     population on y, bonding (+) / antibonding (-) as bars or Gaussian-broadened
     curves, with a line at the HOMO (or Fermi level);
  2. energy-level diagram with each level coloured by the sign and size of the
     overlap population of a chosen pair;
  3. partial DOS by element and by angular momentum l.
- An atom-pair heatmap of total overlap population is available as an option
  for small systems only (unreadable above ~100 atoms). Atoms ordered by input
  order; element axes by atomic number.
- A transition-state annotation figure is decided with ADR 0006's
  implementation.

## Implementation notes (2026-10-02)

- The COOP axis and the level colour scale are set from the occupied orbitals:
  Mulliken overlap populations of diffuse empty orbitals reach -7 per orbital
  (H2O/DZ O-H sigma*) and would otherwise flatten every occupied feature.
  Clipped bars are marked.
- Partial DOS uses Loewdin weights (bounded, sum to 1 per orbital), not
  Mulliken; species are ordered by nuclear charge.
- `--set` cannot reach nested keys, so `overlap_population.*` is set in YAML.
- Possible follow-up: choose the pair type that colours the level diagram
  (today: the type with the largest total overlap population).

## Consequences

The MO-resolved data scales as n_MO x n_pairs; output is restricted to
user-selected pairs or pairs above a bond-order threshold for large clusters.
