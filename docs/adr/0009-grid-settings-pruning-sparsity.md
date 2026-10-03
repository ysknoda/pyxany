# 0009. Every grid in the input file; angular pruning; block-sparse basis

- Status: Accepted
- Date: 2026-10-03
- Decisions: Q27-Q34

## Context

After step 8 (ADR 0007) two-thirds of a 107-atom SCF + forces run is the SCF
itself, dominated by the density and potential-matrix products over the
molecular grid. Only options that change the quadrature remain (option C of
Q26). Separately, several grids were fixed in the code and could not be
converged by a user.

## Decision

### All grids are set in `grid:` (Q27-Q29, Q33)

| key | grid | default |
|---|---|---|
| `n_r`, `n_a` | molecular DV grid (n_r for the second row, scaled per element) | 30, 12 |
| `n_r_by_element` | per-element radial size, overrides the ratio table | `{}` |
| `pruning` | inner-shell angular reduction (below) | on, `auto` |
| `basis_cutoff` | block-sparse basis (below) | 1e-6 |
| `angular` | angular rule: `product` or `lebedev` | product |
| `pair: {n_r, n_a}` | two-centre S/T grids (was `pair_n_r`, `pair_n_a`) | 40, 14 |
| `atom: {n_points}` | atom solver log grid: NAO basis, SAD, pseudo-H (was `basis.nao_n_r`) | 2000 |
| `guess: {n_r, n_a}` | one-atom grid of the SAD projection | 80, 14 |
| `one_centre: {n_r}` | radial quadrature of one-centre non-Slater integrals | 400 |

Every block is optional and may be partial; omitted values keep the defaults,
which reproduce the previous behaviour of the fixed grids. The old keys are
errors that name their replacement. `--set grid.pair.n_r=60` reaches nested
keys. `--dry-run`, the SCF header (`-v`), the text report and the JSON
(`"grids"`) list every grid with its actual sizes.

### Angular pruning (Q30, Q31)

A radial shell at `r < b * R_Bragg` of its atom uses the smaller angular
product grid of the innermost region containing it. `regions: auto` is
`[[0.25, n_a // 3], [0.5, 2 n_a // 3]]` (minimum 4). The Hartree projection of
a pruned shell keeps multipoles up to `l = n_a' - 1` only; those shells see an
almost spherical density.

Bounds chosen by measurement, because the Treutler-Ahlrichs (1995) values
could not be checked against the paper. Criterion: the change from pruning
must stay below 10 % of the grid's own angular error (n_a 12 vs 16, unpruned),
in both the total energy and the largest force component. At n_a = 12, n_r = 30:

| system | grid error dE / dF | pruning dE / dF (adopted scheme) | points kept |
|---|---|---|---|
| CO | 5.0e-4 / 5.8e-3 | 3.2e-7 / 1.3e-4 | 0.67 |
| H2O | 9.2e-6 / 7.9e-5 | 1.9e-8 / 7.0e-6 | 0.67 |
| Si5H12 | 1.3e-3 / 6.7e-4 | 1.1e-6 / 1.8e-5 | 0.67 |
| GaAs4 (passivated) | 7.4e-3 / 5.7e-3 | 1.8e-5 / 2.0e-5 | 0.67 |

(Ha, Ha/Bohr.) A three-region scheme `[[0.25,4],[0.5,6],[1.0,9]]` keeps 0.61
of the points but reaches 11 % on CO forces; pruning out to `R` (`[[1.0,6]]`)
changes CO forces by 9e-3, more than the grid error itself.

### Block-sparse basis (Q32)

The local grid is bisected at the median of its longest axis into blocks of
<= 2048 points. Per block, an atom's functions are kept only if the atom is
closer to the block's bounding box than its reach: the largest distance at
which any of its functions reaches `basis_cutoff` (sampled on a lone atom).
Density and potential matrices are accumulated block by block over the kept
functions; the blocks are cached when they fit a quarter of
`memory_budget_mb`.

Slater functions reach far: at 1e-10, Si (DZ) 31 Bohr and the Si
pseudo-hydrogen 22 Bohr, against a 26 Bohr diameter for Si47H60. Kept
fraction of chi (and of the n_active^2 product cost):

| cluster | 1e-10 | 1e-8 | 1e-6 | 1e-4 |
|---|---|---|---|---|
| Si47H60 (107 atoms) | 0.99 (0.98) | 0.96 (0.93) | 0.88 (0.79) | 0.59 (0.37) |
| Si87H76 (163 atoms) | 0.98 (0.95) | 0.93 (0.86) | 0.74 (0.56) | 0.42 (0.20) |

Accuracy on Si47H60 (SCF + forces, against `basis_cutoff: 0`):

| cutoff | dE (Ha) | max dF (Ha/Bohr) | d gap (eV) | d N_electrons |
|---|---|---|---|---|
| 1e-6 | 1.3e-6 | 2.3e-7 | 2.6e-6 | 4e-6 |
| 1e-4 | 3.5e-3 | 8.2e-5 | 4.4e-4 | 1.3e-3 |

The grid's own error is ~1e-3 Ha (n_a 12 vs 16) and pruning moves Si47H60 by
8e-6 Ha. **Default 1e-6** (Q35, revised from the 1e-10 first agreed, which
saves only 2-5 % at these sizes): its error is ~1/1000 of the grid error and
it removes ~20 % of the matrix-product work at 107 atoms, ~45 % at 163.
1e-4 shifts the electron count by 1e-3 and is not a safe default. 1e-10 is
the conservative choice; 0 restores the dense path exactly.

A single numba kernel now evaluates all Slater functions of a block in one
call (per-atom calls cost one thread launch each); this also halves chi
evaluation on the dense path.

### Lebedev grids (Q34, then Q36-Q38)

`grid.angular: lebedev` replaces the n_a x 2n_a product rule on every shell
(pruned ones included) by the Lebedev rule of the same polynomial degree,
order >= 2 n_a - 1 (Lebedev & Laikov 1999, from `scipy.integrate.lebedev_rule`;
scipy >= 1.15). It resolves the same multipoles (l <= n_a - 1), so pruning and
`poisson_l_max` are unchanged. n_a <= 66.

Measured against product n_a = 20 unpruned, pruning on (|dE| Ha / max|dF| Ha/Bohr):

| system | product n_a=12 | Lebedev n_a=12 (67 % pts) | Lebedev n_a=14 (~92 % pts) |
|---|---|---|---|
| CO | 6.3e-4 / 6.9e-3 | 4.7e-5 / 1.1e-3 | 3.3e-4 / 2.3e-3 |
| H2O | 1.0e-5 / 6.7e-5 | 7.3e-6 / 1.9e-4 | 8.9e-6 / 7.0e-5 |
| Si5H12 | 1.2e-3 / 6.4e-4 | 1.7e-3 / 8.8e-4 | 4.6e-4 / 4.5e-4 |
| GaAs4 | 8.2e-3 / 6.8e-3 | 1.7e-2 / 9.2e-3 | 9.8e-3 / 3.2e-3 |

At the same degree Lebedev saves a third of the points but is not uniformly
as accurate (better on CO, worse on the heavier clusters); at a near-equal
point count it gives equal or smaller force errors in all four, with the
GaAs4 energy slightly worse. Decision (Q38, after these measurements): the
default stays `product`; Lebedev is an option.

## Consequences

- Total energies change with pruning: Tier C values are re-pinned; Tier A and
  B keep their tolerances.
- The J and V_xc matrices are now built as one matrix per spin channel (only
  their sum enters the Fock matrix), halving the potential-matrix work of a
  closed-shell iteration without changing results.

## Measured end to end

Si47H60 (107 atoms, 966 basis functions), SCF (18 iterations) + forces, one
node, nothing else running:

| version | total | per SCF iteration |
|---|---|---|
| step 8 (ADR 0007) | 1875 s | 70 s |
| + pruning, fused J + V_xc | 1246 s | 43 s |
| + basis_cutoff 1e-6, single-call chi kernel (defaults) | 1189 s | 40 s |

Forces agree with the unpruned dense run to 1e-5 Ha/Bohr; peak RSS 6.9 GB.

