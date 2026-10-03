# Changelog

## Unreleased

### Added

- **Slater transition state** (ADR 0006): `transition_state:` removes half an
  electron from a chosen orbital (`homo`, an MO number, or a core shell
  `{atom, shell}`) or moves it between two, and reports the ionisation or
  excitation energy. Checked against Janak's theorem by finite differences
  (1.4e-5 Ha) and Delta-SCF (0.03 eV). Core holes on equivalent atoms are
  localised explicitly; a symmetric start does not localise them by itself.
- **Bond overlap population and bonding figures** (ADR 0005). Atom-pair
  Mulliken overlap populations and their MO decomposition for bonded pairs
  (Mayer bond order >= 0.2, or pairs named in the input), in the text report
  and in the JSON output; a COOP-type plot per element-pair type, a level
  diagram coloured by bonding character, a Loewdin partial DOS by element and
  angular momentum, and an optional atom-pair heatmap, all as PDF and PNG.
  The sum rule (net + overlap populations = N) and the MO decomposition are
  tested.
- **SCF checkpoint and restart** (ADR 0004). The SCF state is written to
  `<prefix>_scf.npz` at the end of every SCF and every `scf.checkpoint_every`
  iterations; `--restart FILE` on `run`, `opt`, `forces` and `scan` resumes it,
  or uses it as the initial guess at another geometry. A different system is
  refused. `opt` and `scan` start each SCF from the previous point's density.
  `scf.checkpoint_every`, previously rejected as unimplemented, is now live.
- **Loewdin populations and a terminator-aware core gap.** `pyxany run` now
  reports Loewdin charges and, for pseudo-hydrogen structures, the Loewdin
  weight of the frontier orbitals on the terminators and the gap between the
  orbitals that are not terminator states (`properties.terminator_weight_max`,
  default 0.5). The passivation check used Mulliken weights, which leave [0, 1]
  with diffuse terminator functions (1.55 on the MgO6 LUMO, Loewdin 0.72).
  It shows that in the small example clusters the LUMO is substantially on the
  terminators (0.52-0.72), which relaxing them does not change; see
  docs/passivation.md.
- **Tier A cross-validation against PySCF** (`tests/test_tier_a_pyscf.py`):
  same X-alpha functional (`1.05 * LDA_X`) and the same Gaussian basis,
  translated from pyxany's own tables, on 10 closed-shell molecules, a
  6-31G(d) case and a UKS radical. Mulliken charges agree to 2e-4 e and dipoles
  to 5e-4 D; the total-energy gap is grid error and falls below 0.2 mHa on a
  fine grid (asserted). PySCF is a development-only tool, not a dependency.

- **Tier B/C validation on real molecules and clusters**
  (`tests/test_tier_b_experiment.py`, `tests/test_tier_bc_clusters.py`), with
  experimental data from NIST CCCBDB (`tools/fetch_cccbdb.py`). Tolerances are
  set from the X-alpha functional's own error, measured with PySCF at large
  basis sets; see `docs/adr/0003-validation-tiers.md`.

- `python -m dvxa` as an entry point, for clusters where the console script's
  directory is off `PATH`.
- Data-table tests are self-contained again (they compared against the deleted
  `dv-xa.py` and had been skipping silently): physical invariants plus pinned
  checksums, 3 → 33 live tests.
- Pinned total-energy regression values, a convergence-**order** test for the
  Poisson solver, and an analytic-vs-finite-difference force check on an
  asymmetric C1 geometry (H₂ alone cannot expose per-atom or per-component
  errors).

### Fixed

- **`--no-figure` had no effect**: the flag was parsed but never applied, so
  `run` wrote the figures whenever the input enabled them.
- **Geometry optimisation could report a geometry that was not its answer.**
  The returned molecule, energy and forces were those of the last evaluation,
  often a rejected line-search trial point. They are now the lowest-energy
  evaluation, i.e. scipy's solution.
- **Optimisation ignored a point that met the force criterion.** On a finite
  grid the energy and analytic forces are slightly inconsistent, so L-BFGS-B
  stalled at the energy minimum with max|F| above f_max: H2O/DZ at n_r = 40
  passed max|F| = 5.8e-6 at step 7, then ran 60 steps without converging. The
  first evaluation with max|F| < f_max now ends the optimisation (7 steps).

- **The Hartree solver was inaccurate or unstable for heavy atoms.** It divided
  ``int_0^r s^(l+2) rho`` by ``r^(l+1)`` and patched the resulting near-nucleus
  blow-up by overwriting rho_l and V_l with r^l power laws inside a radius that
  reached 0.2-0.3 Bohr at l = 10. On GaAs4 the SCF energy moved by 0.15 Ha
  between l_max 8 and 10; with off-centre Gaussian test charges E_H was wrong
  by 1.6e3 Ha at l = 8. Rewritten as a Green's-function recursion in which
  every power is a ratio <= 1, integrating the spline of r^2 rho_l: stable at
  any l, and ~3x more accurate per radial grid size (1s core test charge on Ga:
  1.9e-2 vs 5.9e-2 Ha at n_r = 30). Total energies change: +0.01 to +18 mHa
  for the small pinned molecules on the coarse test grid, and Ga/As clusters
  at n_r = 30 by ~2.5 Ha -- both solvers carry ~0.5 Ha per heavy atom of
  radial quadrature error there, with opposite signs.
- **Analytic forces were the derivative of the continuum functional, not of
  the computed energy.** They omitted the motion of the Becke weights and of
  the atom-centred grid points, and took the S and T Pulay terms from a DV-grid
  quadrature instead of the two-centre matrices the SCF uses. Both are now
  included (each checked against finite differences to 1e-9 relative). Force
  error vs finite differences: H2 ~1e-3 -> 2.5e-5 Ha/Bohr; CO 2e-2 -> 1.4e-3
  at l_max = 12. CO can now be geometry-optimised.
- **Pseudo-hydrogens carried a single-zeta basis whatever `basis.set` was.**
  They now receive the basis set's full hydrogen function list contracted by
  zeta(alpha)/zeta(1), so an alpha = 1 terminator is exactly hydrogen in every
  basis (it was only in `sz`; Si5H12/DZ inner Si charge -0.42 vs -0.03 e).

- **The radial Poisson solver treated partial sums of a Gauss-Legendre rule as
  cumulative integrals.** A Gauss rule is exact only over its whole interval, so
  the full sum was correct to ten digits while the partial sums were not
  integrals at all: 0.107 electrons out of 1.0 wrong at `n_r = 30`, converging
  only as `O(1/n_r)`. This put H₂O about 0.15 Ha (4 eV) off at the default grid.
  Replaced with a cubic-spline antiderivative, restoring `O(n_r⁻⁴)`; the error
  in `V` against `erf(√a·r)/r` falls from 5.1e−3 to 1.2e−4 at `n_r = 30`.
  **Total energies change by ~0.1 Ha** — see `docs/legacy_comparison.md`.
- Near the nucleus the same solver amplified round-off by `r^-(l+1)` and
  projection noise by `r^(1-l)`, reaching `|V| ~ 1e22` at `l = 12` and `1.4e4`
  at the default `l = 6`. The exact `r^l` asymptotics are now imposed below an
  l-dependent radius. `grid.poisson_l_max` is capped at 10 accordingly (`l = 6`
  is already converged to 2.4e−4 Ha).
- **A multiplicity inconsistent with the electron count is now rejected.** A
  radical left at the default `mult: 1` used to converge a half-occupied state
  that is not any real spin state — CH₃ gave a 4.5/4.5 α/β split and an energy
  26 mHa (0.71 eV) above the true doublet, with no warning.
- `fermi_occupations` silently placed fewer electrons than requested when the
  basis was too small; it now raises, as the aufbau path already did.
- With smearing on, the SCF converged on `E_total` rather than the variational
  free energy `E_total − σS`. It now converges on the free energy.
- **`scf.checkpoint_every` was accepted and documented but implemented
  nowhere**, so a long HPC job promised restart protection it did not have. It
  was first rejected with an explanation, and is now implemented (see Added).
- **The package could not be built.** `packages` and a `force-include` both
  mapped the data tables to the same wheel path and hatchling aborted, so
  `pip install .` failed; only editable installs worked, which is what CI used.
  CI now builds and installs a real wheel.
- **Cr's tabulated occupation was Fe's**, making the neutral atom 25 electrons
  on a Z = 24 nucleus. Corrected to the [Ar] 3d⁵ 4s¹ half-filled-shell anomaly.
- The inward Numerov seed inverted the `sqrt(r)` ratio from its own asymptotic
  (eigenvalue impact ≤ 6e−8).
- `dvxa gridcheck` and `dvxa passivate` had no execution-time measurement;
  `passivate` also wrote output from every MPI rank via bare `print()`.

### Performance

- **Every grid is set in the input file** (ADR 0009): `grid.pair`,
  `grid.atom`, `grid.guess`, `grid.one_centre` and `grid.n_r_by_element`
  join `n_r` / `n_a`; omitted values keep the previous behaviour. The old
  `grid.pair_n_r`, `grid.pair_n_a` and `basis.nao_n_r` are errors that name
  their replacement. `--set grid.pair.n_r=60` reaches nested keys.
  `--dry-run`, the report and the JSON (`"grids"`) list every grid.
- **Angular pruning, on by default**: shells inside 0.25 / 0.5 of the Bragg
  radius use n_a/3 and 2n_a/3; 67 % of the points, adding < 10 % to the
  grid's own angular error on CO, H2O, Si5H12 and GaAs4. Tier C re-pinned
  (changes 1e-6 to 9e-5 Ha).
- **Block-sparse basis** (`grid.basis_cutoff`, default 1e-6): per block of
  points only the atoms whose functions reach 1e-6 there; Si47H60 changes by
  1.3e-6 Ha and 2e-7 Ha/Bohr. 0 restores the dense path.
- **J and V_xc built as one matrix** per spin channel (only the sum enters
  the Fock matrix), and a single-call kernel for all Slater functions (chi
  evaluation 2x faster). Si47H60 SCF + forces: 1875 s -> 1246 s with pruning
  and the fused matrix alone.

All of the following reproduce the previous numbers (numba vs numpy to
1e-12, chunked vs single-pass forces to 1e-12, analytic vs finite-difference
grad V_H to < 1e-8 Ha/Bohr in the forces); each has a test.

- **Compiled kernels** (numba, threaded) for the Becke cell weights, their
  gradient, the Hartree multipole summation and its exact gradient, and the
  basis-function gradients. On a 107-atom Si47H60 cluster (1.07 M points):
  grid build 540 s -> 9 s, one Hartree evaluation 366 s -> 21 s, one S/T pair
  52 ms -> 22 ms. The force term grad V_H is now one gradient evaluation
  instead of six Hartree evaluations by finite differences.
- **Exact two-centre S/T gradient in the forces**: the pair quadrature is
  differentiated directly (basis Hessians, moving sub-grid, Becke weights)
  instead of by six-point finite differences; 622 s -> 129 s on Si47H60,
  forces unchanged to 5e-10 Ha/Bohr. Whole SCF + forces: 2381 s -> 1875 s.
- **Forces walk the grid in chunks** within `grid.memory_budget_mb`; the
  107-atom cluster no longer needs 24 GB for the basis gradients.
- **Superposition-of-atomic-densities initial guess.** The previous guess
  started Si47H60 2700 Ha above its energy and diverged; it now converges in
  18 iterations. First-iteration error: H2O 20 -> 0.19 Ha, Si5H12 264 -> 3.9
  Ha. Converged energies are unchanged.

### Changed

- **A negative Mulliken overlap population on a bonded pair now carries a
  note.** The Si-Si total is +0.51 (SZ) and -0.08 (DZ) in Si5H12 at the same
  geometry and -0.88 in Si47H60 (DZ), with Mayer bond orders of 0.8-0.9: a
  basis-set artefact, not antibonding (ADR 0005).
- **Radial grid size depends on the element.** `grid.n_r` now means the
  second-row size; each atom gets `ceil(n_r × ratio)`, the ratio measured for
  all 86 elements by `tools/radial_grid_sizes.py` so that every element
  reaches the ionisation-energy accuracy of C-F (`data/tables/radial_points.json`).
  At n_r = 30: H 36, Si 33, Ga/As 47, Cd 60. `grid.radial_by_element: false`
  restores a uniform grid. Total energies change (Tier C re-pinned).
- **`grid.poisson_l_max` defaults to `auto` = n_a - 1** (11 on the default
  grid; was 6) and may not exceed n_a - 1. The force error is set by this
  truncation (CO: 1.6e-2 Ha/Bohr at 6, 1.3e-3 at 12). The Hartree step costs
  ~(l+1)^2; the compiled summation (see Performance) more than recovers the
  cost (one GaAs4 evaluation 26 s -> 2.4 s).

- **Renamed from `dvxa` to `pyxany`** (distribution, import name, CLI command,
  `python -m` entry and the `PYXANY_DISABLE_NUMBA` environment variable) and
  moved from the `src/` layout to a flat layout with the package at the
  repository root. No `dvxa` compatibility alias is provided. Entries below this
  one refer to the old name. See `docs/adr/0001-flat-layout-and-rename.md`.
- Optimisation trajectories are written to `output.dir` like every other output
  file, instead of the working directory.

Findings from a systematic code review (`/code-check`) of the packaged code,
including a quantum-chemistry review of the electronic-structure core.

### Known limitations

- Only one initial spin guess is tried, so an open-shell or transition-metal
  result should not be trusted without checking alternatives by hand.
- The DV grid and the overlap matrix use different quadratures and so disagree
  slightly on the electron count; a warning now fires when that exceeds 1e−4
  relative.

## 0.1.0

First release as a package. Extracted from a single 6,994-line `dv-xa.py`
script, with the defects found during that extraction fixed.

### Added

- `dvxa` command line: `run`, `opt`, `forces`, `scan`, `passivate`,
  `gridcheck`, `validate`, `init`, `basis`, `info`; `--set` overrides,
  `--dry-run` resource estimates, `--json` output, shell completion, and exit
  codes for batch scripts.
- Single-YAML settings file with strict validation (unknown keys are errors)
  plus a separate `.xyz` structure file.
- **Pseudo-hydrogen passivation** for semiconductor cluster models, using the
  electron-counting rule of Phys. Rev. B **85**, 195328 (2012). Handles
  arbitrary coordination numbers, derives each terminator's basis function from
  its own fractional charge, and reports the electron balance.
- **Analytic forces** (Hellmann-Feynman + Pulay), validated against finite
  differences, and L-BFGS-B geometry optimisation with atom freezing.
- Becke multi-centre Poisson solver for the Hartree potential, replacing a
  pair-sum with an ad-hoc softening length.
- Hierarchical wall-clock timing for every stage, reported and saved as JSON.
- 186 tests covering conservation laws, closed-form integrals, backend
  agreement and MPI equivalence.

### Fixed

Each of these changed every number the original produced; see
[docs/legacy_comparison.md](docs/legacy_comparison.md) for the measurements.

- Spurious `sin θ` in the grid weights (scaled every integral by π/4).
- Missing Becke partitioning (overcounted overlapping regions by roughly the
  atom count — a factor of ~2 for water, ~100 for a 200-atom cluster).
- Slater Xα coefficient a factor of two too large in both the energy density
  and the potential.
- d-orbital angular factors halved in the numba kernels, so results depended on
  whether an optional package was installed; `l ≥ 3` was silently spherical.
- Open-shell occupations doubled the electron count of every UKS calculation.
- GTO contraction coefficients read with the wrong normalisation convention.
- `build_J` never reduced across MPI ranks.
- 6-31G(d) polarisation used a leaked loop variable.
- SCF built J and V_xc from different density matrices.
- Crash for a single atom in a minimal basis.
- Module-level `warnings.filterwarnings("ignore")`.

### Changed

- Kinetic energy uses the exact analytic Laplacian rather than a finite
  difference with a hand-tuned step.
- Two-centre integrals use a Becke-partitioned two-centre grid, matching the
  closed-form H₂ overlap to 1e-9.
- Atomic orbital solver rewritten on a logarithmic grid with Numerov shooting;
  reproduces the hydrogenic spectrum exactly.
- DIIS no longer damps the density on top of its own extrapolation (H₂ went
  from not converging in 40 iterations to converging in 12).
