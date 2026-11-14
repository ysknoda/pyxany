# Changelog

## Unreleased

### Fixed

- **NAO basis results were wrong** (ADR 0012, F10, Critical): the radial
  splines were extrapolated with a constant tail to infinity, so one-center
  S/T integrals were wrong (Si 3p self-overlap 4.09) and NAO SCFs converged to
  nonsense (SiH4 -38.40 Ha against -290.17 Ha now, DZ -289.47). Every NAO
  result since 9eddee4 must be repeated. The splines are now zero beyond
  their grid (|R| < 3e-8 there); `tests/test_nao.py` runs NAO SCFs.
- **UKS used a separate Pulay DIIS per spin channel** (F11): O2 triplet 82
  iterations, H2O O1s transition state 37, N2 1s transition state 321 to a
  commutator of 1e-6; with one DIIS over both channels 15 / 14 / 20, final
  energies equal to 1e-8 Ha. This was the slow mode of ADR 0014.
- `--set A.B.C=v` dropped `.C` when `A.B` was not a mapping, so
  `--set scf.alpha.O=0.73` changed alpha for every element (F13). A uniform
  alpha is now promoted to `{default: old, O: 0.73}`; any other case is an
  error. `--set` accepts dataclass fields only, and values of new mapping
  keys are read as YAML.
- Under MPI, an error on rank 0 alone (output directory, result files,
  checkpoint) left the other ranks waiting until wall time; every rank is now
  ended with `MPI_Abort` (F14).
- UKS bond orders in the report, the JSON and the summary figure used the
  closed-shell formula (O2 triplet 1.13 instead of 1.62; F15). The table is
  now titled Mayer, and the JSON has `mayer_bond_orders` (the old
  `wiberg_bond_orders` key is kept with the same values).
- `pyxany embed` refuses a cell whose Evjen block has a dipole (wurtzite: a
  spurious ~1.7 V/A field on the cluster), unless `--allow-dipole` (F16).
- `passivate --reference` matched existing bonds by exact equality of
  rounded vectors and, with 0.001 A of noise, put terminators 0.85 A from
  other Si atoms (F17). Bonds now match within 30 deg; a terminator closer
  than 0.8 x the covalent-radius sum to another atom is warned about; the
  reference neighbor list is built once (17.7 s -> fast for 512 atoms).
- S/T pair screening dropped NAO pairs that overlap (Si-Si at 5.45 A, |S| up
  to 0.07); pairs are now screened on the sampled reach of each atom (F12).
- `parallel.mpi_version()` always printed an empty mpi4py version.

- **EDIIS ignored the energies**: it solved for the stationary point of the
  error matrix alone (two vectors always gave the midpoint) and clipped
  negative weights. It now minimizes the Kudin-Scuseria-Cances interpolated
  energy over the simplex, with the coefficient 1/4 that F = dE/dP requires;
  UKS uses one coefficient set for both spins. Converged energies unchanged
  (H2O, CO, O2, CH3, NH2, H2O O1s TS to 1e-8 Ha).
- The UKS initial guess put 2S surplus electrons into P_alpha (O2 triplet: 18
  for 16); it now splits P/2 +- a trace-one spin density.
- A transition-state restart rescaled the hole channel to the integer count
  (H2O HOMO TS: 2.26 Ha off, 122 more iterations; now 6 to the same energy);
  `--restart` from a run that stopped at max_iter is an input error.
- With smearing, HOMO/LUMO are the levels straddling half filling (a barely
  populated near-degenerate partner was taken as the HOMO).
- UKS diagonalization honours `scf.s_threshold` (F9); MPI broadcasts P_alpha
  and P_beta; the checkpoint fingerprint hashes alpha values and NAO/GTO
  parameters; an excitation into a full orbital is refused in every branch.
- Input errors exit 1 (XYZError and the input checks of occupations, TS,
  checkpoint, passivation, scan, freeze and basis keys are now ConfigError;
  config fields are type-checked). `--set` values are read as YAML; derived
  defaults follow overrides; `overlap_population.pairs` and numeric settings
  are validated; `embedding.units` takes the `molecule.units` aliases.
- `scan`/`opt` exit 2 when an SCF did not converge (and say how many);
  `scan` bends linear molecules, rejects angles outside (0, 180), writes the
  checkpoint and reports the minimum of converged points only.
- `gridcheck` sums the electron count over MPI ranks; `passivate --role
  EL=anion|cation`; `--emit-yaml` writes a structure path relative to the
  YAML; `embed` checks the element of each removed site
  (`--allow-substitution`); `pyxany basis KEY` reports that key.
- `--log` records the error messages and is opened on rank 0 only (F4); `-q`
  suppresses the timing table, so `-q --json` is valid JSON; result JSON is
  strict (null for non-finite, integer `iter`, electron counts in
  `electron_counts`).
- The UKS summary figure draws both spin channels and a +/- DOS (F5).

- **Spin populations** (UKS): Mulliken and Loewdin n_alpha - n_beta per
  atom in the report and in the JSON (`population.spin_populations`); CH3 puts
  +1.26 (Mulliken) on C and a small negative spin on each H.
- `pyxany embed` writes the charge file on rank 0 only and prints the
  `embedding.charges` path relative to the input file as well as absolute.
- `pseudo_h.describe` takes the atom-solver grid size.

### Performance

- `compute_forces` reuses alpha(r) and the full density in the grid-motion
  term instead of recomputing them.
- The element-specific alpha field and its force term are compiled (numba):
  27 atoms / 50k points 1.6 s -> 0.08 s and 12.7 s -> 0.27 s, numpy path
  unchanged and equal to 5e-14.

### Changed

- **Default `grid.n_r` is 45** (was 30; ADR 0015): at 30 the H2O force error,
  2.2e-3 Ha/Bohr, exceeded `optimize.f_max` (1e-3); at 45 it is 1.7e-4. Total
  energies of the heavy-atom examples move by ~0.1 Ha (gaps <= 0.02 eV), runs
  take 14-32 % longer; examples, templates and Tier C pins updated.
- `pyxany forces --compare` also prints a grid-error estimate (forces at
  n_r + 15) and warns when `optimize.f_max` is below it; docs/forces.md
  separates the implementation check (analytic vs FD, 1.3e-4 Ha/Bohr for H2O)
  from the grid error (2.2e-3 at n_r 30, 1.7e-4 at 45, 2.3e-5 at 60).

- **Configuration keys that did nothing** (ADR 0013): `runtime.numba`
  (auto/on/off) and `--numba`/`--no-numba` now choose the backend (both had
  no effect); `runtime.verbosity` is the default when neither `-v` nor `-q`
  is given. `runtime.mpi`, `passivation.scheme`, `passivation.basis` and
  `properties.bde` are removed and are errors that say why;
  `properties.esp: {enabled: true}` is an error until it is implemented.
  The four passivated examples and `passivate --emit-yaml` no longer write
  the removed keys.

- CI: a manual run takes `suite=all` or `suite=mpi-only`; the MPI test shows
  mpirun's stderr when it fails; the runner's MPI job allows oversubscription
  (Open MPI 5 refused 2 ranks with prte-rmaps-base:alloc-error).

## 0.2.0a2 (2026-11-13)

### Fixed

- **The SCF could report convergence without being self-consistent.** The
  test looked only at the energy change and the density change; a stalled
  Pulay DIIS can hold the density still while F and P do not commute. A core
  hole on one of two equivalent atoms (the N2 1s transition state) converges
  along a slow mode and was stopped this way: at the default tolerances after
  30 iterations with -eps 65 meV too large, and the slow test passed on macOS
  (10 meV off) while failing on Linux CI, where the stall did not happen
  (ADR 0014). The rms of the commutator F P S - S P F, projected onto the
  space left after dropping near-linearly-dependent directions, must now also
  fall below the new `scf.c_tol` (default 1e-5); the H2O, CO, N2 and CH3
  checks met it already when they stopped and take no extra iterations.
  Transition-state runs on equivalent atoms take ~200 iterations and should
  be repeated if they mattered.

### Changed

- `scf.max_iter` defaults to 300 for a transition-state run (100 otherwise)
  when the input does not set it: the equivalent-atom core hole needs ~200.
- A run whose energy and density have settled but whose commutator has not is
  diagnosed as a slow mode, with the advice to raise `scf.max_iter`.
- CI: the slow tests + MPI job runs only when the workflow is started by hand.
- `tools/debug_ts_convergence.py`: per-iteration trace of the N2 core-hole SCF.

## 0.2.0a1 (2026-11-13)

First release published on GitHub (`ysknoda/pyxany`), as an alpha.

### Packaging and CI

- Version `0.2.0a1`; SPDX license declaration (`license = "MIT"`,
  `license-files`), `hatchling>=1.27`, Python 3.10-3.13 classifiers,
  project URLs under `github.com/ysknoda/pyxany`.
- **CI could not have passed**: its lint step still pointed at the `src/`
  directory removed by the flat-layout move (ruff E902), and the wheel check
  expected 12 data tables where 14 ship. It now lints `pyxany`, `tests` and
  `tools` and compares the wheel against the tables in the source tree.
- ruff is pinned to the 0.16 series in the `dev` extra, and the 68 findings
  of ruff 0.16.6 are fixed (two dead assignments, among them an unused
  `S @ C` product in `mo_overlap_population`; no numerical change).
- **`pyxany run` failed without matplotlib** (exit code 3, after the SCF had
  converged and the results were written): the guard imported
  `pyxany.plotting`, which imports matplotlib only inside its functions, so
  the documented "figures are skipped" path was never taken. Found by the
  wheel smoke test; regression test added.
- README: Features, Installation, Limitations (the code is non-relativistic)
  and Testing sections; the `embed` command; SciPy >= 1.15.
- Roadmap and the decisions behind it recorded as ADR 0012 (superseding the
  ADR 0008 backlog) and ADR 0013 (configuration keys that are validated but
  never read -- listed there, not yet changed).

### Added

- **Element-specific X-alpha** (ADR 0011): `scf.alpha` takes a mapping
  `{default: 0.7, Si: 0.73}`, `alpha_hf` (Schwarz's published table, H-Rn, 19
  atoms interpolated as the 1974 paper recommends) or `alpha_hf_polarized`, mixed
  between atoms with the Becke weights; forces include the moving-weight
  term (finite differences to the grid error). Spin-polarized atom solver
  and `tools/alpha_hf.py` for the recomputed alpha_HF table.
- `alpha_hf_polarized` now covers H-Rn (86 elements, none excluded): ROHF
  kept in its configuration by a targeted guess or by SO3 symmetry with fixed
  (l, m) occupations, a level shift, and a settled-energy rule for Sm; the
  recomputed Schwarz alpha_HF matches the 66 published values to a median of
  2.2e-5 (ADR 0011).

- **Point-charge (Madelung) embedding** (ADR 0010): `embedding.charges`
  reads fixed Gaussian point charges; `pyxany embed` builds them from a
  crystal as a neutral, dipole-free Evjen block (rock-salt Madelung constant
  to 7e-6). Energies, forces (analytic = finite differences to the grid
  error) and a warning for electrons leaking onto positive charges. Example
  `examples/mg19o6_embedded.yaml`: the bare MgO6(10-) leaks 1.6 e with DZ and
  does not converge, the cation-terminated Mg19O6 leaks 0.03 e (docs/embedding.md).

- **Slater transition state** (ADR 0006): `transition_state:` removes half an
  electron from a chosen orbital (`homo`, an MO number, or a core shell
  `{atom, shell}`) or moves it between two, and reports the ionization or
  excitation energy. Checked against Janak's theorem by finite differences
  (1.4e-5 Ha) and Delta-SCF (0.03 eV). Core holes on equivalent atoms are
  localized explicitly; a symmetric start does not localize them by itself.
- **Bond overlap population and bonding figures** (ADR 0005). Atom-pair
  Mulliken overlap populations and their MO decomposition for bonded pairs
  (Mayer bond order >= 0.2, or pairs named in the input), in the text report
  and in the JSON output; a COOP-type plot per element-pair type, a level
  diagram colored by bonding character, a Loewdin partial DOS by element and
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

- **Analytic UKS forces were wrong** since the grid-derivative terms were
  added (4f4746c): the exchange-energy density in those terms was twice its
  value (0.46 Ha/Bohr on closed-shell H2O run as UKS). Open-shell forces and
  optimizations from that period should be repeated.

- **`--no-figure` had no effect**: the flag was parsed but never applied, so
  `run` wrote the figures whenever the input enabled them.
- **Geometry optimization could report a geometry that was not its answer.**
  The returned molecule, energy and forces were those of the last evaluation,
  often a rejected line-search trial point. They are now the lowest-energy
  evaluation, i.e. scipy's solution.
- **Optimization ignored a point that met the force criterion.** On a finite
  grid the energy and analytic forces are slightly inconsistent, so L-BFGS-B
  stalled at the energy minimum with max|F| above f_max: H2O/DZ at n_r = 40
  passed max|F| = 5.8e-6 at step 7, then ran 60 steps without converging. The
  first evaluation with max|F| < f_max now ends the optimization (7 steps).

- **The Hartree solver was inaccurate or unstable for heavy atoms.** It divided
  ``int_0^r s^(l+2) rho`` by ``r^(l+1)`` and patched the resulting near-nucleus
  blow-up by overwriting rho_l and V_l with r^l power laws inside a radius that
  reached 0.2-0.3 Bohr at l = 10. On GaAs4 the SCF energy moved by 0.15 Ha
  between l_max 8 and 10; with off-center Gaussian test charges E_H was wrong
  by 1.6e3 Ha at l = 8. Rewritten as a Green's-function recursion in which
  every power is a ratio <= 1, integrating the spline of r^2 rho_l: stable at
  any l, and ~3x more accurate per radial grid size (1s core test charge on Ga:
  1.9e-2 vs 5.9e-2 Ha at n_r = 30). Total energies change: +0.01 to +18 mHa
  for the small pinned molecules on the coarse test grid, and Ga/As clusters
  at n_r = 30 by ~2.5 Ha -- both solvers carry ~0.5 Ha per heavy atom of
  radial quadrature error there, with opposite signs.
- **Analytic forces were the derivative of the continuum functional, not of
  the computed energy.** They omitted the motion of the Becke weights and of
  the atom-centerd grid points, and took the S and T Pulay terms from a DV-grid
  quadrature instead of the two-center matrices the SCF uses. Both are now
  included (each checked against finite differences to 1e-9 relative). Force
  error vs finite differences: H2 ~1e-3 -> 2.5e-5 Ha/Bohr; CO 2e-2 -> 1.4e-3
  at l_max = 12. CO can now be geometry-optimized.
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

- **Lebedev angular grids** (`grid.angular: lebedev`): the Lebedev rule of
  the same degree as the n_a x 2n_a product grid on every shell, ~2/3 of the
  points. Not uniformly as accurate at equal degree (ADR 0009 table), so the
  default stays `product`. Requires scipy >= 1.15 (`lebedev_rule`).

- **Every grid is set in the input file** (ADR 0009): `grid.pair`,
  `grid.atom`, `grid.guess`, `grid.one_center` and `grid.n_r_by_element`
  join `n_r` / `n_a`; omitted values keep the previous behavior. The old
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
  and the fused matrix alone, 1189 s with all of the above (defaults).

All of the following reproduce the previous numbers (numba vs numpy to
1e-12, chunked vs single-pass forces to 1e-12, analytic vs finite-difference
grad V_H to < 1e-8 Ha/Bohr in the forces); each has a test.

- **Compiled kernels** (numba, threaded) for the Becke cell weights, their
  gradient, the Hartree multipole summation and its exact gradient, and the
  basis-function gradients. On a 107-atom Si47H60 cluster (1.07 M points):
  grid build 540 s -> 9 s, one Hartree evaluation 366 s -> 21 s, one S/T pair
  52 ms -> 22 ms. The force term grad V_H is now one gradient evaluation
  instead of six Hartree evaluations by finite differences.
- **Exact two-center S/T gradient in the forces**: the pair quadrature is
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

- **American spelling throughout** (code, input keys, output, docs): the
  input section `optimise:` is now `optimize:`, `grid.one_centre` is
  `grid.one_center`, `scf.alpha: alpha_hf_polarised` is `alpha_hf_polarized`;
  the old names are errors that name the new ones. Modules
  `forces/optimise.py`, `integrals/onecentre.py`, `integrals/twocentre.py`
  are now `optimize.py`, `onecenter.py`, `twocenter.py`; the `opt` JSON block
  is `"optimization"`.

- **A negative Mulliken overlap population on a bonded pair now carries a
  note.** The Si-Si total is +0.51 (SZ) and -0.08 (DZ) in Si5H12 at the same
  geometry and -0.88 in Si47H60 (DZ), with Mayer bond orders of 0.8-0.9: a
  basis-set artefact, not antibonding (ADR 0005).
- **Radial grid size depends on the element.** `grid.n_r` now means the
  second-row size; each atom gets `ceil(n_r × ratio)`, the ratio measured for
  all 86 elements by `tools/radial_grid_sizes.py` so that every element
  reaches the ionization-energy accuracy of C-F (`data/tables/radial_points.json`).
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
- Optimization trajectories are written to `output.dir` like every other output
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
  differences, and L-BFGS-B geometry optimization with atom freezing.
- Becke multi-center Poisson solver for the Hartree potential, replacing a
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
- GTO contraction coefficients read with the wrong normalization convention.
- `build_J` never reduced across MPI ranks.
- 6-31G(d) polarization used a leaked loop variable.
- SCF built J and V_xc from different density matrices.
- Crash for a single atom in a minimal basis.
- Module-level `warnings.filterwarnings("ignore")`.

### Changed

- Kinetic energy uses the exact analytic Laplacian rather than a finite
  difference with a hand-tuned step.
- Two-center integrals use a Becke-partitioned two-center grid, matching the
  closed-form H₂ overlap to 1e-9.
- Atomic orbital solver rewritten on a logarithmic grid with Numerov shooting;
  reproduces the hydrogenic spectrum exactly.
- DIIS no longer damps the density on top of its own extrapolation (H₂ went
  from not converging in 40 iterations to converging in 12).
