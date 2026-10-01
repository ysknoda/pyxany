# Changelog

## Unreleased

Findings from a systematic code review (`/code-check`) of the packaged code,
including a quantum-chemistry review of the electronic-structure core.

### Fixed

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
  is now rejected with an explanation and `docs/hpc.md` corrected.
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

### Added

- `python -m dvxa` as an entry point, for clusters where the console script's
  directory is off `PATH`.
- Data-table tests are self-contained again (they compared against the deleted
  `dv-xa.py` and had been skipping silently): physical invariants plus pinned
  checksums, 3 → 33 live tests.
- Pinned total-energy regression values, a convergence-**order** test for the
  Poisson solver, and an analytic-vs-finite-difference force check on an
  asymmetric C1 geometry (H₂ alone cannot expose per-atom or per-component
  errors).

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
