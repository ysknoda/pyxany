# Changelog

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
