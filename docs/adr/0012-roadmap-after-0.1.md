# 0012. Roadmap after 0.1: fixes, features and the first GitHub release

- Status: Accepted (roadmap); each item below carries its own status
- Date: 2026-11-13
- Decisions: Q65-Q81
- Supersedes: ADR 0008 (backlog); the "nothing is pushed to GitHub" rule of ADR 0007

## Context

Since 0.1.0 the forces, the Hartree solver, the grids and the X-alpha
parameter were reworked (ADRs 0007, 0009, 0011), but the last systematic code
review (`/code-check`) dates from 2026-08-27; about 80 files under `pyxany/`
have changed since. An inventory on 2026-11-13 found, besides the ADR 0008
backlog, the defects listed as F1-F6 below. State at that date: fast suite 335
passed, slow suite 48 passed, `ruff check pyxany tests tools` 68 errors.

The timing of each item is left to the maintainer; this ADR records what is to
be done, why, how large it is, what it depends on, and a recommended order.
An item moves from `Proposed` to `Accepted` when it is scheduled and to `Done`
(with the commit) when it lands.

## Intended use (Q65)

Primary: electronic structure of semiconductor and ionic cluster models
(pseudo-hydrogen passivation, point-charge embedding): gaps, frontier
character, bonding. Secondary: core-level spectroscopy built on the Slater
transition state (ADR 0006). Priorities below follow from this.

## Gate rule for review findings (Q69, Q70)

P0 re-reviews the changed files. Its findings are triaged by severity:

- **Critical / High: must be fixed before any feature item (A*) is started.**
  Each is appended to the table below as `F<n>` when found.
- Medium: added to the fix list.
- Low / Suggestion: batched.

All five Critical findings of the 2026-08-27 review are fixed and carry
regression tests (Poisson partial-sum quadrature; pseudo-hydrogen exponents in
`build_basis`, `opt` and finite-difference forces; the unbuildable wheel). Of
that review only the plotting MEDIUM (F5) and the single spin guess (A5)
remain open.

## Items

Size: S <= 1 day, M a few days, L a week or more.

### Review and fixes

| id | item | size | depends on | status |
|---|---|---|---|---|
| P0 | Re-review (`/code-check`) of every file changed since 2026-08-27, split as before between a general and a quantum-chemistry reviewer; refresh `.code-check/history.md` | M | -- | Done 2026-11-14 (findings below) |
| F1 | CI: `ruff check src tests` points at the `src/` removed by ADR 0001 (E902), and the package job expects 12 data tables while 14 ship. Lint the real paths; compare the wheel against the tables in the source tree instead of a fixed count | S | -- | Done in 0.2.0a1 |
| F2 | 68 ruff findings (ruff 0.16.6); dead code `max_occ` (`scf/driver.py`) and an unused `S @ C` product (`properties/population.py`, an n^3 matrix product per call). Pin the ruff version in the `dev` extra so a new release cannot turn CI red | S | -- | Done in 0.2.0a1 |
| F3 | Seven configuration keys are validated but read nowhere (same class as the `checkpoint_every` HIGH of 2026-08-27): `runtime.numba`, `runtime.mpi`, `runtime.verbosity`, `passivation.scheme`, `passivation.basis`, `properties.bde`, `properties.esp`; `--numba` has no effect. Design in ADR 0013 | S | -- | Done (ADR 0013; YAML on/off booleans accepted) |
| F4 | `--log`: input and runtime errors go to stderr only, so the log of a failed run is empty (measured: 0 bytes); every MPI rank opens the log with mode `w`. Route error messages into the log and open it on rank 0 only | S | -- | Done 2026-11-14 |
| F5 | The summary figure (`<prefix>_results.png`) of a UKS run draws the alpha levels only while its HOMO/LUMO span both channels (`plotting.summary_figure`); the bonding figures were already fixed | S | -- | Done 2026-11-14 |
| F6 | Documentation drift: README (SciPy floor, `embed`, transition state, embedding, `alpha_hf`, restart) -- done in 0.2.0a1; ADR 0005 still says `--set` cannot reach nested keys (fixed by ADR 0009) | S | -- | README done in 0.2.0a1; ADR 0005 note Proposed |
| F7 | Without matplotlib, `pyxany run` failed with exit code 3 after a converged SCF: the guard imported `pyxany.plotting`, which imports matplotlib only inside its functions. Found by the wheel smoke test while preparing 0.2.0a1; the CI package job (bare wheel, `pyxany run h2o.yaml`) would have failed on it. Guard on matplotlib itself; regression test | S | -- | Done in 0.2.0a1 |
| F8 | The SCF could report convergence without self-consistency (energy and density tests only); the N2 1s transition state stopped 65 meV off at the defaults and the slow test failed on Linux CI. Found by the first CI run. Design in ADR 0014 | S | -- | Done in 0.2.0a2 |
| F9 | The UKS diagonalization ignores `scf.s_threshold` (`_diagonalize_uks` calls `solve_generalized` with the default 5e-4); found with F8 | S | -- | Done 2026-11-14 |
| R0 | The code is non-relativistic, which nothing said. README statement (done in 0.2.0a1); a run-time warning, once per run, when an element with Z > 36 is present; for a core-ionization transition state an unconditional note plus, for a 1s hole, the hydrogen-like estimate of the missing correction (Zalpha)^2/4 x abs(E) (Q79) | S | -- | README done in 0.2.0a1; warnings Proposed |

R0 numbers: the Dirac 1s level of a one-electron atom is
`E = -Z^2/2 [1 + (Z alpha)^2/4 + O((Z alpha)^4)]` Ha, so the leading relative
correction is `(Z alpha)^2/4` (confidence: high). It gives ~5 eV for Si 1s and
~130 eV for Ga 1s, i.e. absolute core energies are affected even for light
elements; chemical shifts largely cancel. Z > 36 (fifth period on) as the
point where valence properties need relativity follows the usual rule of thumb
(Pyykko, Chem. Rev. 88, 563 (1988)); confidence: medium -- it is a convention,
not a measured threshold for this code.

### P0 findings (re-review 2026-11-14)

Two reviewers (quantum-chemistry: basis, grid, integrals, scf, kernels,
forces, molecule, embedding; general: cli, io, config, passivation,
properties, plotting, parallel, timing, data). 1 Critical, 9 High,
about 20 Medium, 12 Low. Items marked (v) were reproduced independently
before being recorded. Gate: F10-F18 come before any A item.

| id | sev. | item | status |
|---|---|---|---|
| F10 | Critical | (v) NAO radial splines are extrapolated with `ext=3` (`basis/types.py`, `NAO._R`): a constant tail to infinity, so one-center S/T are wrong (Si 3p S = 4.09, T = 6.6e4 Ha) and NAO SCFs converge to nonsense (SiH4 -38.40 Ha vs DZ -289.47; Si atom +31872 Ha), flagged only by the grid/overlap electron-count warning. Since 9eddee4; no NAO SCF test exists. Fix: zero (`ext=1`) or exponential tail; tests: unit self-overlap for every NAO, NAO SiH4 near DZ | Done (ext=1; tests/test_nao.py) |
| F11 | High | UKS runs two independent Pulay DIIS (one per spin) although F_a depends on P_b: O2 triplet 134 iterations (not converged at the defaults), CH3 40, H2O O1s TS 72, N2 1s TS 321; with one DIIS over the stacked error 17 / 16 / 18 / 20, energies equal to 4e-9 Ha. Very likely the cause of the ADR 0014 slow mode (A9). Re-measure A9/A10 after | Done (SpinPulayDIIS; O2 82->15, N2 1s TS 321->20 iterations, energies equal to 1e-8 Ha) |
| F12 | High | S/T pair screening bound is not conservative (`integrals/twocenter.py` `pair_is_negligible`, `NAO.exponent_scale`): NAO Si-Si pairs dropped at 10.3 Bohr with abs(S) up to 6.9e-2 (dimer energy off by 1.3e-3 Ha, non-variational); STO/GTO negligible. Screen on the sampled reach (`basis_reach`) | Done (screen on sampled reach; tests/test_nao.py) |
| F13 | High | (v) `--set A.B.C=v` drops `.C` when `A.B` is not a mapping: `--set scf.alpha.O=0.73` sets alpha = 0.73 for every element, validated as fine | Done (error, or promotion of a uniform alpha to a mapping; dataclass fields only) |
| F14 | High | Under MPI an exception on rank 0 only (mkdir, result/figure write, checkpoint) hangs the other ranks in the next collective until wall time (reproduced: `mpirun -n 2` with an unwritable outdir never exits). `COMM.Abort` on any caught exception when parallel | Done (COMM.Abort; tests/test_mpi.py) |
| F15 | High | (v) Every UKS report, JSON and summary panel uses the closed-shell bond-order formula (`wiberg_bond_orders(P, S)` in io/report.py, io/results.py, plotting.py): O2 triplet 1.13 vs open-shell Mayer 1.62 in the same report. Pass P_a/P_b; rename to Mayer | Done (bond_orders(result); JSON adds mayer_bond_orders) |
| F16 | High | `pyxany embed` on a polar (wurtzite) cell yields a block with a macroscopic dipole: ZnO polarization 0.058 e/A^2 at every block size, ~1.7 V/A spurious field (rock salt, zinc blende fine). Compute and refuse/warn on the block dipole; document; later a dipole-compensated scheme | Done (block dipole computed; refused unless --allow-dipole); dipole-compensated scheme still open |
| F17 | High | (v) `passivate --reference` decides "bond present" by exact equality of displacement vectors rounded to 1e-4 (`passivation/detect.py`): with 0.001 A noise 9 of 36 terminators land on existing Si (0.85 A away) while the report says closed shell. Match by angle/length tolerance; check terminator-atom distances | Done (30 deg angular match; clash warning; neighbor lists built once) |
| F18 | Medium->gate | (v) `--no-numba` has no effect either: the environment variable is set after `HAVE_NUMBA` is bound at import. ADR 0013's premise for `runtime.numba` needs a lazy backend choice; folded into F3 | Done with F3 (backend chosen at call time) |

Medium and Low: fixed 2026-11-14 together with F4, F5 and F9 (commits after
0253f66), each with a regression test, except the Bragg-Slater attribution
(Low, UNCERTAIN): the values are kept (the radial-grid table was measured with
them) and their source needs the Slater (1964) table to be checked; the
six reviewer suggestions were done afterwards: Mulliken and Loewdin spin
populations for UKS (report and JSON), `embed` writes on rank 0 only, its
input-file hint gives the path relative to the input file and absolute, the
kcal/mol literal in `scan` replaced by the constants, `pseudo_h.describe`
takes the atom grid, and `compute_forces` reuses alpha(r) and the full
density instead of computing them twice. EDIIS now follows Kudin-Scuseria-Cances with the
coefficient 1/4 that F = dE/dP requires (the paper's 1/2 is for dE/dD = 2F),
checked against brute force on an exact quadratic model. Found on the way:
at the default grid (n_r 30) the H2O force error is 2.2e-3 Ha/Bohr, above the
default optimize.f_max of 1e-3 (docs/forces.md); `forces --compare` now warns.

Medium (original list): EDIIS ignores the energies (`scf/diis.py`; not KSC 2002);
UKS guess puts 2S surplus electrons and an over-occupied alpha core into P_a;
a transition-state restart rescales the hole channel to an integer count (122
extra iterations); the force docstring and `forces --compare` miss the
dominant radial-quadrature error (default f_max 1e-3 below it); alpha-field
mixing and its force term are O(N_pts N_at^2) NumPy (~20 min per call
projected for Si47H60 with `alpha_hf`); exit code 3 for many input errors
(XYZError, TypeErrors, scan coords); `--set` values not YAML-parsed and not
limited to fields; derived defaults (prefix, TS max_iter) resolved before
overrides; `overlap_population.pairs` unvalidated (crash after SCF); `scan`
and `opt` exit 0 with unconverged SCFs; bending scan from a linear molecule
along z does nothing; `--restart` at `max_iter` crashes; `gridcheck` under MPI
reports rank 0's share; `passivate` cannot set role overrides (MnO, FeO, TiC
fail); `passivate --reference` rebuilds neighbor lists per atom (17.7 s vs
0.03 s); smeared HOMO/LUMO taken at occupancy 1e-8; `embed` does not check
site element vs cluster atom (sign of the suggested charge flips);
`parallel.mpi_version` still broken although recorded as fixed.

Low (batch): MPI broadcasts P but not P_a/P_b; checkpoint fingerprint gaps
(alpha mapping order, NAO parameters); excite-mode capacity check missing for
an unlocalized core hole; JSON `Infinity` and float `iter`; xyz truncation
off-by-one; `--emit-yaml` structure path; unvalidated numeric settings;
`pyxany basis KEY` ignores KEY; `scan` writes no checkpoint (docs say it
does); doc drift; Bragg-Slater radii attributed to Slater 1964 do not match
it (UNCERTAIN; values themselves fine); `projected_dos(e_min=None)`; `-q
--json` stdout mixing; unlabelled per-channel passivation gap.

### Release (Q66, Q74, Q75-Q78, Q81)

| id | item | status |
|---|---|---|
| G1 | GitHub repository `ysknoda/pyxany`, created private; made public once CI is green. Alpha release `0.2.0a1` (tag `v0.2.0a1`, `Development Status :: 3 - Alpha`, README alpha notice, GitHub pre-release), following the user's published packages (akitsushima, mizuho, amaterasu). Build backend stays hatchling with an SPDX license declaration (Q77). Author name only in LICENSE, pyproject and CITATION.cff (Q78) | In progress |
| G2 | PyPI. The name `pyxany` was unregistered on 2026-11-13 (HTTP 404). Prerequisites: P0 and its gate, A1 (a stable Python API) | Proposed |

### Features

| id | item | size | depends on | status |
|---|---|---|---|---|
| A1 | Python API: `pyxany.run(config) -> Result` (and `opt`, `forces`), the CLI becoming a thin layer over it; for scripted sweeps, notebooks and a stable surface before PyPI | M | P0 gate | Proposed |
| A2 | Volumetric output (Gaussian cube): selected MOs, total and spin density, difference density against the SAD guess, electrostatic potential (realizes `properties.esp`, ADR 0013). For looking at gap and terminator states | M | P0 gate | Proposed |
| A3 | SCF performance: two-thirds of a 107-atom SCF + forces is the SCF, with chi recomputed every iteration because it exceeds the memory budget (ADRs 0007, 0009). Options to be chosen by profiling, as in step 8 | L | P0 gate | Proposed |
| A4 | Cluster-size convergence of the example gaps (Si, GaAs, CdSe at 17/29/47/87 core atoms; from ADR 0008). A study, not code; the small examples have terminator-dominated LUMOs | L | A3 (cost), A1 (scripting) | Proposed |
| A5 | Several initial spin guesses tried automatically, reporting the lowest (ADR 0008; code-check MEDIUM on `scf/guess.py`). Open-shell defects in clusters | M | P0 gate | Proposed |
| A6 | Irreducible-representation labels for MOs: point-group detection and projection of the converged orbitals (a1, t2, ...), without a symmetry-adapted basis | M | P0 gate | Proposed |
| A7 | XANES / XES from transition-state dipole intensities (ADR 0006 (d)) | M-L | R0; A8 for heavy absorbers | Proposed |
| A9 | Faster convergence of equivalent-atom core holes: the slow mode of ADR 0014 halves the commutator only every ~50 iterations (~200 to 1e-5). Candidates: a second-order step in the occupied-occupied rotation between the localized core orbitals, or a damped/level-shifted treatment of that block. Research; measure on N2 and a symmetric cluster first | M-L | F8 | Resolved by F11 (N2 1s TS: 20 iterations to a commutator of 1e-6) |
| A10 | Project the Pulay DIIS error vector onto the retained (non-linearly-dependent) space, as the convergence test now does (ADR 0014). With dropped directions the unprojected error carries a constant component (MgO6H: 1.4e-4); may help DIIS on diffuse bases. Changes every SCF trajectory, so measure first | S-M | F8 | Proposed |
| A8 | Scalar-relativistic ZORA (atomic solver, molecular Hamiltonian, force terms). Variant and gauge treatment to be decided in its own ADR when scheduled. Spin-orbit coupling stays out of scope (no L2/L3 splitting) | L | R0 | Proposed |

### Kept in the backlog (from ADR 0008)

- Effective core potentials on boundary cations of embedded clusters
  (Mg19O6 works around it; ECP parameters need a source).
- A correlation functional (e.g. VWN5): changes what the code is.
- Full point-group symmetry in the SCF (A6 takes the labeling part).
- Vibrational frequencies (finite-difference Hessian of the analytic forces).

### Rejected

- System-tuned alpha (ADR 0011: energies of different systems or geometries
  would not be comparable; the energy has no minimum in alpha).
- `properties.bde`: separate runs or `pyxany scan` give it (ADR 0013).
- Molden output: it cannot represent the numerical (NAO) basis; cube files
  (A2) are basis-independent.

## Recommended order

1. **P0** (done 2026-11-14) and every Critical/High it found: F10 first
   (NAO results are wrong today), then F11 (changes every UKS trajectory;
   re-measure A9/A10 after), F12-F17, F18 with F3.
2. F1, F2, F7, the README part of F6 and of R0, G1 -- done with 0.2.0a1;
   F8 (High, found by the first CI run) -- done with 0.2.0a2.
3. F3, F4, F5, F9, the rest of F6 and R0 (all S, independent of each other).
4. A1 (small enough to come first; unblocks A4 scripting and G2).
5. A2.
6. A3, then A4.
7. A5, A6.
8. A7, then A8 (or A8 first if the absorbers of interest are heavy).
   A9 before A7 if core holes on equivalent atoms (symmetric clusters) are
   routine: each such run costs ~200 SCF iterations until then.

## Consequences

- ADR 0008 is superseded; its open items are A4, A5, A7 and the backlog list
  above, its rejected item is listed under Rejected.
- Pushing to GitHub is now intended (ADR 0007 said otherwise); remote actions
  are still taken one by one with explicit confirmation.
