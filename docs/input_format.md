# Input files

A calculation is described by exactly two files: an `.xyz` structure and a
`.yaml` holding every setting. `pyxany init <name>` writes a commented pair to
start from, and `pyxany validate <input.yaml>` prints the fully resolved settings
without running anything.

## Structure (`.xyz`)

Standard XYZ: atom count, a free-form comment, then `symbol x y z` per line.
Coordinates are Ångström unless `molecule.units: bohr` says otherwise.

```
3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
```

Pseudo-hydrogen terminators appear as their own species labels (`HCd`, `HO`,
…) and must be declared in `passivation.species`. An undeclared label that is
not an element symbol is an error, not a silently skipped line. Geometry
optimisation writes multi-frame trajectories in the same format.

## Settings (`.yaml`)

Every key below is optional except `molecule.structure`; the values shown are
the defaults. **Unknown keys are errors** — a typo is reported rather than
leaving a default silently in place.

```yaml
molecule:
  name:      molecule       # used for output file names
  structure: molecule.xyz   # resolved relative to this YAML file
  charge:    0
  mult:      1              # >1 or an odd electron count switches on UKS
  units:     angstrom       # angstrom | bohr

passivation:                # only for structures containing terminators
  scheme:  none             # none | pseudo_hydrogen | hydrogen
  species: {}               # written by 'pyxany passivate --emit-yaml'
  basis:   derived          # derive the 1s exponent from the fractional charge

basis:
  set:     dz               # sz szp dz dzp tz tzp tzpp | sto3g 631g 631gd | nao
  nao_n_r: 2000             # radial points for the atomic solve behind 'nao'

grid:
  n_r:              30      # radial points for second-row atoms (scaled per element)
  n_a:              12      # polar points per atom (2*n_a azimuthal)
  partition:        becke   # Becke fuzzy cells; 'none' is rejected, see below
  poisson_l_max:    auto    # multipole cut-off of the Hartree expansion (n_a - 1)
  screening_tol:    1.0e-10 # skip atom pairs that cannot overlap this much
  pair_n_r:         40      # radial points of the two-centre S/T grids
  pair_n_a:         14
  chunk_size:       auto    # points per chi block; auto uses memory_budget_mb
  memory_budget_mb: 4096
  radial_by_element: true   # false: every atom gets exactly n_r

scf:
  max_iter:         100
  e_tol:            1.0e-6  # total-energy convergence (Ha)
  d_tol:            1.0e-5  # density-matrix rms convergence
  alpha:            0.7     # Slater X-alpha exchange parameter
  unrestricted:     false   # forced true for odd electrons or mult > 1
  smearing:         0.0     # Fermi smearing kT (Ha)
  s_threshold:      5.0e-4  # drop overlap eigenvalues below this
  ediis_iters:      8       # EDIIS iterations before switching to Pulay DIIS
  mix_ediis:        0.25
  mix_diis:         0.4

forces:
  enabled:              true
  method:               analytic   # analytic | finite_difference
  fd_step:              0.02       # Bohr
  project_translation:  true       # enforce sum(F) = 0

optimise:
  enabled:    false
  method:     lbfgs       # lbfgs | cg
  f_max:      1.0e-3      # Ha/Bohr
  max_steps:  100
  freeze:     none        # none | core | [0, 3, 4]
  trajectory: true

properties:
  population: true
  dipole:     true
  dos: {sigma: 0.3, n_points: 600}
  bde: {enabled: false}
  esp: {enabled: false, n_pts: 20, margin: 3.0}

output:
  dir:    "."
  prefix: ""              # defaults to molecule.name
  figure: true
  json:   true
  npz:    false
  timing: true

runtime:
  numba:     auto         # auto | on | off
  mpi:       auto
  verbosity: 1
```

### Notes on a few settings

**`grid.partition`** only accepts `becke`. Omitting the cell functions
overcounts overlapping regions by roughly the number of atoms; the option
exists in the code purely so `tests/test_grid.py` can demonstrate that, and is
rejected by the input validator.

**`grid.n_r`** is the radial size for second-row atoms. With
`radial_by_element: true` (the default) each atom gets `ceil(n_r × ratio)`
points, the ratio *measured* per element by `tools/radial_grid_sizes.py`: the
grid error in the ionisation energy of the near-exact X-alpha atom is brought
to the level C, N, O and F reach at `n_r`. At `n_r = 30`: H 36, C 25, O 29,
Si 33, Ga/As 47, Cd 60, Pb 71. Without it, a Ga or Cd atom at `n_r = 30` carries
~0.5 Ha of radial quadrature error in its total energy (energy *differences*
are affected far less: 6 meV for the Ga ionisation energy).

**`grid.poisson_l_max`** truncates the multipole expansion of each atom's share
of the density. The default `auto` uses `n_a - 1`, the highest multipole the
angular grid resolves (larger values are rejected). Lower values are cheaper --
cost scales as `(l_max + 1)²` -- and are adequate for energies, but the force
error is set by this truncation: on CO it is 1.6e-2 Ha/Bohr at `l_max = 6`,
1.3e-3 at 12 and 1e-4 at 16, so keep `auto` for optimisations.

**`scf.alpha`** — 0.7 is the customary DV-Xα value. `alpha = 2/3` makes the
functional exactly Dirac (LDA) exchange, which is what
`tests/test_xc.py` uses to pin the absolute scale.

**`optimise.freeze: core`** holds every non-terminator atom fixed, which is how
pseudo-hydrogen positions are meant to be determined.

## Overriding from the command line

```bash
pyxany run water.yaml --set scf.alpha=0.75 --set grid.n_r=40 --basis dzp -o /scratch
```

`--set SECTION.KEY=VALUE` is repeatable and validated exactly like the file.

## Output

| file | contents |
|---|---|
| `<prefix>_results.txt` | human-readable report |
| `<prefix>_results.json` | energies, forces, eigenvalues, charges, timings |
| `<prefix>_final.xyz` | the geometry used (or reached) |
| `<prefix>_traj.xyz` | optimisation trajectory |
| `<prefix>_results.png` | six-panel summary figure |
| `<prefix>_timing.json` | per-section wall times |
| `<prefix>_scan.csv` | `pyxany scan` output |
| `<prefix>_matrices.npz` | P, C, S, T, eigenvalues (`output.npz: true`) |
