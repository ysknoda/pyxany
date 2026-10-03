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

grid:
  n_r:              30      # radial points for second-row atoms (scaled per element)
  n_a:              12      # polar points per atom (2*n_a azimuthal)
  partition:        becke   # Becke fuzzy cells; 'none' is rejected, see below
  angular:          product # product (n_a x 2n_a) | lebedev (same degree, ~2/3 the points)
  poisson_l_max:    auto    # multipole cut-off of the Hartree expansion (n_a - 1)
  screening_tol:    1.0e-10 # skip atom pairs that cannot overlap this much
  n_r_by_element:   {}      # e.g. {Si: 40, Cd: 80}: overrides the per-element table
  pruning:                  # fewer angular points on inner shells (ADR 0009)
    enabled:        true
    regions:        auto    # or [[r/R_Bragg bound, n_a], ...]; auto = [[0.25, n_a/3], [0.5, 2n_a/3]]
  basis_cutoff:     1.0e-6  # drop basis functions below this on a block of points; 0 = dense
  pair:                     # two-centre grids of the S and T matrices
    n_r:            40
    n_a:            14
  atom:                     # log grid of the spherical atom solver (NAO basis,
    n_points:       2000    #   SAD guess, pseudo-hydrogen exponents)
  guess:                    # one-atom grid the SAD guess is projected on
    n_r:            80
    n_a:            14
  one_centre:               # radial quadrature of one-centre non-Slater integrals
    n_r:            400
  chunk_size:       auto    # points per chi block; auto uses memory_budget_mb
  memory_budget_mb: 4096
  radial_by_element: true   # false: every atom gets exactly n_r

embedding:                  # optional point-charge (Madelung) embedding, ADR 0010
  charges:          null    # file of 'x y z q' lines; 'pyxany embed' writes one
  units:            angstrom
  sigma:            0.5     # Gaussian width of every charge, Bohr
  spill_radius:     1.0     # report electrons this close (A) to positive charges
  spill_warn:       0.1     # and warn above this many

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
  overlap_population:              # bond overlap population (ADR 0005)
    pairs: []                      # [[0, 1], ...] 0-based; empty: Mayer B.O. >= min_bond_order
    min_bond_order: 0.2
    heatmap: false                 # atom-pair matrix figure; small systems only
    sigma: 0.3                     # broadening of the COOP and PDOS curves (eV)
  terminator_weight_max: 0.5       # Loewdin weight above which an orbital is a terminator state

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

## Bonding figures

With `output.figure: true` (and matplotlib installed) `pyxany run` writes, next
to the summary PNG, three figures as both PDF and PNG:

| file | content |
|---|---|
| `<prefix>_coop.*` | MO overlap population vs orbital energy, one panel per element-pair type (bars per orbital, broadened curve; > 0 bonding, < 0 antibonding). The axis is scaled to the occupied orbitals; empty-orbital bars beyond it are clipped and marked, because Mulliken overlap populations of diffuse empty orbitals can exceed 1 by far (H2O/DZ: -7 for the O-H sigma*). |
| `<prefix>_levels_op.*` | orbital levels coloured by the overlap population of the dominant pair type (solid occupied, dashed empty). |
| `<prefix>_pdos.*` | Loewdin partial DOS by element and by (element, l); species in order of nuclear charge. Beta drawn negative for open shells. |
| `<prefix>_op_heatmap.*` | atom-pair overlap population matrix, if `overlap_population.heatmap: true`. |

The energy window is HOMO - 20 eV to HOMO + 10 eV. The same numbers are in the
text report (strongest pairs) and in the JSON output under `bonding`.

## Slater transition state

```yaml
transition_state:
  remove: homo                  # valence ionisation: 'homo', or an MO number (1-based)
  # remove: {atom: 3, shell: 1s}  # core ionisation of atom 3 (1-based, as in the .xyz)
  # excite: {from: homo, to: lumo}
  spin: alpha                   # channel the half electron leaves
  electrons: 0.5
```

Half an electron is removed (or moved) every SCF iteration; the run is
unrestricted automatically and `scf.smearing` must be 0. The report gives
`-eps(TS)` (ionisation) or `eps_to - eps_from` (excitation), which by Janak's
theorem equal the Delta-SCF energy to third order (H2O HOMO: 11.16 eV against
Delta-SCF 11.12 eV).

**Indices here are 1-based**, as you read the .xyz file; the `--coord` option
of `pyxany scan` is 0-based.

A core hole needs a basis with principal quantum numbers (STO sets such as
`dz`, or `nao`). With several equivalent atoms the hole is localised on the
one named: the run starts by removing the combination of their core orbitals
with the largest weight on that atom, and switches to the orbital itself once
an eigenvector is localised (N2: 0.998 on the named atom). Keep
`poisson_l_max` at its default (`auto`); a low value spoils the energy/orbital
consistency the method relies on (Janak residual 5.8e-3 Ha at l_max = 2,
1.4e-5 at 13).
