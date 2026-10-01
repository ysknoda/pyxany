# pyxany

All-electron **DV-Xα** (Discrete Variational Xα) molecular orbital calculations,
with a command-line interface aimed at semiconductor cluster models.

Slater-type, contracted-Gaussian and numerical atomic orbital basis sets;
restricted and unrestricted Kohn-Sham SCF; analytic nuclear forces; geometry
optimisation; and **pseudo-hydrogen passivation** of dangling bonds using the
electron-counting rule.

## Install

```bash
pip install -e ".[all]"     # numba, mpi4py and matplotlib are all optional
```

Required: Python ≥ 3.10, NumPy ≥ 2.0, SciPy ≥ 1.11, PyYAML.
Everything works without the optional extras — more slowly, serially, and
without figures.

## Quick start

```bash
pyxany init water                  # writes water.yaml + water.xyz
pyxany run water.yaml              # energy, forces, populations, dipole
```

A calculation is described by exactly two files: an **`.xyz`** structure and a
**`.yaml`** holding every setting.

```yaml
molecule:
  name:      water
  structure: water.xyz
  charge:    0
  mult:      1
basis:
  set: dz
grid:
  n_r: 30
  n_a: 12
scf:
  alpha: 0.7
forces:
  enabled: true
```

## Commands

| command | purpose |
|---|---|
| `pyxany run in.yaml` | single-point SCF: energy, forces, populations, dipole |
| `pyxany opt in.yaml` | geometry optimisation (L-BFGS-B on analytic forces) |
| `pyxany forces in.yaml --compare` | analytic forces, checked against finite differences |
| `pyxany scan in.yaml --coord bond:0,1 --range 1.0 1.6 --steps 13` | energy scan → CSV |
| `pyxany passivate cluster.xyz` | add hydrogen / pseudo-hydrogen terminators |
| `pyxany gridcheck in.yaml` | integration-grid quality diagnostic |
| `pyxany validate in.yaml` | check the input and print the resolved settings |
| `pyxany basis dz --element Se` | basis-set coverage |
| `pyxany info` | versions, active backend, MPI ranks |

Any setting can be overridden without editing the file:

```bash
pyxany run water.yaml --set scf.alpha=0.75 --set grid.n_r=40 --basis dzp
```

`--dry-run` reports grid size, memory and screening before committing to a job.
Exit codes let batch scripts branch on the outcome: `0` success, `1` input
error, `2` SCF not converged, `3` runtime error, `4` optimisation not converged.

### Parallel and long runs

```bash
mpirun -n 8 pyxany run cluster.yaml
nohup setsid mpirun -n 128 pyxany run big.yaml --log run.log &
```

The DV grid is domain-decomposed across ranks; output and file writes happen on
rank 0 only.

## Pseudo-hydrogen passivation

A cluster cut from a crystal has dangling bonds that put spurious states in the
gap. Terminating them with hydrogens of *fractional* nuclear charge satisfies the
electron-counting rule and removes those states.

Following Deng, Li, Li & Wei, *Phys. Rev. B* **85**, 195328 (2012):

```
anion  terminator:  α = (8 − Z) / N
cation terminator:  α = (2M − Z) / M
```

with `Z` the valence-electron count and `N`, `M` the coordination numbers.
**The charge depends on the coordination number, not just on the element** — in
rock salt (N = M = 6) an oxygen terminator carries 1/3 and a magnesium
terminator 5/3, while the tetrahedral values would be 1/2 and 3/2.

```bash
pyxany passivate cdse_cluster.xyz --coordination 4 \
     -o cdse_h.xyz --emit-yaml cdse_h.yaml
pyxany opt cdse_h.yaml --freeze-core      # relax only the terminators
```

| material | structure | anion → α | cation → α |
|---|---|---|---|
| Si, Ge, C | diamond | 1 | 1 |
| GaAs, InP, GaN | zinc blende / wurtzite | 3/4 | 5/4 |
| ZnO, ZnS, CdSe | wurtzite / zinc blende | 1/2 | 3/2 |
| MgO | rock salt | 1/3 | 5/3 |
| TiO₂ | rutile | 2/3 | 4/3 |
| CuCl | zinc blende | 1/4 | 7/4 |

All of these coincide with the pseudo-hydrogen charges distributed with VASP.

Because `pyxany` is an all-electron code there is **no pseudopotential to
parameterise**: a terminator is fully specified by its charge, and its basis
function is obtained by solving the spherical Kohn-Sham problem for that charge
rather than being fitted. The X–H\* distance starts from a covalent-radius
estimate and is then relaxed with the code's own forces.

Setting `--scheme hydrogen` uses ordinary hydrogen everywhere instead, which is
the comparison the reference paper is about.

## When the SCF will not converge

A cluster with a crowded frontier often has two orbitals a fraction of a
millihartree apart. The occupation then swaps between them from iteration to
iteration and the density never settles, even though the energy is already
stable to five digits. `pyxany` recognises this and says so:

```
  [NOT CONVERGED] after 150 iterations
  [diagnosis] the energy is stable to 3.2e-04 Ha but the density is
  not settling, and the HOMO-LUMO gap is only 0.0099 eV (3.63e-04 Ha).
  Near-degenerate frontier orbitals let the occupation swap between
  iterations. Fermi smearing removes the ambiguity:

      pyxany run <input.yaml> --set scf.smearing=0.002
```

On the Cd₄Se₄H₁₂ example that takes it from 150 iterations without converging to
17 iterations converged, at a *lower* total energy. Report `E_free` rather than
`E_total` when smearing is on, and check the result against a second width.

## Accuracy

Every physical ingredient is validated against an independent reference rather
than against itself; `tests/` runs these as assertions.

| quantity | check | result |
|---|---|---|
| grid | ∫ρ dV vs electron count, 1–5 atoms | exact to < 1e-5 |
| Becke weights | Σ_A w_A(r) = 1 | machine precision |
| real harmonics | orthonormality, l ≤ 3 | 1e-15 |
| Xα coefficient | α = 2/3 reproduces Dirac exchange | machine precision |
| kinetic energy | H 1s, T = ζ²/2 | 10 digits |
| two-centre overlap | H₂ 1s–1s vs closed form | 1e-9 … 1e-11 |
| Hartree potential | Gaussian charges vs erf(r)/r | 3e-5 relative on E_H, converging at order 4 |
| atomic solver | hydrogenic 1s/2s/2p/3d | ε exact, R to 1e-11 |
| analytic forces | vs central finite differences | 2e-5 Ha/Bohr at a fine grid |
| force sum rule | \|Σ_A F_A\| | 1e-15 |
| numba vs numpy | total energy | 1e-16 |

Physically, Xα/DZ lands ~1 % above Hartree-Fock for closed-shell molecules
(CO −111.69 vs −112.79 Ha; H₂O −75.29 vs −76.07 Ha), which is the expected
behaviour of the functional rather than a numerical error.

## License

MIT — see [LICENSE](LICENSE).

## References

- Ellis & Painter, *Phys. Rev. B* **2**, 2887 (1970) — DV method
- Adachi & Tsukada, *J. Phys. Soc. Jpn* **45**, 875 (1978) — DV-Xα
- Becke, *J. Chem. Phys.* **88**, 2547 (1988) — fuzzy-cell partitioning
- Treutler & Ahlrichs, *J. Chem. Phys.* **102**, 346 (1995) — M4 radial grid
- Pulay, *Chem. Phys. Lett.* **73**, 393 (1980) — DIIS
- Kudin, Scuseria & Cancès, *J. Chem. Phys.* **116**, 8255 (2002) — EDIIS
- Clementi & Raimondi, *J. Chem. Phys.* **38**, 2686 (1963) — single-zeta STO
- Clementi & Roetti, *At. Data Nucl. Data Tables* **14**, 177 (1974) — double/triple-zeta STO
- Cordero *et al.*, *Dalton Trans.* 2832 (2008) — covalent radii
- Deng, Li, Li & Wei, *Phys. Rev. B* **85**, 195328 (2012) — pseudo-hydrogen passivation
