# What changed relative to the original `dv-xa.py`

`dvxa` was extracted from a single 6,994-line script. The restructuring itself
was mechanical, but reading the code turned up several defects that changed
every number it produced. They are all fixed, so **`dvxa` does not reproduce the
old outputs** — this page records what was wrong, how it was confirmed, and by
how much the results move.

## Baseline measured from the original script

Captured before any code was touched, at `n_r=25, n_a=10, alpha=0.7`:

| case | N electrons | Tr(PS) | ∫ρ dV | ∫ρ dV / N | E_total (Ha) | near-HF reference |
|---|---|---|---|---|---|---|
| Ne (szp) | 10 | 10.000 | 7.930 | **0.793** | −82.219 | −128.55 |
| H₂ (sz) | 2 | 2.000 | 3.059 | **1.529** | −3.193 | −1.13 |
| CO (sz) | 14 | 14.000 | 17.052 | **1.218** | −103.663 | −112.79 |
| CO (dz) | 14 | 14.000 | 17.108 | **1.222** | −111.952 | −112.79 |
| H₂O (sz) | 10 | 10.000 | 22.506 | **2.251** | −196.311 | −76.07 |
| CH₃ (sz, doublet) | 9 | **16.000** | 49.850 | **5.539** | −119.636 | −39.58 |

Two things stand out. The grid does not integrate the density to the electron
count, and the discrepancy grows with the number of atoms. And CH₃, a
nine-electron molecule, has a density matrix holding sixteen electrons.

## The defects

### 1. A spurious `sin θ` in the grid weights

The polar nodes are Gauss-Legendre nodes **in cos θ**, so the volume element is
`r² dr d(cos θ) dφ` and no `sin θ` factor belongs in the weight. The original
included one, scaling every grid integral by

    ∫₋₁¹ √(1−u²) du / ∫₋₁¹ du = (π/2)/2 = π/4 = 0.7854

which is exactly the 0.793 seen for the single neon atom above. The same file
contained a second, correct convention in its one-centre routines — the two
disagreed with each other.

### 2. No Becke partitioning

The molecular grid was a plain concatenation of atom-centred grids. Each of
those integrates over all space on its own, so overlapping regions were counted
once per atom. Measured on `∫|χ_1s|² dV`, whose exact value is 1:

| atoms | original | `sin θ` removed | + Becke partitioning |
|---|---|---|---|
| 1 | 0.78576 | 1.00000 | 1.000000 |
| 2 | 1.20870 | 1.98840 | 0.999999 |
| 3 | 2.14130 | 2.98697 | 1.000003 |
| 4 | — | — | 0.999992 |

For a 200-atom cluster this alone would have been a two-orders-of-magnitude
error. Fixed by removing the `sin θ` and adding Becke (1988) fuzzy-cell weights.

### 3. Slater Xα coefficient a factor of two too large

The original used `ε_x = −2.25 α (3ρ/π)^(1/3)` and `V_x = −3.0 α (3ρ/π)^(1/3)`.
The standard functional is

    ε_x = −(9/8) α (3ρ/π)^(1/3),   V_x = (4/3) ε_x = −(3/2) α (3ρ/π)^(1/3)

i.e. exactly half of each. The internal relation `V = (4/3)ε` held, so the SCF
still converged — it was simply running at an effective α of 1.4.

Confirmed independently: at α = 2/3 the Slater functional **is** Dirac exchange,
`e_x = −(3/4)(3/π)^(1/3) ρ^(4/3)`. The corrected code reproduces that to machine
precision; the original was off by two. (`tests/test_xc.py`)

### 4. d-orbital angular factors halved in the JIT path

The numba kernels used these constants for `l = 2`:

| (l, m) | correct | original JIT | ratio |
|---|---|---|---|
| (2, 0) | 0.31539156525252 | 0.31539156525252 | 1.0 |
| (2, ±2), x²−y² | 0.54627421529604 | 0.27313710764801 | **0.5** |
| (2, ±1), (2,−2) | 1.09254843059208 | 0.54627421529603 | **0.5** |

`l = 0` and `l = 1` were right; `l ≥ 3` fell back to a spherical constant, so
f functions were silently wrong. Because the JIT path was taken whenever numba
happened to be installed, **the answer depended on whether an optional package
was present**. Both paths now consume one shared coefficient table and are
pinned together by a test.

### 5. Open-shell electron count doubled

`_fermi_occupations` was called with `2 × n_alpha` electrons *and* a per-orbital
capacity of 1, so each spin channel received twice its electron count. This is
the `Tr(PS) = 16` for nine-electron CH₃ in the baseline table. Every
odd-electron or `mult > 1` calculation was affected.

### 6. GTO contractions read with the wrong normalisation convention

The tabulated coefficients were treated as multiplying *unnormalised*
primitives. Standard basis-set tables (EMSL / Gaussian / Basis Set Exchange) use
**normalised** primitives; the self-overlap of a shell computed that way comes
out at exactly 1.000000, and at 8.5e-1, 8.9e-3, 9.2e-3 … under the other
convention. A global renormalisation hid the error in the total, but the
relative weights of the primitives — the shape of the function — were wrong.
STO-3G hydrogen now gives T = 0.7600 Ha, the accepted value.

### 7. `build_J` was never reduced across MPI ranks

`V_J` was `Allreduce`d on the grid, but the final `(χ w V_J)ᵀ χ` contraction
returned each rank's partial matrix without summing. `build_Vnuc` and
`build_Vxc` did reduce, so J alone was wrong under `mpirun`.

### 8. Smaller defects

- The 6-31G(d) polarisation block tested a loop variable that had leaked out of
  the shell loop; with an empty shell list it raised `NameError`.
- `_LM_TO_CART` mapped d_z² to the single monomial `z²` and d_x²−y² to `x²`,
  so the "analytical" Obara-Saika path returned approximations.
- `eps[n_occ]` crashed for a single atom in a minimal basis (`n_occ == n_basis`).
- J was built from the freshly diagonalised density while V_xc and the energy
  used the mixed one, so the printed energy was not a functional of any single
  density.
- BDE atomic references used `mult = 1` for every even-Z atom (carbon and oxygen
  are both ³P ground states).
- `warnings.filterwarnings("ignore")` at module scope hid every numerical
  warning in the program.

## Where the numbers land now

Same molecules, `dz` basis, `n_r=40, n_a=14`:

| system | dvxa (Xα) | near-HF | ratio | ∫ρ dV / N | Tr(PS) / N |
|---|---|---|---|---|---|
| He | −2.753 | −2.862 | 0.962 | 1.000000 | 1.000000 |
| Ne | −127.385 | −128.547 | 0.991 | 1.000000 | 1.000000 |
| Ar | −523.857 | −526.817 | 0.994 | 1.000000 | 1.000000 |
| H₂ | −1.067 | −1.134 | 0.941 | 1.000000 | 1.000000 |
| CO | −111.539 | −112.79 | 0.989 | 1.000001 | 1.000000 |
| H₂O | −75.204 | −76.07 | 0.989 | 1.000000 | 1.000000 |
| CH₃ (doublet) | −39.005 | −39.58 | 0.985 | 1.000000 | 1.000000 |

Xα without correlation sitting about 1 % above Hartree-Fock is the expected
behaviour of the functional. The conservation columns are now exact for every
system, open-shell included.

## What was carried over unchanged

The Treutler-Ahlrichs M4 radial map, the DIIS and EDIIS accelerators, canonical
orthogonalisation, the Obara-Saika recursions, the six-panel summary figure, and
all ten element/basis data tables. `tests/test_data.py` re-imports the original
script and asserts that every key and every value survived the conversion to
JSON; it runs as long as `dv-xa.py` is present and skips afterwards.
