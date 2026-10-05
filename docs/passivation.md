# Pseudo-hydrogen passivation

A cluster cut out of a crystal has dangling bonds. Left alone they put states in
the gap that have nothing to do with the material. Terminating them with
hydrogen-like atoms of *fractional* nuclear charge satisfies the electron
counting rule and removes those states.

## The charge

From Deng, Li, Li & Wei, *Phys. Rev. B* **85**, 195328 (2012):

> the fractional charge α = (8 − Z)/N for passivation of anion atoms, and
> α = (2M − Z)/M for passivation of cation atoms should be used. Here, N and M
> are the number of the bonds (coordination number) of the corresponding anions
> and cations, respectively.

`Z` is the valence-electron count of the atom being terminated.

**The charge depends on the coordination number as well as the element.** For
tetrahedral coordination (N = M = 4) both expressions reduce to `2 − Z/4`, but
elsewhere they differ: in rock salt (N = M = 6) an oxygen terminator carries 1/3
while a magnesium one carries 5/3, against the tetrahedral 1/2 and 3/2.

| material | structure | anion (Z, N) → α | cation (Z, M) → α |
|---|---|---|---|
| Si, Ge, C | diamond | (4, 4) → 1 | (4, 4) → 1 |
| SiC | zinc blende | C (4, 4) → 1 | Si (4, 4) → 1 |
| GaAs, InP | zinc blende | As/P (5, 4) → 3/4 | Ga/In (3, 4) → 5/4 |
| GaN | wurtzite | N (5, 4) → 3/4 | Ga (3, 4) → 5/4 |
| ZnO, ZnS, CdSe | wurtzite / zinc blende | O/S/Se (6, 4) → 1/2 | Zn/Cd (2, 4) → 3/2 |
| MgO | rock salt | O (6, 6) → 1/3 | Mg (2, 6) → 5/3 |
| TiO₂ | rutile | O (6, 3) → 2/3 | Ti (4, 6) → 4/3 |
| CuCl | zinc blende | Cl (7, 4) → 1/4 | Cu (1, 4) → 7/4 |

All of these coincide with the pseudo-hydrogen charges distributed with VASP
(1/4, 1/3, 5/12, 1/2, 7/12, 2/3, 3/4, 5/4, 4/3, 3/2, 5/3, 7/4). The rule is
implemented in general, so any charge a structure calls for is available; the
report flags whether the value is one of the standard ones.

## Why there are no free parameters here

VASP needs a PAW dataset per pseudo-hydrogen because it is a pseudopotential
code. `pyxany` is all-electron: a terminator with charge α is *completely*
specified by putting `−α/r` in the nuclear attraction, `α_A Z_B/R` in the nuclear
repulsion, and α electrons in its 1s. There is nothing left to parameterize.

Its **basis function** is likewise solved for rather than fitted: the spherical
Kohn-Sham problem is solved for nuclear charge α with α electrons, and a 1s
Slater function is fitted to the result. Two limits make that checkable:

- switching off the electron-electron terms recovers the exact hydrogenic 1s,
  and the fitted exponent is exactly α;
- at α = 1 the raw fit gives ζ = 0.882 while STO tables use ζ = 1.24 for
  hydrogen, the difference being Xα self-interaction. Mixing the two would make
  terminators systematically more diffuse than every other function in the
  molecule, so the exponent is rescaled by the single constant that makes α = 1
  reproduce the basis set's own hydrogen value. That constant is fixed entirely
  by the requirement that a unit-charge pseudo-hydrogen *be* a hydrogen.

  The terminator then receives **every hydrogen function of the chosen basis
  set**, each contracted by the single ratio ζ(α)/ζ(1) (Slater exponents times
  the ratio, Gaussian exponents times its square). An α = 1 terminator is
  therefore bit-for-bit a hydrogen in `sz`, `dz`, `dzp` and the Gaussian sets
  alike (asserted in `tests/test_pseudo_h_basis.py`). Earlier versions gave it
  only the one SZ-calibrated 1s, which in Si5H12/DZ moved the inner Si charge
  to -0.42 e against -0.03 e with real hydrogens.

| α | 1/4 | 1/3 | 5/12 | 1/2 | 7/12 | 2/3 | 3/4 | 1 | 5/4 | 4/3 | 3/2 | 5/3 | 7/4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ζ | 0.451 | 0.547 | 0.640 | 0.730 | 0.818 | 0.904 | 0.990 | 1.240 | 1.485 | 1.565 | 1.726 | 1.884 | 1.963 |
| ⟨r⟩ (Bohr) | 4.68 | 3.85 | 3.30 | 2.89 | 2.58 | 2.33 | 2.13 | 1.70 | 1.42 | 1.35 | 1.22 | 1.12 | 1.08 |

## The X–H\* distance

There is no recommended value in the literature to copy, and none is needed:
the distance is a quantity to **optimize**, not to tabulate. The initial guess
scales this structure's own nearest-neighbor bond by the ratio of summed
covalent radii — an explicitly additive combination rule, not a per-element
number reused as a pair quantity:

```
d(X–H*) = d_nn(X–Y) · [r_cov(X) + r_cov(H)] / [r_cov(X) + r_cov(Y)]
```

with radii from Cordero *et al.*, *Dalton Trans.* 2832 (2008). Checked against
real molecular bonds:

| host | reference bond used | d(X–H\*) initial | known X–H | difference |
|---|---|---|---|---|
| Si | Si–Si 2.352 Å | 1.504 Å | SiH₄ 1.48 Å | +1.6 % |
| Se | Se–Cd 2.620 Å | 1.498 Å | H₂Se 1.46 Å | +2.6 % |
| O | O–Mg 2.105 Å | 0.986 Å | H₂O 0.958 Å | +2.9 % |

Then relax them:

```bash
pyxany opt cluster.yaml --freeze-core     # only the terminators move
```

## Workflow

```bash
pyxany passivate cluster.xyz --coordination 4 \
     -o cluster_h.xyz --emit-yaml cluster_h.yaml
pyxany opt cluster_h.yaml --freeze-core
pyxany run cluster_h.yaml
```

The report prints every detected bond with the cutoff used, the origin of each
terminator charge, each X–H\* distance with the radii it came from, and the
total electron count — which must be an even integer if the counting rule has
been satisfied.

**Coordination numbers.** These are inferred as the largest coordination seen
for each element, on the assumption that a fragment cut from a crystal keeps
fully coordinated atoms in its interior. For small clusters that assumption
fails and `--coordination` should be given explicitly; the tool warns when the
inferred value looks too low to be a bulk number.

## Comparing against real hydrogen

`--scheme hydrogen` terminates with ordinary hydrogen (α = 1) instead, which is
the comparison the reference paper is about: real-hydrogen passivation can raise
or lower the gap depending on the ionicity of the compound, and the difference
from pseudo-hydrogen shrinks as the material becomes more covalent. Run the same
cluster both ways to see it.

## Checking that the terminators stay out of the gap

`pyxany run` prints a **passivation check** for any structure with
pseudo-hydrogens: the Loewdin weight of the HOMO and LUMO on the terminators,
the plain gap, and the **core gap** -- between the highest occupied and lowest
empty orbitals whose terminator weight is below
`properties.terminator_weight_max` (default 0.5, a heuristic). The JSON output
carries the per-orbital weights.

Loewdin weights are used rather than Mulliken: with the diffuse functions of
fractional terminators Mulliken weights leave [0, 1] (1.55 on the MgO6 LUMO)
and overstate the terminator share about twofold. A real-space check
(integrating |psi|^2 over the terminators' Becke cells) agrees with Loewdin
except for alpha = 1/3 terminators, whose large Becke cells bias it upward.

What the examples show (2026-10-01, DZ basis, unrelaxed X-H* distances):

| cluster | HOMO weight | LUMO weight | gap (eV) | core gap (eV) |
|---|---|---|---|---|
| GaAs4 | 0.18 | 0.52 | 5.31 | 5.81 (1 terminator state below) |
| Cd4Se4 | 0.04 | 0.64 | 2.54 | 4.98 (6 terminator states below) |
| MgO6 | 0.10 | 0.72 | 5.87 | 10.76 (18 below -- not meaningful, see below) |

The occupied edge is the cluster interior in every case, but in these very
small clusters (5-7 core atoms, 12-30 terminators) empty states with
substantial terminator character sit at or below the interior's conduction
states. A single-1s terminator basis hid them (CdSe terminator band at
+2.2 eV instead of +0.3 eV).

Relaxing the terminators does not remove them: GaAs4 after
`opt --freeze-core` (X-H* 1.524 -> 1.546 A) has LUMO weight 0.54 and gap
5.21 eV against 0.52 and 5.31 eV unrelaxed. The effect belongs to clusters
this small, not to an unrelaxed geometry.

The core gap is only a guide. Its threshold is a heuristic, and when most
atoms are terminators (MgO6: 30 of 37) even delocalized empty states carry
more than half their weight on them, so the core gap runs off to the first
state that happens to avoid them (10.8 eV). For a gap that represents the
material, enlarge the cluster until the terminators are a minority and the
frontier weights fall well below 0.5.
