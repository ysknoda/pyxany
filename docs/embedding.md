# Embedding a cluster in point charges

An ionic cluster cut from a crystal misses the Madelung potential of the rest
of the lattice. pyxany adds it as fixed point charges (ADR 0010).

## 1. Describe the crystal

```yaml
# mgo_crystal.yaml
lattice:                    # conventional cell, Angstrom
  - [4.21, 0.0, 0.0]
  - [0.0, 4.21, 0.0]
  - [0.0, 0.0, 4.21]
sites:                      # [label, fractional x, y, z, formal charge]
  - [Mg, 0.0, 0.0, 0.0, +2]
  ...
origin: [0.0, 0.0, 0.0]     # where fractional (0, 0, 0) sits in the cluster file
cells: 3                    # Evjen block 7 cells wide
cluster: mg19o6.xyz         # its sites are left out
```

## 2. Generate the charges

```bash
pyxany embed mgo_crystal.yaml -o mg19o6_charges.xyz
```

The command prints the formal charge of the removed sites; give the cluster
that charge (`molecule.charge: 26` for Mg19O6) so that cluster plus charges
are neutral.

## 3. Run

```yaml
embedding:
  charges: mg19o6_charges.xyz
```

The report adds the electron-charge and nucleus-charge energies and the number
of electrons within 1 A of a positive charge.

## Choosing the cluster

Point charges have no Pauli repulsion. Electrons of an anion next to a
point-charge cation leak into it, the more so with diffuse basis functions:
for `[MgO6]10-` with DZ, 1.6 electrons do, the gap closes and the SCF does
not converge. Make every cation bonded to a quantum anion quantum as well
(MgO6 -> Mg19O6): 0.03 electrons, gap 3.3 eV. If the report warns about
electrons on positive charges, enlarge the cluster this way.

The forces include the field of the charges, which stay fixed; their net
force on the cluster is physical, so it is not projected out.
