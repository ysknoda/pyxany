# 0010. Point-charge (Madelung) embedding

- Status: Accepted
- Date: 2026-10-03
- Decisions: Q42-Q44

## Decision

- `embedding.charges` names a file of fixed point charges (`x y z q` or
  `label x y z q`, `embedding.units`). Each charge is a normalised Gaussian of
  width `embedding.sigma` (Bohr, default 0.5), so the electron potential
  energy `-sum q erf(d/sigma)/d` is finite on every grid point.
- `pyxany embed crystal.yaml -o charges.xyz` builds the charges from a
  conventional cell (lattice, fractional sites with formal charges): an
  Evjen block (Phys. Rev. 39, 675 (1932)) 2 cells + 1 wide, centred on the
  origin, with face / edge / corner sites at 1/2, 1/4, 1/8 of their charge;
  lattice sites occupied by cluster atoms are removed and their total formal
  charge printed as the `molecule.charge` the cluster needs. Check: the block
  is neutral and dipole-free and gives the rock-salt Madelung constant
  1.747565 to 7e-6 at 6 cells (1.747548 at 3).
- The potential enters the one-electron Hamiltonian with the nuclear
  attraction; the nucleus-charge energy is added to E_nn. Both are also
  reported separately, with the number of electrons within
  `embedding.spill_radius` (1 A) of a positive charge; a warning is printed
  above `embedding.spill_warn` (0.1 e).
- Forces: the charges are fixed. Their field enters the Pulay and grid terms
  through the potential and the nuclear term directly; the Hellmann-Feynman
  term is unchanged (the potential does not depend on the cluster nuclei).
  The net force is physical, so translation projection is switched off, in
  `opt` as well. Analytic vs finite differences on H2O with three charges:
  9.3e-5 Ha/Bohr at n_r 40 / n_a 14 (1.0e-4 without charges), 1.7e-5 at
  n_r 60 / n_a 22 -- the grid error, no error of its own.
- The checkpoint fingerprint includes the charges.

## Cluster boundary (Q43): measured, and the remedy

No effective core potentials on the boundary cations (deferred, ADR 0008).
On the bare `[MgO6]10-` cluster in the Evjen field (DZ, sigma 0.5 Bohr) the
diffuse O functions put 1.57 electrons on the neighbouring Mg2+ point charges,
the gap closes and the SCF does not converge; sigma 1.36 / 2.0 Bohr does not
help (1.2 / 1.6 e). SZ converges (gap 2.2 eV, 0.48 e). Making every cation
bonded to a quantum O quantum too (`Mg19O6`, charge +26) gives, with DZ,
convergence in 16 iterations, gap 3.3 eV, 0.03 e near positive charges and
six equivalent O (Loewdin -1.41). That is the shipped example
(`examples/mg19o6_embedded.yaml`) and the documented practice.

## Consequences

- Ionic clusters can be computed without pseudo-hydrogen termination.
- Cation-terminated clusters are larger; boundary ECPs would allow smaller
  ones and are recorded in ADR 0008.
