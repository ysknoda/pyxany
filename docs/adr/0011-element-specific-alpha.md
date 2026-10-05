# 0011. Element-specific X-alpha parameter

- Status: Accepted
- Date: 2026-10-04
- Decisions: Q39-Q41, Q46-Q48, Q47', Q47'', Q49-Q51

## Decision

- `scf.alpha` is a number (all elements; default 0.7, ADR 0002), `alpha_hf`
  (recomputed table, below) or a mapping `{default: 0.7, Si: 0.73}`.
  Pseudo-hydrogens use hydrogen's value.
- Between atoms: the Becke-weighted mean `alpha(r) = sum_A w_A(r) alpha_A`.
  A uniform setting stays a plain float, so every uniform calculation is
  unchanged bit for bit.
- Forces: alpha(r) moves with the nuclei. E_x is linear in alpha, so the extra
  term is `sum_k w_k (e_x/alpha)_k d alpha(r_k)/dR_B` with the grid points
  moving with their owners (`alpha_mixture_gradient`); d alpha/dR matches
  finite differences to 1.6e-10, the forces (H2O, alpha_O 0.74, alpha_H 0.97)
  to 1.3e-4 Ha/Bohr for RKS and UKS (1.0e-4 at uniform alpha).
- The SAD guess, NAO basis and pseudo-hydrogen exponents solve each atom
  with its own alpha.

## The alpha_HF tables (Q47'', Q49-Q51)

Schwarz's papers (in references/) define alpha_HF as the alpha for which the
statistical total energy of the free atom equals Mann's configuration-averaged
Hartree-Fock energy (spin-restricted X-alpha, Herman-Skillman program).

- `alpha_hf` -- **Schwarz's published values** (Q50 a), transcribed in full
  into `data/tables/schwarz_alpha.json`: Phys. Rev. B 5, 2466 (1972) Table I,
  41 rows H-Nb (alpha_vt, alpha_HF, dE/dalpha; H has alpha_vt only), and
  Theor. Chim. Acta 34, 225 (1974) Table 1, 26 heavier atoms with their
  configurations. Every digit checked against 400-dpi renderings (Rb alpha_vt
  0.70525 from the image; the text layer was unclear). The 19 atoms the 1974
  paper omits (Sn-I, Nd-Sm, Ho-Tm, Hf-Os, Pb-At) are interpolated linearly in
  Z between the end points of their subshell sequence, as that paper
  recommends; H (no alpha_HF in the table) is the recomputed 0.97866. Each
  value carries its origin: 66 published, 19 interpolated, 1 recomputed.
- `alpha_hf_polarized` -- no published counterpart; computed by
  `tools/alpha_hf.py` (Q49, Q51): spin-polarized spherical X-alpha (Hund's
  rule) against the ROHF energy of the same configuration, H-Rn, run on one
  core (Q51). Elements that fail (ROHF configuration differs from the
  table's, time limit) are excluded and listed in the file.
- The same tool recomputes Schwarz's definition (spin-restricted X-alpha
  against the HF average-of-configuration energy from spherically rebuilt
  ROHF orbitals). He-Sc agree with the published values to 4e-5 on average,
  1.9e-4 at most (N); the systematic small deficit is the ANO-RCC basis
  sitting ~0.5 mHa above the HF limit (Ne). This is a test.

Using the spin-polarized table in UKS and the restricted one in RKS is the
consistent pairing (the restricted fit absorbs the missing spin-polarization
energy of open-shell atoms into alpha: H 0.979 vs 0.777); the choice is
explicit in the input, not automatic (Q49 b).

Per-calculation alpha by energy minimization was considered and rejected:
dE/dalpha = E_x/alpha < 0, so the energy has no minimum in alpha, and alpha is
a parameter of the functional, not a variational variable. Tuning alpha per
system by a non-energetic condition (e.g. -eps_HOMO = IP) is possible but
makes energies of different systems or geometries incomparable (ADR 0008).

## Found on the way

The UKS grid terms of the analytic forces used twice the exchange-energy
density (`2 rho_s eps(2 rho_s)` instead of `rho_s eps(2 rho_s)`) since
4f4746c: 0.46 Ha/Bohr on H2O run as UKS. Fixed; UKS = RKS forces for a closed
shell and CH3 against finite differences are now tested.
