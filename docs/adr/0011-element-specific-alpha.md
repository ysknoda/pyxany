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
  core (Q51, the last reruns on two). All 86 elements are in the table; the
  file records for each which ROHF was used (`rohf`) and any `notes`.
- The same tool recomputes Schwarz's definition (spin-restricted X-alpha
  against the HF average-of-configuration energy from spherically rebuilt
  ROHF orbitals). Against the 66 published values: median 2.2e-5, at most
  1.9e-4 (N) apart from Tb, 3.7e-4, because Schwarz used 4f8 5d 6s2 and the
  tool the NIST ground configuration 4f9 6s2 (Q58). The small systematic
  deficit is the ANO-RCC basis sitting ~0.5 mHa above the HF limit (Ne). The
  19 interpolated values agree to 7.2e-5 (Pm). This is a test.

### Getting the ROHF into the right configuration

Unconstrained ROHF from PySCF's default guess lands in another configuration
for a number of transition metals and lanthanides (a 4s/5s/6s electron moves
into d, or 5d into 4f), and the per-run checks catch that through the d/f
electron counts. Successive remedies, each applied to what the previous one
left over:

1. A configuration-targeted guess: spherically averaged RHF orbitals dealt to
   the configuration's shells with Hund occupations, kept by the maximum
   overlap method, Newton as fallback (Q52, Q53; lanthanides Pm-Lu).
   Gd still fell into 4f7 6s2 6p-like (one d electron short).
2. SO3 symmetry with the (alpha, beta) electron count of every (l, m) irrep
   fixed (Q54-Q57): electrons cannot change l. Partly filled shells fill
   m = +l, l-1, ... (Q55); the irreps are real harmonics, so for one open-shell
   electron beside filled or half-filled shells (Gd 4f7 5d1) the determinant is
   a component of the Hund term, otherwise one fixed determinant of the
   configuration. Used for every flagged element (Q56): Fe, Co, Ni, Y, Zr, Tc,
   La, Ce, Pr, Nd, Sm, Eu, Gd, Ho, W, Re, Os, Ir.
3. A 0.5 Ha level shift, then an unshifted pass from that density (Q60): in
   the extended basis Nd alternated between two solutions 2.1 Ha apart.
4. Sm never converges: DIIS alternates between two states and Newton stalls
   within 2e-6 Ha with the orbital gradient at 0.08 (near-degenerate f orbitals
   swap). An unconverged ROHF whose last ten energies agree within 1e-5 Ha is
   kept with a note, since alpha then moves by < 1e-7 (Q62); Newton is capped
   at 50 macro cycles (Q63; 500 cost Dy 32 h for nothing).
5. Dy: the symmetric run reached a state 8.6 Ha too high and was rejected;
   the targeted-guess result is kept although its basis check is 1.04e-4,
   just over the 1e-4 limit (Q64).
6. Fe with the targeted guess passed every per-run check yet sat 0.39 Ha
   above the HF ground state (alpha_hf 6e-3 low); rerun with symmetry it
   agrees with Schwarz to 8e-5. The table writer now also flags any element
   whose recomputed alpha_hf is more than 1e-3 from the published one.

Symmetry-constrained and unconstrained ROHF are both the lowest single
determinant of the configuration, but the unconstrained one may hybridize
(s-d) and lie lower: Sc 2.1 mHa (alpha_hf_polarized 4e-5), Ho 47 mHa
(1.3e-4). alpha_hf, from the spherically averaged E_av, is unaffected. The
table mixes the two, with the method recorded per element (Q59).

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
