# 0011. Element-specific X-alpha parameter

- Status: Accepted (mixing, input, forces); the alpha_HF table definition is open (Q47'')
- Date: 2026-10-04
- Decisions: Q39-Q41, Q46-Q48, Q47'

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

## alpha_HF without the original papers (Q39, Q46-Q48)

Schwarz's tables were not available; `tools/alpha_hf.py` recomputes alpha_HF
from its definition (X-alpha total energy of the atom = Hartree-Fock energy),
with pyxany's atom solver and PySCF ROHF (uncontracted ANO-RCC, checked by an
extended basis; PySCF only in the tool). PySCF's spherically averaged RHF
(`AtomSphAverageRHF`) gives open-shell energies 0.2-0.55 Ha above ROHF and
was not used.

Which X-alpha atom and which HF energy reproduce Schwarz is open: closed
shells (He, Be, Ne, Mg, Ar) agree with the values recalled from the
literature to 3e-5 in every variant; spin-restricted X-alpha against the
ROHF energy reproduces H, Li, B, F, Na, Al, Cl (0.9787 vs 0.9780, others to
3e-5) but gives C, N, O, Si, P 0.002-0.012 too high, consistent with Schwarz
using the HF average-of-configuration energy. The spin-polarised variant
(Q47') is lower for every open-shell atom (H 0.777).

## Found on the way

The UKS grid terms of the analytic forces used twice the exchange-energy
density (`2 rho_s eps(2 rho_s)` instead of `rho_s eps(2 rho_s)`) since
4f4746c: 0.46 Ha/Bohr on H2O run as UKS. Fixed; UKS = RKS forces for a closed
shell and CH3 against finite differences are now tested.
