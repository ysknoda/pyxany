# 0002. Uniform X-alpha parameter (alpha = 0.7)

- Status: Accepted
- Date: 2026-10-01

## Context

`scf.alpha` is a single value applied to every atom and every grid point. The
DV-Xalpha convention (Adachi and co-workers) is alpha = 0.7 for all elements.
The alternative is element-specific values such as Schwarz's alpha_HF
(Phys. Rev. B 5, 2466 (1972)), which fit atomic Hartree-Fock exchange energies.

Element-specific alpha raises a question with no unique answer -- which alpha
applies between atoms (e.g. Becke-weighted mixing) -- and cannot be reproduced
by PySCF, which would remove the strict cross-code check of ADR 0003.

## Decision

Keep a single uniform alpha, default 0.7, configurable via `scf.alpha`. All
validation runs use a uniform alpha.

## Consequences

Strict Tier A comparison with PySCF (`1.05 * Slater exchange` for alpha = 0.7)
stays possible. Element-specific alpha is deferred to ADR 0008; if added, the
Schwarz table must be transcribed in full from the original paper (not from
memory) and the inter-atomic mixing rule stated explicitly.
