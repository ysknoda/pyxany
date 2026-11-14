# Forces

`pyxany` computes nuclear forces two ways.

## Analytic (default)

Hellmann-Feynman plus Pulay basis-derivative terms:

```
dE/dR_A = − Z_A ∫ ρ(r) (r − R_A)/|r − R_A|³ dr                    Hellmann-Feynman
          − 2 Σ_{μ∈A,ν} P_μν ∫ D_μ (ĥ_eff χ_ν) dr                  Pulay, operator
          + 2 Σ_{μ∈A,ν} W_μν ∫ D_μ χ_ν dr                          Pulay, orthonormality
          + dE_nn/dR_A                                             nuclear repulsion
```

with `D_μ = ∇χ_μ`, `ĥ_eff = −½∇² + v_nuc + V_H + v_xc`, and `W` the
energy-weighted density matrix. Every basis type supplies an exact analytic
gradient *and* an exact analytic Laplacian (§ `pyxany.basis.types`), so no finite
differences appear anywhere in the derivation.

Cost: roughly one extra SCF iteration.

## Finite difference (reference)

Central differences of the total energy, `6N` SCF runs per geometry. Slow, but
it is by construction the exact gradient of the number the code actually
computes, so it is the arbiter.

```bash
pyxany forces molecule.yaml --compare      # both, side by side
```

## Accuracy, measured

The computed energy is a quadrature on a grid whose points and Becke weights
move with the nuclei. The analytic forces include that motion (the
grid-derivative term `Σ_k [dw_k/dR_B · G(r_k) + w_k ∇G(r_k)·dr_k/dR_B]`,
`G = ρ(v_nuc + V_H/2 + ε_xc)`) and the S/T Pulay terms of the matrices the SCF
uses, so they are the derivative of the computed energy. Two different errors
therefore have to be kept apart:

1. **Implementation**: analytic against central finite differences of the same
   energy on the same grid. H₂O (DZ, n_r 30 / n_a 12): max 1.3e-4 Ha/Bohr; NH₂
   (UKS, element-specific α, three point charges, Lebedev, pruning) 2.7e-4 at
   (30, 12) and 4.7e-5 at (45, 18). This is set by the FD step and the SCF
   tolerance, not by the grid.
2. **Grid**: the forces of one grid against a much finer one. This is what an
   optimization's `f_max` must be compared with, and it is dominated by the
   **radial** quadrature.

H₂O (DZ) against n_r 80 / n_a 26 (max |F| there 2.3e-2 Ha/Bohr; measured 2026-11-14):

| grid (n_r, n_a) | \|ΔE\| (Ha) | max \|ΔF\| (Ha/Bohr) |
|---|---|---|
| 30, 12 (default before 0.2.0a3) | 2.0e-3 | 2.2e-3 |
| 45, 12 (default since ADR 0015) | 3.4e-4 | 1.7e-4 |
| 60, 12 | 8.7e-5 | 2.3e-5 |
| 30, 18 | 2.0e-3 | 2.1e-3 |

More angular points change nothing; more radial points do. At n_r 30 the force
error (2e-3) exceeded the default `optimize.f_max` (1e-3), which is why the
default is n_r 45 since ADR 0015 (1.7e-4, a factor 6 below f_max).
`pyxany forces --compare` prints both numbers: the FD difference and a grid
error estimate from n_r + 15, with a warning when `f_max` is below the latter.

The translational sum rule `Σ_A F_A = 0` holds after
`forces.project_translation` (on by default); it is necessary, not sufficient.

### Choosing

- **Geometry optimization**: analytic forces with `grid.n_r ≥ 45`. Do not set
  `optimize.f_max` tighter than the grid error estimate that
  `pyxany forces --compare` prints for your system.
- **Publication-quality gradients on a small system**: `forces.method:
  finite_difference`.
- **Screening or a first relaxation**: analytic at the default grid is fine.

The one identity that always holds exactly is the translational sum rule; it is
asserted in `tests/test_forces.py` and is not a substitute for the comparison
above.
