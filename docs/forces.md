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

The analytic expressions are the exact gradient of the *continuum* energy
functional. The computed energy is a quadrature of it on a grid whose points and
Becke weights both move with the nuclei, contributing

```
Σ_k [ dw_k/dR_B · G(r_k) + w_k · ∇G(r_k)·dr_k/dR_B ],   G = ρ(v_nuc + V_H/2 + ε_xc)
```

For an exact quadrature this vanishes; what is left is the derivative of the
quadrature error, and it falls as the grid is refined. Its rigid-translation
component is removed exactly (`forces.project_translation`, on by default), which
restores `Σ_A F_A = 0` to machine precision.

Measured against finite differences:

**H₂ at 0.90 Å** (max |F| ≈ 5.3e-2 Ha/Bohr)

| grid | l_max | max \|analytic − FD\| |
|---|---|---|
| 30 × 12 | 4 | 3.9e-4 |
| 40 × 14 | 4 | 2.3e-4 |
| 50 × 16 | 6 | 5.6e-5 |
| 60 × 18 | 6 | 2.0e-5 |

**H₂O near equilibrium** (max |F| ≈ 3.5e-2 Ha/Bohr)

| grid | l_max | \|Σ F\| unprojected | \|Σ F\| projected | max \|analytic − FD\| |
|---|---|---|---|---|
| 25 × 10 | 4 | 1.9e-2 | 4.9e-18 | 6.6e-3 |
| 30 × 12 | 6 | 7.8e-3 | 2.5e-29 | 3.5e-3 |

A symmetric diatomic is a special case: the grid-derivative contributions cancel
by symmetry, which is why H₂ is one to two orders of magnitude more accurate
than a polyatomic at the same grid. **Do not calibrate expectations on H₂.**

### Choosing

- **Geometry optimisation**: analytic forces, with `grid.n_r ≥ 40` and
  `grid.n_a ≥ 14`. Do not set `optimise.f_max` tighter than the force error at
  your grid — check it once with `pyxany forces --compare`.
- **Publication-quality gradients on a small system**: `forces.method:
  finite_difference`.
- **Screening or a first relaxation**: analytic at the default grid is fine.

The one identity that always holds exactly is the translational sum rule; it is
asserted in `tests/test_forces.py` and is not a substitute for the comparison
above.
