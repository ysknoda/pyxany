# 0008. Deferred work (backlog)

- Status: Accepted
- Date: 2026-10-01

Not in the current scope; recorded so the decision to defer is explicit.

| item | why deferred / prerequisite |
|---|---|
| **Cluster-size convergence of the example gaps** | The example clusters (5-7 core atoms) have terminator-dominated LUMOs (Loewdin 0.52-0.72) that relaxing the terminators does not remove. A size series (e.g. Si/GaAs/CdSe with 17, 29, 47, 87 core atoms) is needed before their gaps can stand for the material; cost is set by step 8 (performance). |
| Multiple initial spin guesses, tried automatically | Next stage after the current scope. |
| Effective core potentials on boundary cations of embedded clusters | Would allow anion-terminated embedded clusters (MgO6 rather than Mg19O6); needs semi-local ECP projectors on one-centre grids at each boundary site and an ECP parameter source (ADR 0010). |
| XANES / XES spectra from transition-state dipole intensities | Builds on ADR 0006 (c). |
| Correlation functional (e.g. VWN5) | Changes the code's identity as an X-alpha code; separate decision. |
| Point-group symmetry | Large effort; performance and labelling benefit only. |
| System-tuned alpha (e.g. -eps_HOMO = IP, as in optimally tuned range-separated functionals) | Physically motivated, but alpha would differ between systems and geometries, so their energies could not be compared, and forces need dalpha/dR; energy minimisation in alpha has no minimum (ADR 0011). |
| Vibrational frequencies (finite-difference Hessian of analytic forces) | Low priority for current use. |

## Resolved since this ADR was written

- **Grid-weight derivative term in the analytic forces** -- implemented
  (Becke weight derivatives plus atom-centred point motion), together with
  S/T Pulay terms taken from the matrices the SCF actually uses. Force error
  vs finite differences: H2 ~1e-3 -> 2.5e-5 Ha/Bohr; CO 2e-2 -> 1.4e-3 at
  l_max = 12, the remainder being multipole truncation (1e-4 at l_max = 16).
- **Unstable Hartree solver at high l** -- rewritten as a ratio-damped
  Green's-function recursion; the l_max cap is now the angular grid (n_a - 1)
  and the default is `auto` = n_a - 1.
- **Pseudo-hydrogen basis always single-zeta** -- terminators now receive
  the basis set's full hydrogen function list scaled by zeta(alpha)/zeta(1);
  alpha = 1 equals hydrogen exactly in every basis.
- **Element-dependent radial grid size** -- measured instead of taken from
  Treutler & Ahlrichs (1995), which was not at hand: `tools/radial_grid_sizes.py`
  finds, for all 86 elements, the n_r at which the grid error in the
  ionisation energy of the near-exact X-alpha atom matches what C-F reach at
  the input n_r (criterion chosen over total energy per electron, which is
  core-dominated and cancels in energy differences). Ratios to the second row:
  ~1.2 (row 3), ~1.5 (row 4), ~1.9 (row 5), ~2.2 (row 6), against PySCF's
  per-period defaults 1.07 / 1.20 / 1.27 / 1.33 with its own mapping.
- **Adaptive multipole order in the Hartree evaluation** -- tried in step 8
  and removed: once the summation was compiled, cutting l for far atoms gave
  no measurable gain.
- **Large-cluster SCF divergence** -- caused by the initial guess; replaced by
  a superposition of atomic densities (ADR 0007, step 8 outcome).
- **Lebedev angular grids** -- available as `grid.angular: lebedev` (ADR 0009).
- **Point-charge (Madelung) embedding** -- `embedding:` and `pyxany embed` (ADR 0010).
- **Element-specific X-alpha** -- `scf.alpha`: mapping, `alpha_hf` (Schwarz), `alpha_hf_polarised` (ADR 0011).
