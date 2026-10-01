# 0008. Deferred work (backlog)

- Status: Accepted
- Date: 2026-10-01

Not in the current scope; recorded so the decision to defer is explicit.

| item | why deferred / prerequisite |
|---|---|
| **Element-dependent radial grid size** | n_r is the same for every element; Ga/As cores at n_r = 30 carry ~0.5 Ha of Hartree quadrature error per atom (1s test charge: 1.9e-2 Ha at n_r = 30, 9.5e-4 at 60). Treutler & Ahlrichs (1995) tabulate per-row sizes but the paper is not at hand, so the sizes are to be *measured* on pyxany's own grid against its near-exact atomic solver (target error per electron), with PySCF's per-period defaults as a sanity check. Pending the author's go-ahead (Q22). |
| Adaptive multipole order in the Hartree evaluation | The default l_max = n_a - 1 costs ~2.5x per SCF iteration on GaAs4; far atoms only need low l (tails decay as r^-(l+1)). Planned for step 8 (performance). |
| **Element-specific X-alpha alpha** (e.g. Schwarz alpha_HF) | Needs an inter-atomic mixing rule and the full Schwarz table from the source paper; breaks Tier A PySCF comparison (ADR 0002). |
| Multiple initial spin guesses, tried automatically | Next stage after the current scope. |
| Point-charge (Madelung) embedding for ionic clusters | Next stage; relevant to MgO-type clusters. |
| XANES / XES spectra from transition-state dipole intensities | Builds on ADR 0006 (c). |
| Correlation functional (e.g. VWN5) | Changes the code's identity as an X-alpha code; separate decision. |
| Point-group symmetry | Large effort; performance and labelling benefit only. |
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
