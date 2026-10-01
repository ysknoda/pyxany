# 0008. Deferred work (backlog)

- Status: Accepted
- Date: 2026-10-01

Not in the current scope; recorded so the decision to defer is explicit.

| item | why deferred / prerequisite |
|---|---|
| **Grid-weight derivative term in the analytic forces** (Johnson, Gill & Pople, J. Chem. Phys. 98, 5612 (1993)) | Found 2026-10-01: omitted today. Only its translation part is removed by projection; the rest is ~3.6e-4 Ha/Bohr for H2O but ~1e-2 Ha/Bohr for CO at n_r=40 (vs FD), above the default f_max = 1e-3, so CO cannot be optimised. Tier B CO bond test is `xfail(strict=True)` until this lands. Priority to be set by the author. |
| **Pseudo-hydrogen basis is always single-zeta** | Found 2026-10-01: a terminator gets one 1s STO calibrated to the SZ hydrogen exponent whatever `basis.set` is, while a real H in `dz` gets two. So at alpha = 1 a pseudo-H equals H only in `sz`; in Si5H12/dz the inner Si charge is -0.42 e with pseudo-H vs -0.03 e with H, and the LUMO weight on terminators 0.000 vs 0.189. A fix (scale every H function of the chosen basis by zeta(alpha)/zeta(1)) changes physics and needs the author's decision. |
| **Element-specific X-alpha alpha** (e.g. Schwarz alpha_HF) | Needs an inter-atomic mixing rule and the full Schwarz table from the source paper; breaks Tier A PySCF comparison (ADR 0002). |
| Multiple initial spin guesses, tried automatically | Next stage after the current scope. |
| Point-charge (Madelung) embedding for ionic clusters | Next stage; relevant to MgO-type clusters. |
| XANES / XES spectra from transition-state dipole intensities | Builds on ADR 0006 (c). |
| Correlation functional (e.g. VWN5) | Changes the code's identity as an X-alpha code; separate decision. |
| Point-group symmetry | Large effort; performance and labelling benefit only. |
| Vibrational frequencies (finite-difference Hessian of analytic forces) | Low priority for current use. |
