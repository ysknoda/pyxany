# 0008. Deferred work (backlog)

- Status: Accepted
- Date: 2026-10-01

Not in the current scope; recorded so the decision to defer is explicit.

| item | why deferred / prerequisite |
|---|---|
| **Element-specific X-alpha alpha** (e.g. Schwarz alpha_HF) | Needs an inter-atomic mixing rule and the full Schwarz table from the source paper; breaks Tier A PySCF comparison (ADR 0002). |
| Multiple initial spin guesses, tried automatically | Next stage after the current scope. |
| Point-charge (Madelung) embedding for ionic clusters | Next stage; relevant to MgO-type clusters. |
| XANES / XES spectra from transition-state dipole intensities | Builds on ADR 0006 (c). |
| Correlation functional (e.g. VWN5) | Changes the code's identity as an X-alpha code; separate decision. |
| Point-group symmetry | Large effort; performance and labelling benefit only. |
| Vibrational frequencies (finite-difference Hessian of analytic forces) | Low priority for current use. |
