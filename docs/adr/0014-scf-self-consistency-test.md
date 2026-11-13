# 0014. SCF convergence requires self-consistency (commutator test)

- Status: Accepted
- Date: 2026-11-13
- Decisions: Q82-Q85 (D1-D4 of the session that found it)
- Roadmap item: F8 (ADR 0012)

## Context

The first CI run of 0.2.0a1 failed one slow test on Linux x86-64:
`test_core_hole_localizes_on_one_of_two_equivalent_atoms` (N2, 1s hole on
atom 1, DZ, n_r 30 / n_a 12 / l_max 11) did not converge in 200 iterations.
The same test passed on macOS arm64 in 97 iterations.

A per-iteration trace (`tools/debug_ts_convergence.py`) showed:

- The hole localization is not involved: from iteration 2 on the hole sits in
  an eigenvector with Loewdin weight 0.96 on atom 1 throughout.
- The SCF converges along a slow mode: the commutator halves only every ~50
  iterations (500 iterations: 5e-8). Neither the DIIS subspace (4, 6, 10, 20
  vectors) nor the multipole truncation (l_max 4, 11, 15) changes the rate.
  Without an equivalent atom (H2O O1s) the same kind of run reaches 1e-8 in
  ~110 iterations.
- The convergence test used only `|dE| < e_tol` and `d_rms < d_tol`, where
  `d_rms` compares the density with the one from the *extrapolated* Fock
  matrix. A stalled DIIS can hold that density still while F and P do not
  commute. On macOS this happened at iteration 97 (eps 10 meV off the
  converged value); on Linux it did not, hence the platform difference.

At the shipped defaults (e_tol 1e-6, d_tol 1e-5) the N2 core-hole run stopped
after 30 iterations, reported as converged, with -eps **65 meV** too large
(at e_tol 1e-7 / d_tol 1e-6: 49 iterations, 45 meV).

### Which commutator

The first implementation measured `F P S - S P F` in the full AO basis. The
slow suite then failed on the shipped MgO6H example: its energy and density
settle by iteration 15 (energy within 8e-9 Ha of the 500-iteration value),
but that commutator stays at 1.4e-4 for all 500 iterations. 9 of its 132
overlap eigenvalues lie below `s_threshold` (5e-4; smallest 6.8e-5) and are
dropped by the canonical orthogonalization, and the commutator component in
those directions is out of the SCF's reach. Projected onto the retained space,
`X^T (F P S - S P F) X` with `X^T S X = 1`, it falls to 3.8e-7 by iteration
15. The test therefore uses the projected commutator.

Measured relation for N2 1s (projected commutator; first iteration below the
threshold; eps relative to the 500-iteration value -14.997803748 Ha):

| commutator rms | iteration | eps deviation |
|---|---|---|
| 1e-4 | 61 | 30.5 meV |
| 3e-5 | 111 | 9.1 meV |
| 1e-5 | 189 | 2.4 meV |
| 3e-6 | 259 | 0.87 meV |
| 1e-6 | 307 | 0.39 meV |
| 1e-7 | 450 | 0.01 meV |

At the shipped defaults, ordinary molecules stop with the projected commutator
already below 1e-5 and take the same iterations with or without the new test,
to the same energy: H2O 7.8e-7, CO 2.3e-6, N2 4.6e-8, CH3 (UKS) 8.6e-6.

## Decision

- **D1 (Q82):** new key `scf.c_tol`, default **1e-5**. Convergence requires
  `|dE| < e_tol`, `d_rms < d_tol` and the rms of the projected commutator
  of the *unextrapolated* F with the P it was built from below `c_tol` (UKS:
  the larger of the two channels). It is computed every iteration, EDIIS
  included, and stored in the SCF history as `commutator`.
- **D2 (Q83):** `scf.max_iter` defaults to 300 for a transition-state run and
  to 100 otherwise, when the input does not set it; an explicit value is kept.
  A run whose energy and density have settled but whose commutator has not is
  diagnosed as a slow mode, with the advice to raise `scf.max_iter` rather
  than loosen `c_tol`.
- **D3 (Q84):** the N2 slow test runs 400 iterations at c_tol 1e-6 and checks
  eps against the 500-iteration value to 2e-5 Ha; a second slow test checks
  that the shipped defaults land within 5 meV of it (measured: 201
  iterations, -2.37 meV).
- **D4 (Q85):** release as 0.2.0a2. The CI slow tests + MPI job runs only when
  the workflow is started by hand (`workflow_dispatch`), not on every push.

## Consequences

- Transition-state runs on equivalent atoms take ~200 iterations; speeding up
  the slow mode is roadmap item A9.
- Results of earlier transition-state runs with a core hole on one of several
  equivalent atoms may be off by tens of meV and should be repeated.
- The Pulay DIIS error vector is still the unprojected commutator, so with
  dropped directions it carries a constant component. Projecting it too may
  help DIIS on near-linearly-dependent bases but changes every SCF trajectory;
  not done here (roadmap item A10).
- Found on the way: the UKS diagonalization ignores `scf.s_threshold` and
  always uses the default 5e-4 (roadmap item F9).
