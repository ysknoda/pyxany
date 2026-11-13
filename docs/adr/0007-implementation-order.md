# 0007. Implementation order

- Status: Accepted
- Date: 2026-10-01

## Decision

Each step ends with the full test suite passing and a local commit. Nothing is
pushed to GitHub. (Superseded by ADR 0012: the repository is published on
GitHub from 0.2.0a1 on.)

1. Commit the pending working-tree changes.
2. Flat layout and rename (ADR 0001).
3. Install PySCF locally; build Tier A tests (ADR 0003).
4. Tier B and C tests on real molecules (ADR 0003).
5. SCF checkpoint and restart (ADR 0004).
6. Bond overlap population and plots (ADR 0005).
7. Slater transition-state method (ADR 0006).
8. Profile, then performance work. Candidates, in expected order of payoff for
   100-300 atom clusters on one to a few MPI nodes:
   - atom-block sparsity screening of basis values on the grid (no dense chi);
   - angular grid pruning near nuclei;
   - reuse of the previous density / Fock matrix (shared with step 5);
   - caching Becke weights;
   - chunking the chi-gradient in the force evaluation (current memory
     high-water mark).

## Outcome of step 8 (2026-10-03)

Profiling the 107-atom Si47H60 cluster (1.07 M grid points, 966 basis
functions) rather than the candidate list decided the order. Done, all
without changing results (Q26 options A and B):

| hot spot | before | after | how |
|---|---|---|---|
| grid build (Becke weights) | 540 s | 9 s | compiled own-cell kernel |
| Hartree evaluation | 366 s | 21 s | compiled multipole summation |
| force grad V_H | 6 Hartree evaluations | 1 gradient evaluation | exact derivative kernel |
| S/T, per pair | 52 ms | 22 ms | compiled basis gradients, contiguous layout |
| S/T gradient (forces) | 622 s | 129 s | exact pair derivative (basis Hessians, Becke weights) instead of 6-point FD |
| force memory | 24 GB | memory budget | chunked over points |
| SCF start | diverged | 18 iterations | SAD initial guess |

End to end on Si47H60 (SCF + forces, one node): 2381 s -> 1875 s, peak RSS
6.9 GB; forces unchanged to 5e-10 Ha/Bohr. What remains is the SCF itself
(67 %): density and XC builds with chi recomputed per iteration because the
8 GB chi exceeds the memory budget. Reducing that needs chi sparsity
(option C) or a larger budget.

Not done: an adaptive multipole order (tried, no measurable gain once the
summation was compiled); chi sparsity and angular pruning (they change the
quadrature, so they are option C and need a separate decision).

## Rationale

Tests come before features so that a feature that breaks a physical quantity
is caught immediately.
