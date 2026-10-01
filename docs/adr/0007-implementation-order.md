# 0007. Implementation order

- Status: Accepted
- Date: 2026-10-01

## Decision

Each step ends with the full test suite passing and a local commit. Nothing is
pushed to GitHub.

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

## Rationale

Tests come before features so that a feature that breaks a physical quantity
is caught immediately.
