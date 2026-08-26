# Running at scale

## Before submitting

```bash
dvxa run cluster.yaml --dry-run
```

reports grid size, the memory one rank needs for the chi matrix, how many atom
pairs survive screening, and the per-iteration Hartree work. It is much cheaper
than discovering a node's memory limit an hour into a job.

## Parallelism

The DV grid is decomposed across MPI ranks. Each rank builds its own slice of
the chi matrix and accumulates its share of every grid integral; the small
`n_basis × n_basis` matrices are reduced and replicated, and the eigenproblem
is solved once on rank 0 and broadcast. Console output and file writes happen
on rank 0 only.

```bash
mpirun -n 128 dvxa run cluster.yaml
```

Keep BLAS single-threaded so ranks do not oversubscribe the node:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`tests/test_mpi.py` asserts that the parallel energy and forces match the serial
ones to 1e-8.

## Long jobs

```bash
nohup setsid mpirun -n 128 dvxa run big.yaml --log big.out &
```

`--log` mirrors everything to a file, and `setsid` detaches the job so it
survives the session ending. Sample Slurm and PBS scripts are in
`examples/hpc/`.

Exit codes let a script branch on the outcome: `0` success, `1` input error,
`2` SCF not converged, `3` runtime error, `4` optimisation not converged.

## Cost and how to control it

Per SCF iteration, roughly:

| term | scaling | knob |
|---|---|---|
| chi matrix (cached) | N_grid × n_basis | `grid.n_r`, `grid.n_a`, `memory_budget_mb` |
| Hartree potential | N_atoms × N_grid × (l_max+1)² | `grid.poisson_l_max` |
| exchange-correlation | N_grid × n_basis² | grid density |
| diagonalisation | n_basis³ | basis set |

The Hartree solve dominates for a typical cluster. It uses a Becke multi-centre
multipole expansion — each atom's share of the density is expanded in spherical
harmonics and its radial Poisson equation solved analytically — which is linear
in the number of atoms rather than the quadratic pair sum a direct evaluation
would need, and introduces no softening parameter.

Two things that matter at scale:

- **`memory_budget_mb`** decides whether the chi matrix is cached or the grid is
  walked in chunks. Caching is faster; chunking bounds memory at the cost of one
  extra chi evaluation per SCF iteration, and gives numerically identical
  results (`tests/test_chunking.py` asserts agreement to 1e-9 on energies and
  forces). `--dry-run` reports which will happen and at what size.

  The **force evaluation** builds chi and its gradient explicitly — the gradient
  block is three times the size of chi — so it, not the SCF, is a run's memory
  high-water mark. Set `forces.enabled: false` for a single-point energy on a
  system where that is tight.
- **`screening_tol`** skips atom pairs whose basis functions cannot overlap.
  For an extended structure this turns the `O(N²)` pair loop over `S` and `T`
  into a linear one; for a compact cluster where every atom is close to every
  other, expect little benefit.

## Checkpointing

Set `scf.checkpoint_every` to write the density matrix periodically, so a job
killed at a wall-clock limit loses at most that many iterations.
