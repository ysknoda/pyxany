# Running at scale

## Before submitting

```bash
pyxany run cluster.yaml --dry-run
```

reports grid size, the memory one rank needs for the chi matrix, how many atom
pairs survive screening, and the per-iteration Hartree work. It is much cheaper
than discovering a node's memory limit an hour into a job.

## Parallelizm

The DV grid is decomposed across MPI ranks. Each rank builds its own slice of
the chi matrix and accumulates its share of every grid integral; the small
`n_basis × n_basis` matrices are reduced and replicated, and the eigenproblem
is solved once on rank 0 and broadcast. Console output and file writes happen
on rank 0 only.

```bash
mpirun -n 128 pyxany run cluster.yaml
```

Keep BLAS single-threaded so ranks do not oversubscribe the node:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`tests/test_mpi.py` asserts that the parallel energy and forces match the serial
ones to 1e-8.

## Long jobs

```bash
nohup setsid mpirun -n 128 pyxany run big.yaml --log big.out &
```

`--log` mirrors everything to a file, and `setsid` detaches the job so it
survives the session ending. Sample Slurm and PBS scripts are in
`examples/hpc/`.

Exit codes let a script branch on the outcome: `0` success, `1` input error,
`2` SCF not converged, `3` runtime error, `4` optimization not converged.

## Cost and how to control it

Per SCF iteration, roughly:

| term | scaling | knob |
|---|---|---|
| chi matrix (cached) | N_grid × n_basis | `grid.n_r`, `grid.n_a`, `memory_budget_mb` |
| Hartree potential | N_atoms × N_grid × (l_max+1)² | `grid.poisson_l_max` |
| exchange-correlation | N_grid × n_basis² | grid density |
| diagonalization | n_basis³ | basis set |

The Hartree solve dominates for a typical cluster. It uses a Becke multi-center
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

  The **force evaluation** walks the grid in chunks within the same
  `memory_budget_mb` (chi, its gradient and v*chi: five doubles per point and
  basis function), so a 107-atom cluster whose full chi would take 8 GB and
  whose gradient 24 GB computes forces in the same budget as its SCF; the split
  changes the forces by < 1e-12 (`tests/test_forces.py`).

  With numba installed, the Becke cell weights, their gradient and the Hartree
  multipole summation run as compiled, threaded kernels (identical to the numpy
  reference to 1e-12, `tests/test_backends.py`). On a 107-atom Si cluster
  (1.07 M points) they took the grid build from 540 s to 9 s and one Hartree
  evaluation from 366 s to 21 s; for one GaAs4 evaluation 26 s -> 2.4 s.
- **`screening_tol`** skips atom pairs whose basis functions cannot overlap.
  For an extended structure this turns the `O(N²)` pair loop over `S` and `T`
  into a linear one; for a compact cluster where every atom is close to every
  other, expect little benefit.

## Checkpointing

`run`, `opt`, `forces` and `scan` write the SCF state to
`<output.dir>/<prefix>_scf.npz` at the end of every SCF, and every
`scf.checkpoint_every` iterations if that is > 0:

```yaml
scf:
  checkpoint_every: 10
```

A job killed at the wall-clock limit is resumed with

```bash
pyxany run cluster.yaml --restart results/cluster_scf.npz
```

The file holds the density matrix (both spins for UKS), the orbitals and the
iteration count, plus fingerprints of the system and of the geometry. Same
system and geometry: the run resumes where it stopped. Same system, other
geometry: the stored density is only the initial guess (a warning says so).
Anything else -- other atoms, charge, multiplicity, basis or alpha -- is
refused. DIIS history is not stored (it would multiply the file size by ~10
and is rebuilt within a few iterations), so the first iterations after a
restart converge a little more slowly than an uninterrupted run would.

Files are written by rank 0 to a temporary name and renamed into place, so a
kill during the write leaves the previous checkpoint intact. `opt` and `scan`
also start every SCF from the previous point's density.
