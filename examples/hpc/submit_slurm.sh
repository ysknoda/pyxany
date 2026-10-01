#!/bin/bash
#SBATCH --job-name=pyxany
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

# Keep BLAS single-threaded: pyxany parallelises over MPI ranks, and letting each
# rank spawn its own thread pool oversubscribes the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Check the resource estimate before committing the job:
#   pyxany run cluster.yaml --dry-run
srun pyxany run cluster.yaml --log "$SLURM_JOB_NAME-$SLURM_JOB_ID.out"
