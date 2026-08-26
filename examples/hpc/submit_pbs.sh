#!/bin/bash
#PBS -N dvxa
#PBS -l select=2:ncpus=64:mpiprocs=64
#PBS -l walltime=24:00:00
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit 1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpirun -n 128 dvxa run cluster.yaml --log dvxa.out
