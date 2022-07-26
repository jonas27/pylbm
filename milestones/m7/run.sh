#!/bin/bash

# pip install --user --upgrade numpy matplotlib mpi4py

module load devel/python/3.10.0_gnu_11.1
module load compiler/gnu/12.1
module load mpi/openmpi/4.1

sbatch -x ./m7.job