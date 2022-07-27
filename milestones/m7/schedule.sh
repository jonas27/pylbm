#!/bin/bash

# pip install --user --upgrade numpy matplotlib mpi4py

# module load devel/python/3.10.0_gnu_11.1
# module load compiler/gnu/12.1
# module load mpi/openmpi/4.1

# list queue in server
# squeue  
# watch squeue  

# run job
# sbatch -x ./m7.job

# for nodes = 4
# for i in 1 4 9 16 25 36
# do
# ./m7.job 4 $i
# done

# for nodes = 2
# for i in 1 4 9 16
# do
# ./m7.job 2 $((i*2))
# done

# for nodes = 1
for i in 1 4 9
do
./m7.job 1 $((i*4))
done