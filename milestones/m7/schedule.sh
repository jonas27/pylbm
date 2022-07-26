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


for i in 1 4 9 16
do
sbatch ./m7.job 4 $i
done

# for i in 1 4 9 16 25 36
# do
# sbatch ./m7.job 4 $i
# done


#  t='/usr/bin/time -f "%U" ls 2>&1 )
# t=`(/usr/bin/time -f "%U" ls) 2>&1 `
# echo $t
# echo $t
# echo $t

# t=$( TIMEFORMAT="%R"; { time ls; } 2>&1 | awk '{print $NF}') 
# echo $t