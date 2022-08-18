#!/bin/bash

# pip install --user --upgrade numpy matplotlib mpi4py

# for nodes = 2
# not possible for 50 72
# 32 is not quadratic
for i in 2 8 18 32
do
./m7.job 2 $i
done

# for nodes = 4
# 16 is not quadratic
for i in 1 4 9 16 25 36 
do
./m7.job 4 $i
done

# for nodes = 6
# can't use 6 nodes
# for i in 6 24
# do
# ./m7.job 6 $i
# done

# for nodes = 8
# can't use 8 nodes
# for i in 2 18
# do
# ./m7.job 8 $i
# done

# one node not working!!! 
# for nodes = 1
# for i in 1 4 9
# do
# ./m7.job 1 $((i*4))
# done