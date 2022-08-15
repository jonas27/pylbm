#!/bin/bash

# pip install --user --upgrade numpy matplotlib mpi4py

# for nodes = 2
# not possible for
for i in 2 8 18 50 72
do
./m7.job 2 $i
done

# for nodes = 4
# for i in 1 4 9 25 36
# do
# ./m7.job 4 $i
# done

# ./m7.job 4 25
# ./m7.job 4 36

# for nodes = 8
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