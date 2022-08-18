# pylbm

Implementation of the Lattice-Boltzmann-Method (LBM) for a D29Q-model. 
The code is implemented in Numpy and can be used with MPI.
However, there is an alternative Pytorch implementation which can be used to run on GPUs.

The code is part of the course *High-Performance Computing: Fluid Mechanics with Python* at the University of Freiburg. 

## Installation
Install this package into your pip env by executing the below line in the repo root dir:

```pip install -e .```

## Naming
The naming convention is adopted from the lecture. 
The first index is the rolling index.
x and y are the dimensions of the physical 2D space.
E.g: "f_cxy" means the c=velocity space and x and y.

## Numpy funcs
Important numpy funcs

### einsum
Makes the sum over all dot products.
helps to calculate all densities over all gridpoints.
np.einsum: sum probabilities over all velocities
np.einsum('c_ij -> ij', f_cij)
f_cij[:] = np.einsum('ij,c->ij', np.ones(()),W_C)
shoud be the same as
f_cij[:] = w_c[:,np.newaxis, np.newaxis]

u_aij = np.einsum('cij,ca->aij',f_cij,c_ca)

rho_ij = np.einsum('cij -> ij', f_cij)

## Lid driven cavity problem
![](https://github.com/jonas27/pylbm/blob/d4e7bc1c4a4b967b3517914cad728d4b4d3b81ac/milestones/m6/m6.gif?raw=true)

## Parallization results
![](https://github.com/jonas27/pylbm/blob/754a690538eafa0f065327e84305ff308d3eb957/milestones/final/img/m7-mlups.png?raw=true)