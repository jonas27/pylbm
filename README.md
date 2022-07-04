# pylbm

low reynolds number means very viscos fluid

## Setup
Create a conda environemnt:

```conda create --name high python=3.9 && conda activate high```

>  mpi4py can be install via conda `conda install -c conda-forge mpi4py`

Then install this package into env high:

```pip install -e .```

## Naming
name variable as <name_dim>
eg: "c_ca"

## Numpy funcs

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

## From Lecture (23.5)
Deadline: init june
We have to show that our viscocity vs theoretical viscocity (picture 2).

$y=a(t)/a(0)$

$x= t$

$curve = e^(-v*t*k_y ^2)$

shearwave decay is nothing else than testing streaming and collision

## next deadline
Couette flow is super important because we will paralyze it.

### Last Paralization