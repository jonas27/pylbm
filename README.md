# high-performance-python-lbm
A Lattice Boltzmann method implementation for the university course high performance python - fluid computing.

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
