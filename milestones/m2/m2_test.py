import cProfile
import logging

import numpy as np
from pylbm import lbm

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()


def test_collision():
    eps = 0.01
    r_mean, u_mean = 0.5, 0.5
    i_dim, j_dim = 5, 10
    omega = 2
    epochs = 300
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(x_dim=i_dim, y_dim=j_dim, u_mean=u_mean, eps=eps)
    f_cij = lbm.f_eq(u_axy=u_aij, r_xy=r_ij)

    print(u_aij)
    for _ in range(epochs):
        f_cij = lbm.stream(f_cxy=f_cij)
        f_cij, u_aij = lbm.collision(f_cxy=f_cij, omega=omega)
    print(u_aij)


def test_uniform_density():
    """test what happens when the local density is increased in the center

    Results:
        min local density decreases over time
        max local density first decreases then increases over time over initial max
        the density mainly follows a diagonal direction upwards direction and is rotated at the grid end
    """
    np.set_printoptions(precision=4, suppress=True)
    eps = 0.0
    r_mean, u_mean = 0.5, 0.5
    i_dim, j_dim = 8, 8
    omega = 0.1
    epochs = 300
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    # increase local density in the center
    r_ij[int(i_dim / 2 - 1) : int(i_dim / 2 + 1), int(j_dim / 2 - 1) : int(j_dim / 2 + 1)] = 0.9
    u_aij = lbm.local_avg_velocity_init(x_dim=i_dim, y_dim=j_dim, u_mean=u_mean, eps=eps)
    f_cij = lbm.f_eq(u_axy=u_aij, r_xy=r_ij)
    for e in range(epochs):
        if e % 10 == 0:
            print(r_ij)
            print(r_ij.min())
            print(r_ij.max())
            print("==================================")
        f_cij = lbm.stream(f_cxy=f_cij)
        r_ij = lbm.rho(f_cij)
        f_cij, u_aij = lbm.collision(f_cxy=f_cij, omega=omega)
