import numpy as np
from matplotlib import pyplot as plt
from pylbm import lbm


def m6_1(x_dim, y_dim, epochs, omega, top_vel):
    r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

    d_0 = r_xy.sum()
    velocities = []
    for t in range(epochs):
        velocities.append(u_axy)
        f_cxy = lbm.stream(f_cxy=f_cxy)

        f_cxy = lbm.apply_sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
        f_cxy = lbm.apply_bottom_wall(f_cxy=f_cxy)
        f_cxy = lbm.left_wall(f_cxy=f_cxy)
        f_cxy = lbm.right_wall(f_cxy=f_cxy)

        f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)

        # print(d_0 - lbm.density(f_cxy).sum())

    return velocities


# m6_1(50,50,100,1.,1)
