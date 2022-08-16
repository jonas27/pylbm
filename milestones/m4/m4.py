import numpy as np
from matplotlib import pyplot as plt
from pylbm import lbm


def m4_1(x_dim, y_dim, epochs, omega, top_vel):
    r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=1.0)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_init=0.0)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

    velocities = []
    for t in range(epochs):
        if t % 100 == 0:
            velocities.append(u_axy)
        f_cxy = lbm.stream(f_cxy=f_cxy)
        f_cxy = lbm.bottom_wall(f_cxy=f_cxy)
        f_cxy = lbm.sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
        f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)

    np.save("./velocities.npy", velocities)
