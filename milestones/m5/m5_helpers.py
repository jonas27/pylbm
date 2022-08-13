import matplotlib.pyplot as plt
import numpy as np
from pylbm import lbm


def m5_1(x_dim, y_dim, epochs, omega, rho_in, rho_out):
    r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

    print_epoch = round(epochs / 12)
    if print_epoch == 0:
        print_epoch = 1

    velocities = []
    for _ in range(epochs):
        f_cxy = lbm.stream(f_cxy=f_cxy)
        # d_0 = lbm.density(f_cxy).sum()
        f_cxy = lbm.in_out_pressure(f_cxy=f_cxy, rho_in=rho_in, rho_out=rho_out)
        # print(d_0 - lbm.density(f_cxy).sum())
        f_cxy = lbm.bottom_wall(f_cxy=f_cxy)
        f_cxy = lbm.top_wall(f_cxy=f_cxy)
        f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
        velocities.append(u_axy)

    return velocities, f_cxy


def m5_1_fig(fig, x_dim, y_dim, velocities):
    i = 1
    for v in velocities:
        if i > 25:
            break
        ax = plt.subplot(5, 5, i)
        ax.plot(np.arange(x_dim), np.zeros((x_dim)) + y_dim - 1.5, color="black", linewidth=3.0)
        ax.plot(np.arange(x_dim), np.zeros((x_dim)) + 0.5, color="black", linewidth=3.0)
        ax.axis("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # streamplot is really slow at big grids
        strm = ax.streamplot(np.arange(x_dim), np.arange(y_dim), v[0][0, :, :].T, v[0][1, :, :].T, cmap="autumn")
        ax.set_title("Velocities in x direction (t={}) as streamplot.".format(v[1] + 1))
        i += 1
        fig.add_axes(ax)
    return fig

def m5_1_only_final(x_dim, y_dim, epochs, omega, rho_in, rho_out):
    r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

    print_epoch = round(epochs / 12)
    if print_epoch == 0:
        print_epoch = 1

    for _ in range(epochs):
        f_cxy = lbm.stream(f_cxy=f_cxy)
        # d_0 = lbm.density(f_cxy).sum()
        f_cxy = lbm.in_out_pressure(f_cxy=f_cxy, rho_in=rho_in, rho_out=rho_out)
        # print(d_0 - lbm.density(f_cxy).sum())
        f_cxy = lbm.bottom_wall(f_cxy=f_cxy)
        f_cxy = lbm.top_wall(f_cxy=f_cxy)
        f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)

    return u_axy