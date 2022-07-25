import numpy as np
from matplotlib import pyplot as plt
from pylbm import lbm


def m4_1(x_dim, y_dim, epochs, omega, top_vel):
    r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

    velocities = []
    for t in range(epochs):
        velocities.append(u_axy)
        f_cxy = lbm.stream(f_cxy=f_cxy)
        f_cxy = lbm.bottom_wall(f_cxy=f_cxy)
        f_cxy = lbm.apply_sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
        f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)

    return velocities


def m4_1_fig(fig, x_dim, y_dim, num_plots, velocities):
    top_wall, left_wall, right_wall = False, False, False
    top_moving_wall, bottom_wall = True, True
    i_plot = 1
    for i in range(0, len(velocities), int(len(velocities) / num_plots)):
        # for v in velocities:
        v = velocities[i]
        if i_plot > 25:
            break
        ax = plt.subplot(5, 5, i_plot)
        # ax.margins(0.05)
        if top_wall:
            ax.plot(np.arange(x_dim), np.zeros((x_dim)) + y_dim - 0.5, color="black", linewidth=3.0)
        if top_moving_wall:
            ax.plot(np.arange(x_dim), np.zeros((x_dim)) + y_dim - 1.5, color="orange", linewidth=5.0)
        if bottom_wall:
            ax.plot(np.arange(x_dim), np.zeros((x_dim)) + 0.5, color="black", linewidth=3.0)
        if left_wall:
            ax.plot(np.zeros(y_dim) - 1, np.arange((y_dim)), color="black", linewidth=3.0)
        if right_wall:
            ax.plot(np.zeros(y_dim) + x_dim, np.arange((y_dim)), color="black", linewidth=3.0)
        ax.set_xticks(np.arange(0, x_dim, x_dim / 10))
        ax.set_yticks(np.arange(0, y_dim, y_dim / 10))
        # ax.grid(True)
        ax.axis("equal")
        # streamplot is really slow at big grids
        strm = ax.streamplot(np.arange(x_dim), np.arange(y_dim), v[0, :, :].T, v[1, :, :].T, cmap="autumn")
        # fig.colorbar(strm.lines)
        ax.set_title("t={}".format(i))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        i_plot += 1
        fig.add_axes(ax)
    return fig


if __name__ == "__main__":
    x_dim, y_dim = 10, 8
    m4_1(x_dim=x_dim, y_dim=y_dim, epochs=5, omega=0.5)
