import numpy as np
from matplotlib import pyplot as plt
from pylbm import lbm

def m4_1():
    epochs = 2001
    i_dim, j_dim = 25, 25
    print_epoch = 60



    top_wall, left_wall, right_wall = False, False, False
    top_moving_wall, bottom_wall = True, True

    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=1.0, eps=0.)
    u_aij = lbm.local_avg_velocity_init(i_dim=i_dim, j_dim=j_dim, u_mean=0.0, eps=0.)
    f_cij = lbm.f_eq(u_aij=u_aij, r_ij=r_ij)

    axes = []
    prints = []
    for t in range(epochs):
        f_cij_old = f_cij.copy()
        f_cij = lbm.stream(f_cij=f_cij)
        if top_wall: f_cij = lbm.apply_top_wall(f_cij=f_cij, f_cij_old=f_cij_old)
        if bottom_wall: f_cij = lbm.apply_bottom_wall(f_cij=f_cij, f_cij_old=f_cij_old)
        if left_wall: f_cij = lbm.apply_left_wall(f_cij=f_cij, f_cij_old=f_cij_old)
        if right_wall: f_cij = lbm.apply_right_wall(f_cij=f_cij, f_cij_old=f_cij_old)
        f_cij = lbm.apply_sliding_top_wall(f_cij=f_cij, f_cij_old=f_cij_old, velocity=1)
        f_cij, u_aij = lbm.collision(f_cij=f_cij, omega=omega)




    if t%print_epoch == 1:
        prints.append([u_aij[0,:,:].T, u_aij[1,:,:].T, t])




def m4_1_fig(fig, x_dim, y_dim, prints):
    top_wall, left_wall, right_wall = False, False, False
    top_moving_wall, bottom_wall = True, True
    i = 1
    for p in prints:
        if i > 25:
            break
        ax = plt.subplot(5, 5, i)
        # ax.margins(0.05)
        if top_wall:
            ax.plot(np.arange(x_dim), np.zeros((x_dim)) + y_dim - 0.5, color="black", linewidth=3.0)
        if top_moving_wall:
            ax.plot(np.arange(x_dim), np.zeros((x_dim)) + y_dim - 0.5, color="orange", linewidth=3.0)
        if bottom_wall:
            ax.plot(np.arange(x_dim), np.zeros((x_dim)) + 0.5, color="black", linewidth=3.0)
        if left_wall:
            ax.plot(np.zeros(y_dim) - 1, np.arange((y_dim)), color="black", linewidth=3.0)
        if right_wall:
            ax.plot(np.zeros(y_dim) + x_dim, np.arange((y_dim)), color="black", linewidth=3.0)
        ax.set_xticks(np.arange(0, x_dim + 1, x_dim / 10))
        ax.set_yticks(np.arange(0, y_dim + 1, y_dim / 10))
        # ax.grid(True)
        ax.axis("equal")
        # streamplot is really slow at big grids
        ax.streamplot(np.arange(x_dim), np.arange(y_dim), p[0], p[1], color="blue")
        ax.set_title("epoch is {}".format(p[2] + 1))
        i += 1
        fig.add_axes(ax)
    return fig
