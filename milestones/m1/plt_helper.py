from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pylbm import lbm


def set_ax(ax: plt.Axes, data: np.array) -> plt.Axes:
    """set_ax uses the data to populate the plt.Axes.
    It also sets ticks and labels etc.
    """
    row_labels = list([i for i in range(data.shape[0])])
    column_labels = list([j for j in range(data.shape[1])])
    # put the major ticks at the middle of each cell
    ax.pcolor(data, cmap=plt.cm.Reds)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(column_labels, minor=False)
    ax.set_yticklabels(row_labels, minor=False)
    ax.set_ylabel("axis 0, first  index")
    ax.set_title("axis 1, second index")
    return ax


def plot_streaming(fig, streaming_direction: int) -> List[plt.Axes]:
    eps = 0.01
    r_mean, u_mean = 0.5, 0.5
    i_dim, j_dim = 5, 10
    r_ij = lbm.density_init(x_dim=i_dim, y_dim=j_dim, r_mean=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(x_dim=i_dim, y_dim=j_dim, u_mean=u_mean, eps=eps)
    f_ijc = lbm.f_eq(u_axy=u_aij, r_xy=r_ij)
    epochs = 3
    axs = fig.subplots(nrows=epochs, ncols=1)

    for ax in axs:
        data = f_ijc[:, :, streaming_direction]
        set_ax(ax, data=data)
        f_ijc = lbm.stream(f_cxy=f_ijc)
    return fig
