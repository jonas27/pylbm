import numpy as np
from matplotlib import pyplot as plt
from pylbm import lbm
from scipy.optimize import curve_fit


def m3_1(x_dim, y_dim, eps, epochs, omega):
    x = np.arange(x_dim)
    y = np.arange(y_dim)
    X, Y = np.meshgrid(x, y)
    r_mean, u_mean = 1, 0.0
    vs_emp = []

    r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=r_mean, eps=0.0)
    r_xy += eps * np.sin(2.0 * np.pi / x_dim * X).T

    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=u_mean, eps=0.0)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

    densities = []
    for e in range(epochs):
        # use max density (j dim doesn't matter)
        densities.append(r_xy[x_dim // 4, 0])
        f_cxy = lbm.stream(f_cxy=f_cxy)
        f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
        r_xy = lbm.density(f_cxy=f_cxy)

    def func(x, a, b, c, d):
        return a * np.cos(b * x) * np.exp(-c * x) + d

    popt, pcov = curve_fit(func, np.arange(epochs), densities)
    v = popt[2] / (2 * np.pi / x_dim) ** 2
    vs_emp.append(v)

    return vs_emp, densities


def m3_2(x_dim, y_dim, eps, epochs, omegas):
    x = np.arange(x_dim)
    y = np.arange(y_dim)
    X, Y = np.meshgrid(x, y)
    r_mean, u_mean = 1, 0.0

    amps_total = []
    velocities_omega_05 = []
    for omega in omegas:
        r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=r_mean, eps=0.0)

        u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=u_mean, eps=0.0)
        u_axy[0] += eps * np.sin(2.0 * np.pi / y_dim * Y).T
        f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

        amps_emp = []
        for e in range(epochs):
            # amps_emp.append(r_xy[x_dim // 4, 0])
            if omega == 0.5:
                # choose any i value, all y values
                velocities_omega_05.append([u_axy[0, 0, :], e])

            amp = u_axy[0, 0, :]
            amps_emp.append(amp.max() - amp.mean())
            # amps_emp.append(u_axy[0, 0, :].max())

            f_cxy = lbm.stream(f_cxy=f_cxy)
            f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
            r_xy = lbm.density(f_cxy=f_cxy)

        # break
        amps_emp = np.array(amps_emp)
        amps_total.append([omega, amps_emp])
    return amps_total, velocities_omega_05

    # def m3_1_fig(fig, epochs, densities_omega):
    #     ax = plt.subplot(15, 5)
    #     densities_omega = np.array(densities_omega)
    #     ax.plot(np.arange(epochs),densities_omega)
    #     ax.set_title("Density over time, omega=0.5")
    #     ax.set_xlabel("time")
    #     ax.set_ylabel("density")
    #     ax.grid(True)
    #     fig.add_axes(ax)
    #     return fig