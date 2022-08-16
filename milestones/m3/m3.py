import numpy as np
from pylbm import lbm
from scipy.optimize import curve_fit


def m3_1():
    epochs = 3000
    eps = 0.1
    x_dim, y_dim = 30, 30
    path = "./m3-1.npy"

    x = np.arange(x_dim)
    y = np.arange(y_dim)
    X, Y = np.meshgrid(x, y)
    r_init, u_init = 0.5, 0.0

    oms = [1.7, 1, 0.5]
    densities_oms = []
    for o in oms:
        r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_init)
        r_xy += eps * np.sin(2.0 * np.pi / x_dim * X).T

        u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_init=u_init)
        f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

        densities = []
        for e in range(epochs):
            densities.append(r_xy[x_dim // 4, 0])
            f_cxy = lbm.stream(f_cxy=f_cxy)
            f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=o)
            r_xy = lbm.local_density(f_cxy=f_cxy)
        densities = np.array(densities)
        densities_oms.append(densities)
    np.save(path, densities_oms)


def m3_2():
    epochs = 3000
    eps = 0.1
    x_dim, y_dim = 30, 30
    x = np.arange(x_dim)
    y = np.arange(y_dim)
    X, Y = np.meshgrid(x, y)
    omegas = np.arange(0.1, 1.91, 0.2).round(decimals=1)
    r_init, u_init = 1, 0.0

    amps_total = []
    velocities_print = []
    for omega in omegas:
        velocities_omega = []
        r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_init)

        u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_init=u_init)
        u_axy[0] += eps * np.sin(2.0 * np.pi / y_dim * Y).T
        f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

        amps_emp = []
        for e in range(epochs):
            # amps_emp.append(r_xy[x_dim // 4, 0])
            if e % (epochs // 10) == 0:
                # choose any i value, all y values
                velocities_omega.append(u_axy[0, 0, :])

            amp = u_axy[0, 0, :]
            # amps_emp.append(amp.max() - amp.mean())
            amps_emp.append(u_axy[0, 0, :].max())

            f_cxy = lbm.stream(f_cxy=f_cxy)
            f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
        velocities_print.append(velocities_omega)

        amps_emp = np.array(amps_emp)
        amps_total.append(amps_emp)

    np.save('./amps_total.npy', amps_total)
    np.save('./velocities_print.npy', velocities_print)


def m3_2_updated():
    epochs = 3000
    eps = 0.1
    x_dim, y_dim = 300, 300
    x = np.arange(x_dim)
    y = np.arange(y_dim)
    X, Y = np.meshgrid(x, y)
    omegas = np.arange(0.1, 1.91, 0.2).round(decimals=1)[:2]
    r_init, u_init = 1, 0.0

    amps_total = []
    velocities_print = []
    for omega in omegas:
        velocities_omega = []
        r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_init)

        u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_init=u_init)
        u_axy[0] += eps * np.sin(2.0 * np.pi / y_dim * Y).T
        f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

        amps_emp = []
        for e in range(epochs):
            # amps_emp.append(r_xy[x_dim // 4, 0])
            if e % (epochs // 10) == 0:
                # choose any i value, all y values
                velocities_omega.append(u_axy[0, 0, :])

            amp = u_axy[0, 0, :]
            # amps_emp.append(amp.max() - amp.mean())
            amps_emp.append(u_axy[0, 0, :].max())

            f_cxy = lbm.stream(f_cxy=f_cxy)
            f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
        velocities_print.append(velocities_omega)

        amps_emp = np.array(amps_emp)
        amps_total.append(amps_emp)

    np.save('./amps_total_updated.npy', amps_total)
    np.save('./velocities_print_updated.npy', velocities_print)