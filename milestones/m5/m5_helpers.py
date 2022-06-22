from pylbm import lbm


def m5_1(x_dim, y_dim, epochs, omega):
    r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

    print_epoch = round(epochs / 12)
    if print_epoch == 0:
        print_epoch = 1

    velocities = []
    for _ in range(epochs):
        f_cxy = lbm.stream(f_cxy=f_cxy)
        f_cxy = lbm.apply_bottom_wall(f_cxy=f_cxy)
        f_cxy = lbm.apply_top_wall(f_cxy=f_cxy)
        f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
        velocities.append(u_axy)
    return velocities
