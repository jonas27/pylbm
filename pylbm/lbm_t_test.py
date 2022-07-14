from pylbm import lbm_t as lbm
import matplotlib.pyplot as plt
import torch

x_dim, y_dim = 100, 80
epochs = 100
top_vel = 1
omega = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0, device=device)
u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0, device=device)
f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

velocities = []
for t in range(epochs):
    velocities.append(u_axy)
    f_cxy = lbm.stream(f_cxy=f_cxy)
    f_cxy = lbm.apply_bottom_wall(f_cxy=f_cxy)
    f_cxy = lbm.apply_sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
    f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
