import numpy as np
from pylbm.torchlbm import lbm_t as lbm

x_dim = 300
y_dim = 300
epochs = 100000
omega = 1.7
top_vel = 0.1

path = "./vels.npy"

r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

velocities = []
for e in range(epochs):
    f_cxy = lbm.stream(f_cxy=f_cxy)
    f_cxy = lbm.apply_sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
    f_cxy = lbm.bottom_wall(f_cxy=f_cxy)
    f_cxy = lbm.left_wall(f_cxy=f_cxy)
    f_cxy = lbm.right_wall(f_cxy=f_cxy)
    f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)
    if e - 1 % 1000 == 0:
        velocities.append(u_axy.cpu().numpy())

np.save(path, velocities)
