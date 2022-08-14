import numpy as np
from pylbm.torchlbm import lbm_t as lbm

x_dim = 300
y_dim = 300
omega = 0.5
epochs = 100000
rho_in = 1.01
rho_out = 0.99
path = "./velocities.npy"

path = "./vel_t_100000.npy"

r_xy = lbm.density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)

print_epoch = round(epochs / 12)
if print_epoch == 0:
    print_epoch = 1

for _ in range(epochs):
    f_cxy = lbm.stream(f_cxy=f_cxy)
    f_cxy = lbm.in_out_pressure(f_cxy=f_cxy, rho_in=rho_in, rho_out=rho_out)
    f_cxy = lbm.bottom_wall(f_cxy=f_cxy)
    f_cxy = lbm.top_wall(f_cxy=f_cxy)
    f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=omega)

v_final = u_axy.cpu().numpy()
np.save(path, v_final)
