"""
Notations:
f_ijc: is the probability density function with space dim i and j and velocity dim c

TODO: Moving from global lists to tuples could be faster.

"""

from typing import Tuple

import torch

T_TYPE = torch.double

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_CA: torch.tensor = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], device=DEVICE)

W_C: torch.tensor = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=T_TYPE, device=DEVICE)

C_REVERSED: torch.tensor = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=DEVICE)


def local_density_init(x_dim: int, y_dim: int, r_init: float = 0.5) -> torch.tensor:
    r_ij = torch.zeros((x_dim, y_dim), device=DEVICE, dtype=T_TYPE) + r_init
    return r_ij


def local_density(f_cxy: torch.tensor) -> torch.tensor:
    r_xy = torch.einsum("cxy -> xy", f_cxy)
    return r_xy


def local_avg_velocity_init(x_dim, y_dim, u_init: float):
    u_aij = torch.zeros((2, x_dim, y_dim), device=DEVICE, dtype=T_TYPE) + u_init
    return u_aij


def local_avg_velocity(f_cxy: torch.tensor, r_xy: torch.tensor) -> torch.tensor:
    u_aij = torch.einsum("ac, cxy->axy", C_CA.T.double(), f_cxy) / r_xy
    return u_aij


def f_eq(u_axy: torch.tensor, r_xy: torch.tensor) -> torch.tensor:
    cu_cxy_3 = 3 * torch.einsum("ac,axy->cxy", C_CA.T.double(), u_axy)
    u_xy_2 = torch.einsum("axy->xy", u_axy * u_axy)
    r_cxy_w = torch.einsum("c,xy->cxy", W_C, r_xy)
    f_eq_cxy = r_cxy_w * (1 + cu_cxy_3 * (1 + 0.5 * cu_cxy_3) - 1.5 * u_xy_2[None, :, :])
    return f_eq_cxy


def stream(f_cxy: torch.tensor) -> torch.tensor:
    for i in range(1, 9):
        f_cxy[i, :, :] = torch.roll(f_cxy[i, :, :], shifts=C_CA[i].tolist(), dims=(0, 1))
    return f_cxy


def bottom_wall(f_cxy: torch.tensor) -> torch.tensor:
    f_cxy[2, :, 1] = f_cxy[4, :, 0]
    f_cxy[5, :, 1] = f_cxy[7, :, 0]
    f_cxy[6, :, 1] = f_cxy[8, :, 0]
    return f_cxy


def top_wall(f_cxy: torch.tensor) -> torch.tensor:
    f_cxy[4, :, -2] = f_cxy[2, :, -1]
    f_cxy[7, :, -2] = f_cxy[5, :, -1]
    f_cxy[8, :, -2] = f_cxy[6, :, -1]
    return f_cxy


def left_wall(f_cxy: torch.tensor) -> torch.tensor:
    f_cxy[1, 1, :] = f_cxy[3, 0, :]
    f_cxy[5, 1, :] = f_cxy[7, 0, :]
    f_cxy[8, 1, :] = f_cxy[6, 0, :]
    return f_cxy


def right_wall(f_cxy: torch.tensor) -> torch.tensor:
    f_cxy[3, -2, :] = f_cxy[1, -1, :]
    f_cxy[7, -2, :] = f_cxy[5, -1, :]
    f_cxy[6, -2, :] = f_cxy[8, -1, :]
    return f_cxy


def apply_sliding_top_wall(f_cxy: torch.tensor, velocity: float) -> torch.tensor:
    r_x_top = f_cxy[[0, 1, 3], 1:-1, -2].sum(axis=0) + 2.0 * (f_cxy[2, 1:-1, -1] + f_cxy[5, 2:, -1] + f_cxy[6, :-2, -1])
    f_cxy[4, 1:-1, -2] = f_cxy[2, 1:-1, -1]
    f_cxy[7, 1:-1, -2] = f_cxy[5, 2:, -1] - 6.0 * W_C[5] * r_x_top * velocity
    f_cxy[8, 1:-1, -2] = f_cxy[6, :-2, -1] + 6.0 * W_C[6] * r_x_top * velocity
    return f_cxy


def apply_sliding_top_wall_simple(f_cxy: torch.tensor, velocity: float = None) -> torch.tensor:
    """for incompressible fluids with  we can say that at the wall we have a density of -1/6, 0 and 1/6"""
    f_cxy[4, :, -2] = f_cxy[2, :, -1]
    f_cxy[7, :, -2] = f_cxy[5, :, -1] - 1 / 6.0 * velocity
    f_cxy[8, :, -2] = f_cxy[6, :, -1] + 1 / 6.0 * velocity
    return f_cxy


def in_out_pressure(f_cxy: torch.tensor, rho_in: float, rho_out: float) -> torch.tensor:
    r_xy = local_density(f_cxy)

    r_xy_in = torch.full((1, r_xy.shape[1] - 2), rho_in, device=DEVICE, dtype=T_TYPE)
    r_xy_out = torch.full((1, r_xy.shape[1] - 2), rho_out, device=DEVICE, dtype=T_TYPE)

    u_axy = local_avg_velocity(f_cxy=f_cxy, r_xy=r_xy)
    f_eq_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)

    f_eq_in = f_eq(u_axy=u_axy[:, -2:-1, 1:-1], r_xy=r_xy_in)
    f_cxy[:, :1, 1:-1] = f_eq_in + (f_cxy[:, -2:-1, 1:-1] - f_eq_cxy[:, -2:-1, 1:-1])

    f_eq_out = f_eq(u_axy=u_axy[:, 1:2, 1:-1], r_xy=r_xy_out)
    f_cxy[:, -1:, 1:-1] = f_eq_out + (f_cxy[:, 1:2, 1:-1] - f_eq_cxy[:, 1:2, 1:-1])
    return f_cxy


def collision(f_cxy: torch.tensor, omega: float) -> Tuple[torch.tensor, torch.tensor]:
    r_xy = local_density(f_cxy)
    u_axy = local_avg_velocity(f_cxy=f_cxy, r_xy=r_xy)
    f_eq_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)
    f_cxy += omega * (f_eq_cxy - f_cxy)
    return f_cxy, u_axy


def run():
    x_dim, y_dim = 300, 300
    epochs = 100000
    top_vel = 1.7
    omega = 0.1

    r_xy = local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=1.0)
    u_axy = local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_init=0.0)
    f_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)

    velocities = []
    for t in range(epochs):
        # velocities.append(u_axy)
        f_cxy = stream(f_cxy=f_cxy)
        f_cxy = bottom_wall(f_cxy=f_cxy)
        f_cxy = apply_sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
        f_cxy, u_axy = collision(f_cxy=f_cxy, omega=omega)


if __name__ == "__main__":
    run()
