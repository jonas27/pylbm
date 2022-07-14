"""
Notations:
f_ijc: is the probability density function with space dim i and j and velocity dim c

TODO: Moving from global lists to tuples could be faster.

"""

from typing import Tuple

import torch

TAU = 0.5

C_CA: torch.tensor = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

W_C: torch.tensor = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=torch.float)
# W_C: Tuple = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)

# The bounce back direction
C_REVERSED: torch.tensor = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6])
# C_REVERSED: Tuple = (0, 3, 4, 1, 2, 7, 8, 5, 6)

NORTH = "north"
EAST = "east"
SOUTH = "south"
WEST = "west"


def density_init(x_dim: int, y_dim: int, r_mean: float = 0.5, eps: float = 0.01, device="cpu") -> torch.tensor:
    """rho_init based on dim, a mean and a deviation factor eps."""
    r_ij = eps * torch.randn(x_dim, y_dim, device=device)
    r_ij[:, :] += r_mean
    return r_ij


def density(f_cxy: torch.tensor) -> torch.tensor:
    """rho is the local density."""
    r_xy = torch.einsum("cxy -> xy", f_cxy)
    return r_xy


def local_avg_velocity_init(x_dim, y_dim, u_mean: float, eps: float, device="cpu"):
    """local_avg_velocity_init based on dim, a mean and a deviation factor eps."""
    u_aij = eps * torch.randn(2, x_dim, y_dim, device=device)
    u_aij[:, :] += u_mean
    return u_aij


def local_avg_velocity(f_cxy: torch.tensor, r_xy: torch.tensor) -> torch.tensor:
    """local_avg_velocity calculation based on f and rho."""
    u_aij = torch.einsum("ac, cxy->axy", C_CA.T.float(), f_cxy) / r_xy
    return u_aij


def f_eq(u_axy: torch.tensor, r_xy: torch.tensor) -> torch.tensor:
    """f_eq calculates the probability equilibrium distribution function.

    The probability density function  f(r,v,t)  has a non trivial physical meaning. Therefore we suggest, at the beginning, to think of the value of  fi(r,t)  as the number of "particles" which are in the position  r  at the time  t  and that are moving in the direction  ci

    Returns:
        f_eq_cxy: equilibrium distribution function.
        u_ija: the local average velocity u(r) for testing.
    """
    cu_cxy_3 = 3 * torch.einsum("ac,axy->cxy", C_CA.T.float(), u_axy)
    u_xy_2 = torch.einsum("axy->xy", u_axy * u_axy)
    r_cxy_w = torch.einsum("c,xy->cxy", W_C, r_xy)
    f_eq_cxy = r_cxy_w * (1 + cu_cxy_3 * (1 + 0.5 * cu_cxy_3) - 1.5 * u_xy_2[None, :, :])
    return f_eq_cxy


def stream(f_cxy: torch.tensor) -> torch.tensor:
    """stream shifts all values according to the velocities.

    Implementation:
        Ranges over all velocities and rolls the values.
        Starts at index 1 as it doesn't change in 0 channel.

    Rational:
        Rolling over after tensor ends is ok, as we simulate a large plane which in its limit is transformed into a cylinder.
    """
    for i in range(1, 9):
        f_cxy[i, :, :] = torch.roll(f_cxy[i, :, :], shifts=C_CA[i].tolist(), dims=(0, 1))
    return f_cxy


def apply_bottom_wall(f_cxy: torch.tensor) -> torch.tensor:
    """m4: in the couette flow we have no side walls and thus need to copy all values up."""
    # should be torch roll
    # argument ohne torch roll: im steady state ist es egal
    f_cxy[2, :, 1] = f_cxy[4, :, 0]
    f_cxy[5, :, 1] = f_cxy[7, :, 0]
    f_cxy[6, :, 1] = f_cxy[8, :, 0]
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
    # calc vel at sliding wall
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
    r_xy = density(f_cxy)

    r_xy_in = torch.full((1, r_xy.shape[1] - 2), rho_in)
    r_xy_out = torch.full((1, r_xy.shape[1] - 2), rho_out)

    u_axy = local_avg_velocity(f_cxy=f_cxy, r_xy=r_xy)
    f_eq_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)

    f_eq_in = f_eq(u_axy=u_axy[:, -2:-1, 1:-1], r_xy=r_xy_in)
    f_cxy[:, :1, 1:-1] = f_eq_in + (f_cxy[:, -2:-1, 1:-1] - f_eq_cxy[:, -2:-1, 1:-1])

    f_eq_out = f_eq(u_axy=u_axy[:, 1:2, 1:-1], r_xy=r_xy_out)
    f_cxy[:, -1:, 1:-1] = f_eq_out + (f_cxy[:, 1:2, 1:-1] - f_eq_cxy[:, 1:2, 1:-1])
    return f_cxy


def collision(f_cxy: torch.tensor, omega: float) -> Tuple[torch.tensor, torch.tensor]:
    """collision adds the collision term \Delta_i to the probability distribution function f.

        Here we use the Bhatnagar-Gross-Krook-Operator (BGK-Operator):
        {\displaystyle \Delta _{i}={\frac {1}{\tau }}(f_{i}^{\mathrm {eq} }-f_{i})}{\displaystyle \Delta _{i}={\frac {1}{\tau }}(f_{i}^{\mathrm {eq} }-f_{i})}.

    Source:
        https://de.wikipedia.org/wiki/Lattice-Boltzmann-Methode

    Args:
        Omega:  Das statistische Gewicht {\displaystyle \Omega }\Omega  ist ein Maß für die Wahrscheinlichkeit eines bestimmten Makrozustandes.
                Die Relaxationszeit {\displaystyle \tau }\tau  bestimmt, wie schnell sich das Fluid dem Gleichgewicht nähert und hängt somit direkt von der Viskosität des Fluids ab. Der Wert {\displaystyle f_{i}^{\mathrm {eq} }}{\displaystyle f_{i}^{\mathrm {eq} }} ist die lokale Gleichgewichtsfunktion, welche die Boltzmannverteilung approximiert. (wiki)
                omega = 1 / tau wobei tau

    """
    r_xy = density(f_cxy)
    u_axy = local_avg_velocity(f_cxy=f_cxy, r_xy=r_xy)
    f_eq_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)
    f_cxy += omega * (f_eq_cxy - f_cxy)
    return f_cxy, u_axy


def reynolds(y_dim, omega, top_vel) -> float:
    nu = 1 / 3 * (1 / omega - 1 / 2)
    return (top_vel * y_dim) / (nu)
