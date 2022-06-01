"""
Notations:
f_ijc: is the probability density function with space dim i and j and velocity dim c


"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pylbm import log

TAU = 0.5

C_CA: np.array = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

W_C: np.array = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# The bounce back direction
C_REVERSED: np.array = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

NORTH = "north"
EAST = "east"
SOUTH = "south"
WEST = "west"


def rho_init(i_dim: int, j_dim: int, r_mean: float = 0.5, eps: float = 0.01) -> np.array:
    """rho_init based on dim, a mean and a deviation factor eps."""
    r_ij = eps * np.random.randn(i_dim, j_dim)
    r_ij[:, :] += r_mean
    return r_ij


def rho(f_cij: np.array) -> np.array:
    """rho is the local density."""
    r_ij = np.einsum("cij -> ij", f_cij)
    return r_ij


def local_avg_velocity_init(i_dim, j_dim, u_mean: float, eps: float):
    """local_avg_velocity_init based on dim, a mean and a deviation factor eps."""
    u_aij = eps * np.random.randn(2, i_dim, j_dim)
    u_aij[:, :] += u_mean
    return u_aij


def local_avg_velocity(f_cij: np.array, r_ij: np.array) -> np.array:
    """local_avg_velocity calculation based on f and rho."""
    u_aij = np.einsum("ac, cij->aij", C_CA.T, f_cij) / r_ij
    return u_aij


def f_eq(u_aij: np.array, r_ij: np.array) -> np.array:
    """f_eq calculates the probability equilibrium distribution function.

    The probability density function  f(r,v,t)  has a non trivial physical meaning. Therefore we suggest, at the beginning, to think of the value of  fi(r,t)  as the number of "particles" which are in the position  r  at the time  t  and that are moving in the direction  ci

    Returns:
        f_eq_ijc: equilibrium distribution function.
        u_ija: the local average velocity u(r) for testing.
    """
    cu_cij_3 = 3 * np.einsum("ac,aij->cij", C_CA.T, u_aij)
    u_ij_2 = np.einsum("aij->ij", u_aij * u_aij)
    r_cij_w = np.einsum("c,ij->cij", W_C, r_ij)
    f_eq_cij = r_cij_w * (1 + cu_cij_3 * (1 + 0.5 * cu_cij_3) - 1.5 * u_ij_2[np.newaxis, :, :])
    return f_eq_cij


def stream(f_cij: np.array) -> np.array:
    """stream shifts all values according to the velocities.

    Implementation:
        Ranges over all velocities and rolls the values.
        Starts at index 1 as it doesn't change in 0 channel.

    Rational:
        Rolling over after array ends is ok, as we simulate a large plane which in its limit is transformed into a cylinder.
    """
    for i in range(1, 9):
        f_cij[i, :, :] = np.roll(f_cij[i, :, :], shift=C_CA[i], axis=(0, 1))
    return f_cij


def make_boundries(i_dim, j_dim, north: bool, east: bool, south: bool, west: bool) -> np.array:
    b = np.zeros((i_dim, j_dim), dtype=bool)
    if north:
        b[0, :] = True
    if east:
        b[:, 0] = True
    if south:
        b[-1, :] = True
    if west:
        b[:, -1] = True
    return b


def apply_boundries(f_cij: np.array, f_cij_old: np.array, boundries: np.array) -> np.array:
    for i in range(1, 9):
        # use old values where boundry is true.
        f_cij[i, :, :] = np.where(boundries, f_cij_old[i, :, :], f_cij[i, :, :])
    return f_cij


def apply_sliding_top_boundry(f_cij: np.array, f_cij_old: np.array, velocity: float) -> np.array:
    # calc vel at sliding boundry
    rho_i_top = f_cij_old[[2, 5, 6], :, -1].sum(axis=0) * 2 + f_cij_old[[0, 1, 3], :, -1].sum(axis=0)
    # update bottom velocities
    for i in [4, 7, 8]:
        rev = C_REVERSED[i]
        f_cij[i, :, -1] = f_cij_old[rev, :, -1] - 6 * W_C[rev] * rho_i_top * velocity
    return f_cij


def collision(f_cij: np.array, omega: float) -> Tuple[np.array, np.array]:
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
    r_ij = rho(f_cij)
    u_aij = local_avg_velocity(f_cij=f_cij, r_ij=r_ij)
    f_eq_ijc = f_eq(u_aij=u_aij, r_ij=r_ij)
    f_cij += omega * (f_eq_ijc - f_cij)
    return f_cij, u_aij
