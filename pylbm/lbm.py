"""
This module provides all the functions necessary to run all milestones.

Notations:
    _cxy: is the probability density function with velocity dim c and space dim x and y.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from mpi4py import MPI
from numpy.lib.format import dtype_to_descr, magic

REPO_DIR = str(Path(__file__).absolute().parents[1])

""" The particle speed vector for each velocity.
    It is a 9x2 matrix.
"""
C_CA = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

""" The weight factors of each velocity for the distribution function.
    It is a 9D vector.
"""
W_C = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

""" The bounce back directions corresponding to each velocity direction.
    It is a 9D vector.
"""
C_REVERSED = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def local_density_init(x_dim: int, y_dim: int, r_init: float) -> np.array:
    """local_density_init initializes the local densities denoted by $\rho$.

    Args:
        - x_dim: The dimension of the x axis.
        - y_dim: The dimension of the y axis.
        - r_init: Initializes all values of $\rho_{x,y}$ to r_init

    Returns:
        - r_ij: The initialized local density. Dimension: x_dim x y_dim.
    """
    r_xy = np.zeros((x_dim, y_dim)) + r_init
    return r_xy


def local_density(f_cxy: np.array) -> np.array:
    """local_density calculates the local density based on the PDF by summing over the velocities.

    Args:
        - f_cxy: The PDF with dimension c,x and y.
            c is the velocity dim and x and y the spatial dim.

    Returns:
        - r_xy: The local density matrix. Dimension: x_dim x y_dim.
    """
    r_xy = np.einsum("cxy -> xy", f_cxy)
    return r_xy


def local_avg_velocity_init(x_dim: int, y_dim: int, u_init: float) -> np.array:
    """local_avg_velcotiy_init inits the local avg densities denoted by $u$.

    Args:
        - x_dim: The dimension of the x axis.
        - y_dim: The dimension of the y axis.
        - u_init: Initializes all values of $u_{a,x,y}$ to u_init

    Returns:
        - u_axy: The initialized local average velocity. Dimension: 2 x x_dim x y_dim.
    """
    u_axy = np.zeros((2, x_dim, y_dim)) + u_init
    return u_axy


def local_avg_velocity(f_cxy: np.array, r_xy: np.array) -> np.array:
    """local_avg_velocity calculates the local density based on the PDF by summing over the velocities.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.

    Returns:
        - r_xy: The local density matrix. Dimension: x_dim x y_dim.
    """
    u_aij = np.einsum("ac, cxy->axy", C_CA.T, f_cxy) / r_xy
    return u_aij


def f_eq(u_axy: np.array, r_xy: np.array) -> np.array:
    """f_eq calculates the equilibrium PDF.

    The PDF $f(r,v,t)$ has a non trivial physical meaning. Therefore I suggest, at the beginning, to think of the value of $f(r,t)$ as the number of "particles" which are in the position $r$ at the time $t$ and that are moving in the direction $c(i)$

    Args:
        u_axy: The local avgerage velocities.
        r_xy: The local densities.

    Returns:
        f_eq_cxy: equilibrium PDF.
    """
    cu_cxy_3 = 3 * np.einsum("ac,axy->cxy", C_CA.T, u_axy)
    u_xy_2 = np.einsum("axy->xy", u_axy * u_axy)
    r_cxy_w = np.einsum("c,xy->cxy", W_C, r_xy)
    f_eq_cxy = r_cxy_w * (1 + cu_cxy_3 * (1 + 0.5 * cu_cxy_3) - 1.5 * u_xy_2[np.newaxis, :, :])
    return f_eq_cxy


def stream(f_cxy: np.array) -> np.array:
    """stream shifts all values according to the particle speed vector for each velocity (C_CA).


    Ranges over all velocity directions and rolls the values. Starts at index 1 as the 0 channel doesn't change.
    Rational: Rolling over after array ends is ok, as I simulate a large plane which in its limit is transformed into a cylinder.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.

    Returns:
        - f_cxy: The PDF after the streaming.
    """
    for i in range(1, 9):
        f_cxy[i, :, :] = np.roll(f_cxy[i, :, :], shift=C_CA[i], axis=(0, 1))
    return f_cxy


def bottom_wall(f_cxy: np.array) -> np.array:
    """bottom_wall reflects the values from the bottom wall in the opposite velocity directions.

    This step has to be done after the streaming and before the collision step.
    Optimally, it shoud be implemented using the np.roll function in the opposite direction,
    but because we are in a steady state this can be omitted and instead we can simply copy all values.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.

    Returns:
        - f_cxy: The PDF after applying the bottom wall.
    """
    f_cxy[2, :, 1] = f_cxy[4, :, 0]
    f_cxy[5, :, 1] = f_cxy[7, :, 0]
    f_cxy[6, :, 1] = f_cxy[8, :, 0]
    return f_cxy


def left_wall(f_cxy: np.array) -> np.array:
    """left_wall reflects the values from the left wall in the opposite velocity directions.

    This step has to be done after the streaming and before the collision step.
    Optimally, it shoud be implemented using the np.roll function in the opposite direction,
    but because we are in a steady state this can be omitted and instead we can simply copy all values.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.

    Returns:
        - f_cxy: The PDF after applying the left wall.
    """
    f_cxy[1, 1, :] = f_cxy[3, 0, :]
    f_cxy[5, 1, :] = f_cxy[7, 0, :]
    f_cxy[8, 1, :] = f_cxy[6, 0, :]
    return f_cxy


def right_wall(f_cxy: np.array) -> np.array:
    """right_wall reflects the values from the right wall in the opposite velocity directions.

    This step has to be done after the streaming and before the collision step.
    Optimally, it shoud be implemented using the np.roll function in the opposite direction,
    but because we are in a steady state this can be omitted and instead we can simply copy all values.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.

    Returns:
        - f_cxy: The PDF after applying the right wall.
    """
    f_cxy[3, -2, :] = f_cxy[1, -1, :]
    f_cxy[7, -2, :] = f_cxy[5, -1, :]
    f_cxy[6, -2, :] = f_cxy[8, -1, :]
    return f_cxy


def top_wall(f_cxy: np.array) -> np.array:
    """top_wall reflects the values from the top wall in the opposite velocity directions.

    This step has to be done after the streaming and before the collision step.
    Optimally, it shoud be implemented using the np.roll function in the opposite direction,
    but because we are in a steady state this can be omitted and instead we can simply copy all values.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.

    Returns:
        - f_cxy: The PDF after applying the top wall.
    """
    f_cxy[4, :, -2] = f_cxy[2, :, -1]
    f_cxy[7, :, -2] = f_cxy[5, :, -1]
    f_cxy[8, :, -2] = f_cxy[6, :, -1]
    return f_cxy


def sliding_top_wall(f_cxy: np.array, velocity: float) -> np.array:
    """sliding_top_wall implements the sliding top lid.

    The lid velocity is constantly applied to the top of the grid.
    This step has to be done after the streaming and before the collision step.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.
        - velocity: The velocity at the top lid.

    Returns:
        - f_cxy: The PDF after applying the top sliding wall.
    """
    r_top = f_cxy[[0, 1, 3], :, -2].sum(axis=0) + 2.0 * (f_cxy[2, :, -1] + f_cxy[5, :, -1] + f_cxy[6, :, -1])
    f_cxy[4, :, -2] = f_cxy[2, :, -1]
    f_cxy[7, :, -2] = f_cxy[5, :, -1] - 6.0 * W_C[5] * r_top * velocity
    f_cxy[8, :, -2] = f_cxy[6, :, -1] + 6.0 * W_C[6] * r_top * velocity
    return f_cxy


def sliding_top_wall_simple(f_cxy: np.array, velocity: float = None) -> np.array:
    """sliding_top_wall_simple implements the sliding top lid.

    The lid velocity is constantly applied to the top of the grid.
    This step has to be done after the streaming and before the collision step.
    Here, I approximate the density at the lid to 1/6 and multiply it by the velocity.

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.
        - velocity: The velocity at the top lid.

    Returns:
        - f_cxy: The PDF after applying the top sliding wall.
    """
    """for incompressible fluids with we can say that at the wall we have a density of -1/6, 0 and 1/6"""
    f_cxy[4, :, -2] = f_cxy[2, :, -1]
    f_cxy[7, :, -2] = f_cxy[5, :, -1] - 1 / 6.0 * velocity
    f_cxy[8, :, -2] = f_cxy[6, :, -1] + 1 / 6.0 * velocity
    return f_cxy


def in_out_pressure(f_cxy: np.array, rho_in: float, rho_out: float) -> np.array:
    """in_out_pressure is used for the Poiseuille flow in M5."""
    r_xy = local_density(f_cxy)

    r_xy_in = np.full((1, r_xy.shape[1] - 2), rho_in)
    r_xy_out = np.full((1, r_xy.shape[1] - 2), rho_out)

    u_axy = local_avg_velocity(f_cxy=f_cxy, r_xy=r_xy)
    f_eq_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)

    f_eq_in = f_eq(u_axy=u_axy[:, -2:-1, 1:-1], r_xy=r_xy_in)
    f_cxy[:, :1, 1:-1] = f_eq_in + (f_cxy[:, -2:-1, 1:-1] - f_eq_cxy[:, -2:-1, 1:-1])

    f_eq_out = f_eq(u_axy=u_axy[:, 1:2, 1:-1], r_xy=r_xy_out)
    f_cxy[:, -1:, 1:-1] = f_eq_out + (f_cxy[:, 1:2, 1:-1] - f_eq_cxy[:, 1:2, 1:-1])
    return f_cxy


def collision(f_cxy: np.array, omega: float) -> Tuple[np.array, np.array]:
    """collision adds the collision term \Delta_i to the probability distribution function f.

        Here we use the Bhatnagar-Gross-Krook-Operator (BGK-Operator):
        {\displaystyle \Delta _{i}={\frac {1}{\tau }}(f_{i}^{\mathrm {eq} }-f_{i})}{\displaystyle \Delta _{i}={\frac {1}{\tau }}(f_{i}^{\mathrm {eq} }-f_{i})}.

    Source:
        https://de.wikipedia.org/wiki/Lattice-Boltzmann-Methode

    Args:
        - f_cxy: The PDF with dimension c,x and y. c is the velocity dim and x and y the spatial dim.
        - omega: the collision frequency. It can be thought of as the static weight as a measure for the probability for a certain macro state. It directly influences $\tau$ which defines how fast the fluid reaches the equilibrium is dependent on the viscocity. Omega is defined as $\omega = 1 / \tau$

    Returns:
        f_cxy: PDF after the collision.
        u_ija: the local average velocity $u(r)$ for testing and plotting.
    """
    r_xy = local_density(f_cxy)
    u_axy = local_avg_velocity(f_cxy=f_cxy, r_xy=r_xy)
    f_eq_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)
    f_cxy += omega * (f_eq_cxy - f_cxy)
    return f_cxy, u_axy


def reynolds(y_dim: int, omega: float, top_vel: float) -> float:
    """reynolds calculates the reynolds number for a given simulation.

    Args:
        - y_dim: The largest physical dim in the simulation.
        - omega: The collision frequency.
        - top_vel: The velocity of the moving top lid.

    Returns:
        - the reynolds number.
    """
    nu = 1 / 3 * (1 / omega - 1 / 2)
    return (top_vel * y_dim) / (nu)


def sync_shifts(cartcomm: MPI.Cartcomm) -> Tuple:
    """sync_shifts defines how the cartcomm is shifted

    Args:
        - cartcomm: the cartesian communicator from the openMPI lib.

    Returns:
        - shifts: The shifts for the cartesian communicator.
    """
    sL, dL = cartcomm.Shift(0, -1)
    sR, dR = cartcomm.Shift(0, 1)
    sD, dD = cartcomm.Shift(1, 1)
    sU, dU = cartcomm.Shift(1, -1)
    shifts = (sL, dL, sR, dR, sU, dU, sD, dD)
    return shifts


def sync_f(cartcomm: MPI.Cartcomm, shifts: Tuple, f_cxy: np.array) -> np.array:
    """sync_f syncs the pdf calculation across multiple processes.

    Syncing can be achieved by multithreading and thus does not require multiple CPUs.

    Args:
        - cartcomm: The cartesian communicator from the openMPI lib.
        - shifts: The tuple of shifts for the cartesian communicator.
        - f_cxy: The PDF.

    Returns:
        - f_cxy: The PDF with defined send and receive buffers.

    """
    sL, dL, sR, dR, sU, dU, sD, dD = shifts

    recvbuf = f_cxy[:, -1, :].copy()
    cartcomm.Sendrecv(f_cxy[:, 1, :].copy(), dest=dL, recvbuf=recvbuf, source=sL)
    f_cxy[:, -1, :] = recvbuf

    recvbuf = f_cxy[:, 0, :].copy()
    cartcomm.Sendrecv(f_cxy[:, -2, :].copy(), dest=dR, recvbuf=recvbuf, source=sR)
    f_cxy[:, 0, :] = recvbuf

    recvbuf = f_cxy[:, :, -1].copy()
    cartcomm.Sendrecv(f_cxy[:, :, 1].copy(), dest=dU, recvbuf=recvbuf, source=sU)
    f_cxy[:, :, -1] = recvbuf

    recvbuf = f_cxy[:, :, 0].copy()
    cartcomm.Sendrecv(f_cxy[:, :, -2].copy(), dest=dD, recvbuf=recvbuf, source=sD)
    f_cxy[:, :, 0] = recvbuf

    return f_cxy


def save_fig(fig, name):
    """convient function to save img to be used in latex."""
    path = REPO_DIR + "/milestones/final/img/" + name
    fig.savefig(path)


def save_mpiio(comm: MPI.Cartcomm, fn, g_kl):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm
        MPI communicator.
    fn : str
        File name.
    g_kl : array_like
        Portion of the array on this MPI processes. This needs to be a
        two-dimensional array.
    """

    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({"descr": dtype_to_descr(g_kl.dtype), "fortran_order": False, "shape": (np.asscalar(nx), np.asscalar(ny))})
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += " "
    arr_dict_str += "\n"
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny * local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode("latin-1"))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety + offsetx) * mpitype.Get_size(), filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()


def run_m7():
    """run_m7 runs the lid driven cavity problem from milestone 7."""

    logger = logging.getLogger("pylbm")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    parser = argparse.ArgumentParser(description="graphdriver")
    parser.add_argument("-d", default=False, action="store_true", help="Debug: Run with logging.debug")
    parser.add_argument("--lid_vel", default="0.1", help="The lid velocity used in the simulation.", type=float)
    parser.add_argument("--omega", default="1.7", help="The omega/collision frequency used in the simulation.", type=float)
    parser.add_argument("--density", default="1.0", help="The initial local density used in the simulation.", type=float)
    parser.add_argument("--time", default="300000", help="The number of time steps.", type=int)
    parser.add_argument("--velocity", default="0.0", help="The initial local average velocity used in the simulation.", type=float)
    parser.add_argument("--x_dim", default="300", help="The x dimension used in the simulation.", type=int)
    parser.add_argument("--y_dim", default="300", help="The y dimension used in the simulation.", type=int)
    argss = parser.parse_args()

    if argss.d:
        logger.setLevel(logging.DEBUG)

    logger.debug(argss)

    org_x_dim = argss.x_dim
    org_y_dim = argss.y_dim
    epochs = argss.time
    omega = argss.omega
    top_vel = argss.lid_vel
    r_init = argss.density
    u_init = argss.velocity

    # get comm_world
    comm = MPI.COMM_WORLD
    # get numbers of cpus
    size = comm.Get_size()
    # calc the number of x_sects
    x_sects = int(np.floor(np.sqrt(size)))
    # calc the number of y_sects depending on the x_sects and size
    y_sects = int(size / x_sects)
    # calc the x dimension of each section
    x_dim = org_x_dim // x_sects
    # calc the y dimension of each section
    y_dim = org_y_dim // y_sects

    # create new cartcomm with sects, no periods and no reordering.
    cartcomm = comm.Create_cart(dims=(x_sects, y_sects), periods=(False, False), reorder=False)

    # Init values
    r_xy = local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_init)
    u_axy = local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_init=u_init)
    f_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)
    # Get static shifts
    shifts = sync_shifts(cartcomm)

    for e in range(epochs):
        if e % 1000 == 0:
            logger.debug("Current epoch is: {}".format(e))
        # sync the PDF via cartcomm
        f_cxy = sync_f(cartcomm, shifts, f_cxy)
        f_cxy = stream(f_cxy=f_cxy)
        rank = cartcomm.Get_rank()
        coords = cartcomm.Get_coords(rank)
        # if sect is at the top outer wall apply sliding lid
        if coords[1] == y_sects - 1:
            f_cxy = sliding_top_wall(f_cxy=f_cxy, velocity=top_vel)
            # f_cxy = sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
        # if sect is at the bottom outer wall apply bounceback
        if coords[1] == 0:
            f_cxy = bottom_wall(f_cxy=f_cxy)
        # if sect is at the lef outer wall apply bounceback
        if coords[0] == 0:
            f_cxy = left_wall(f_cxy=f_cxy)
        # if sect is at the right outer wall apply bounceback
        if coords[0] == x_sects - 1:
            f_cxy = right_wall(f_cxy=f_cxy)
        f_cxy, u_axy = collision(f_cxy=f_cxy, omega=omega)

    # save final output.
    save_mpiio(cartcomm, "./ux_{}.npy".format(size), u_axy[0, :, :])
    save_mpiio(cartcomm, "./uy_{}.npy".format(size), u_axy[1, :, :])


if __name__ == "__main__":
    run_m7()
