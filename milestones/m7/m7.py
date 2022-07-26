"""
Notations:
f_ijc: is the probability density function with space dim i and j and velocity dim c

TODO: Moving from global lists to tuples could be faster.

"""

import numpy as np
from mpi4py import MPI

C_CA = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

W_C = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
# W_C: Tuple = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)

# The bounce back direction
C_REVERSED = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
# C_REVERSED: Tuple = (0, 3, 4, 1, 2, 7, 8, 5, 6)


def density_init(x_dim, y_dim, r_mean, eps) -> np.array:
    r_ij = eps * np.random.randn(x_dim, y_dim)
    r_ij[:, :] += r_mean
    return r_ij


def density(f_cxy) -> np.array:
    r_xy = np.einsum("cxy -> xy", f_cxy)
    return r_xy


def local_avg_velocity_init(x_dim, y_dim, u_mean, eps):
    u_aij = eps * np.random.randn(2, x_dim, y_dim)
    u_aij[:, :] += u_mean
    return u_aij


def local_avg_velocity(f_cxy: np.array, r_xy: np.array):
    """local_avg_velocity calculation based on f and rho."""
    u_aij = np.einsum("ac, cxy->axy", C_CA.T, f_cxy) / r_xy
    return u_aij


def f_eq(u_axy, r_xy):

    cu_cxy_3 = 3 * np.einsum("ac,axy->cxy", C_CA.T, u_axy)
    u_xy_2 = np.einsum("axy->xy", u_axy * u_axy)
    r_cxy_w = np.einsum("c,xy->cxy", W_C, r_xy)
    f_eq_cxy = r_cxy_w * (1 + cu_cxy_3 * (1 + 0.5 * cu_cxy_3) - 1.5 * u_xy_2[np.newaxis, :, :])
    return f_eq_cxy


def stream(f_cxy):
    """stream shifts all values according to the velocities.

    Implementation:
        Ranges over all velocities and rolls the values.
        Starts at index 1 as it doesn't change in 0 channel.

    Rational:
        Rolling over after array ends is ok, as we simulate a large plane which in its limit is transformed into a cylinder.
    """
    for i in range(1, 9):
        f_cxy[i, :, :] = np.roll(f_cxy[i, :, :], shift=C_CA[i], axis=(0, 1))
    return f_cxy


def bottom_wall(f_cxy):
    """m4: in the couette flow we have no side walls and thus need to copy all values up."""
    # should be np roll
    # argument ohne np roll: im steady state ist es egal
    f_cxy[2, :, 1] = f_cxy[4, :, 0]
    f_cxy[5, :, 1] = f_cxy[7, :, 0]
    f_cxy[6, :, 1] = f_cxy[8, :, 0]
    return f_cxy


def left_wall(f_cxy):
    f_cxy[1, 1, :] = f_cxy[3, 0, :]
    f_cxy[5, 1, :] = f_cxy[7, 0, :]
    f_cxy[8, 1, :] = f_cxy[6, 0, :]
    return f_cxy


def right_wall(f_cxy):
    f_cxy[3, -2, :] = f_cxy[1, -1, :]
    f_cxy[7, -2, :] = f_cxy[5, -1, :]
    f_cxy[6, -2, :] = f_cxy[8, -1, :]
    return f_cxy


def apply_sliding_top_wall_simple(f_cxy, velocity=None):
    """for incompressible fluids with  we can say that at the wall we have a density of -1/6, 0 and 1/6"""
    f_cxy[4, :, -2] = f_cxy[2, :, -1]
    f_cxy[7, :, -2] = f_cxy[5, :, -1] - 1 / 6.0 * velocity
    f_cxy[8, :, -2] = f_cxy[6, :, -1] + 1 / 6.0 * velocity
    return f_cxy


def in_out_pressure(f_cxy, rho_in, rho_out):
    r_xy = density(f_cxy)

    r_xy_in = np.full((1, r_xy.shape[1] - 2), rho_in)
    r_xy_out = np.full((1, r_xy.shape[1] - 2), rho_out)

    u_axy = local_avg_velocity(f_cxy=f_cxy, r_xy=r_xy)
    f_eq_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)

    f_eq_in = f_eq(u_axy=u_axy[:, -2:-1, 1:-1], r_xy=r_xy_in)
    f_cxy[:, :1, 1:-1] = f_eq_in + (f_cxy[:, -2:-1, 1:-1] - f_eq_cxy[:, -2:-1, 1:-1])

    f_eq_out = f_eq(u_axy=u_axy[:, 1:2, 1:-1], r_xy=r_xy_out)
    f_cxy[:, -1:, 1:-1] = f_eq_out + (f_cxy[:, 1:2, 1:-1] - f_eq_cxy[:, 1:2, 1:-1])
    return f_cxy


def collision(f_cxy, omega):
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


def reynolds(y_dim, omega, top_vel):
    """y_dim should be largest physical dimension."""
    nu = 1 / 3 * (1 / omega - 1 / 2)
    return (top_vel * y_dim) / (nu)


def parallelize(x_dim, y_dim, x_grids, y_grids):
    comm = MPI.COMM_WORLD
    cartcomm = comm.Create_cart((x_grids, y_grids), periods=(False, False))
    rows = x_dim // x_grids
    columns = y_dim // y_grids


def sync_shifts(cartcomm):
    sL, dL = cartcomm.Shift(0, -1)
    sR, dR = cartcomm.Shift(0, 1)
    sD, dD = cartcomm.Shift(1, 1)
    sU, dU = cartcomm.Shift(1, -1)
    shifts = (sL, dL, sR, dR, sU, dU, sD, dD)
    return shifts


def sync_f(cartcomm, shifts, f_cxy):
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
    from numpy.lib.format import dtype_to_descr, magic

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


def run():
    org_x_dim = 300
    org_y_dim = 300
    size = org_y_dim * org_x_dim
    epochs = 100
    omega = 1.7
    top_vel = 0.1

    comm = MPI.COMM_WORLD
    cartcomm = comm.Create_cart(dims=(150, 150), periods=(False, False), reorder=False)

    size = comm.Get_size()
    x_sects = int(np.floor(np.sqrt(size)))
    y_sects = int(size / x_sects)
    x_dim = org_x_dim // x_sects
    y_dim = org_y_dim // y_sects

    r_xy = density_init(x_dim=x_dim, y_dim=y_dim, r_mean=1.0, eps=0.0)
    u_axy = local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.0, eps=0.0)
    f_cxy = f_eq(u_axy=u_axy, r_xy=r_xy)
    shifts = sync_shifts(cartcomm)

    for _ in range(epochs):
        f_cxy = sync_f(cartcomm, shifts, f_cxy)
        f_cxy = stream(f_cxy=f_cxy)
        rank = cartcomm.Get_rank()
        coords = cartcomm.Get_coords(rank)
        if coords == [1, 1] or coords == [0, 1]:
            f_cxy = apply_sliding_top_wall_simple(f_cxy=f_cxy, velocity=top_vel)
        if coords == [0, 0] or coords == [1, 0]:
            f_cxy = bottom_wall(f_cxy=f_cxy)
        if coords == [0, 0] or coords == [0, 1]:
            f_cxy = left_wall(f_cxy=f_cxy)
        if coords == [1, 0] or coords == [1, 1]:
            f_cxy = right_wall(f_cxy=f_cxy)
        f_cxy, u_axy = collision(f_cxy=f_cxy, omega=omega)

    save_mpiio(cartcomm, "./ux.npy", u_axy[0, :, :])
    save_mpiio(cartcomm, "./uy.npy", u_axy[1, :, :])


run()
