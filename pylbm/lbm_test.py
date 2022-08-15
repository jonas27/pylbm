import numpy as np

from pylbm import lbm


def test_density_init():
    x_dim, y_dim = 3, 5
    r_value = 0.5
    eps = 0.01
    r_ij = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_value, eps=eps)
    assert r_ij.shape == (x_dim, y_dim)
    assert np.isclose(r_ij.mean(), r_value, rtol=eps)
    assert np.isclose(r_ij.sum(), x_dim * y_dim * r_value, rtol=eps)


def test_density():
    eps = 0.01
    r_mean = 0.5
    x_dim, y_dim = 5, 10
    r_ij = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_mean, eps=eps)
    assert r_ij.shape == (x_dim, y_dim)


def test_local_average_velocity():
    eps = 0.01
    r_mean = 0.5
    u_mean = 0.5
    x_dim, y_dim = 5, 10
    r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=u_mean, eps=eps)
    f_cij = lbm.f_eq(u_axy=u_aij, r_xy=r_xy)
    u_aij = lbm.local_avg_velocity(f_cxy=f_cij, r_xy=r_xy)
    assert u_aij.shape == (2, x_dim, y_dim)


def test_f_eq():
    eps = 0.01
    r_mean = 0.5
    x_dim, y_dim, v_dim = 5, 10, 9
    r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_mean, eps=eps)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=0.5, eps=0.01)
    f_eq_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)
    assert f_eq_cxy.shape == (v_dim, x_dim, y_dim)


def test_stream():
    eps = 0.0001
    r_mean = 0.5
    u_mean = 0.5
    x_dim, y_dim = 5, 10
    r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=u_mean, eps=eps)
    f_cxy_init = lbm.f_eq(u_axy=u_aij, r_xy=r_xy)
    f_cxy = lbm.stream(f_cxy_init.copy())
    # assert correct shape
    assert f_cxy.shape == (9, x_dim, y_dim)
    # assert correct streaming
    for i in range(9):
        i_start, j_start = 1, 1
        assert f_cxy_init[i, i_start, j_start] == f_cxy[i, i_start + lbm.C_CA[i, 0], j_start + lbm.C_CA[i, 1]]


def test_bottom_wall():
    # TODO: better testing needed. Not sure if correct.
    eps = 0.0001
    r_mean = 0.5
    u_mean = 0.5
    xi_dim, y_dim = 12, 4

    r_xy = lbm.local_density_init(x_dim=xi_dim, y_dim=y_dim, r_init=r_mean, eps=eps)
    u_axy = lbm.local_avg_velocity_init(x_dim=xi_dim, y_dim=y_dim, u_mean=u_mean, eps=eps)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)
    f_cxy_old = f_cxy.copy()
    f_cxy = lbm.stream(f_cxy)
    f_cxy = lbm.bottom_wall(f_cxy=f_cxy, f_cxy_old=f_cxy_old)
    directions = [0, 1, 2, 3, 5, 6]
    np.testing.assert_almost_equal(f_cxy[directions, :, 0].sum(), f_cxy_old[lbm.C_REVERSED[directions], :, 0].sum())


def test_sliding_top_wall():
    """check how to test this."""
    eps = 0.0
    velocity = 0.1
    r_mean = 1
    u_mean = 0
    x_dim, y_dim = 5, 10

    r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_mean, eps=eps)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=u_mean, eps=eps)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)
    f_cxy_old = f_cxy.copy()
    f_cxy = lbm.stream(f_cxy)
    f_cxy = lbm.sliding_top_wall(f_cxy=f_cxy, f_cij_old=f_cxy_old, velocity=velocity)
    assert f_cxy.shape == (9, x_dim, y_dim)


def test_sliding_top_wall_simple():
    """check how to test this."""
    eps = 0.0
    velocity = 0.1
    r_mean = 1
    u_mean = 0
    x_dim, y_dim = 5, 10

    r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=y_dim, r_init=r_mean, eps=eps)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=y_dim, u_mean=u_mean, eps=eps)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)
    f_cxy_old = f_cxy.copy()
    f_cxy = lbm.stream(f_cxy)
    f_cxy = lbm.sliding_top_wall_simple(f_cxy=f_cxy, f_cij_old=f_cxy_old, velocity=velocity)
    assert f_cxy.shape == (9, x_dim, y_dim)


def test_collision():
    eps = 0.01
    r_value = 0.5
    x_dim, x_dim = 5, 10
    r_xy = lbm.local_density_init(x_dim=x_dim, y_dim=x_dim, r_init=r_value, eps=eps)
    u_axy = lbm.local_avg_velocity_init(x_dim=x_dim, y_dim=x_dim, u_mean=0.5, eps=0.01)
    f_cxy = lbm.f_eq(u_axy=u_axy, r_xy=r_xy)
    f_cxy, u_axy = lbm.collision(f_cxy=f_cxy, omega=0.5)
    assert f_cxy.shape == (9, x_dim, x_dim)
