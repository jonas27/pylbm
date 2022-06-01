import numpy as np

from pylbm import lbm


def test_roh_init():
    i_dim, j_dim = 3, 5
    r_value = 0.5
    eps = 0.01
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_value, eps=eps)
    assert r_ij.shape == (i_dim, j_dim)
    assert np.isclose(r_ij.mean(), r_value, rtol=eps)
    assert np.isclose(r_ij.sum(), i_dim * j_dim * r_value, rtol=eps)


def test_roh():
    eps = 0.01
    r_mean = 0.5
    i_dim, j_dim = 5, 10
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    assert r_ij.shape == (i_dim, j_dim)


def test_local_average_velocity():
    eps = 0.01
    r_mean = 0.5
    u_mean = 0.5
    i_dim, j_dim = 5, 10
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(i_dim=i_dim, j_dim=j_dim, u_mean=u_mean, eps=eps)
    f_cij = lbm.f_eq(u_aij=u_aij, r_ij=r_ij)
    u_aij = lbm.local_avg_velocity(f_cij=f_cij, r_ij=r_ij)
    assert u_aij.shape == (2, i_dim, j_dim)


def test_f_eq():
    eps = 0.01
    r_mean = 0.5
    i_dim, j_dim, v_dim = 5, 10, 9
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(i_dim=i_dim, j_dim=j_dim, u_mean=0.5, eps=0.01)
    f_eq_cij = lbm.f_eq(u_aij=u_aij, r_ij=r_ij)
    assert f_eq_cij.shape == (v_dim, i_dim, j_dim)


def test_stream():
    eps = 0.0001
    r_mean = 0.5
    u_mean = 0.5
    i_dim, j_dim, v_dim = 5, 10, 9
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(i_dim=i_dim, j_dim=j_dim, u_mean=u_mean, eps=eps)
    f_cij_init = lbm.f_eq(u_aij=u_aij, r_ij=r_ij)
    f_cij = lbm.stream(f_cij_init.copy())
    # assert correct shape
    assert f_cij.shape == (v_dim, i_dim, j_dim)
    # assert correct streaming
    for i in range(9):
        i_start, j_start = 1, 1
        assert f_cij_init[i, i_start, j_start] == f_cij[i, i_start + lbm.C_CA[i, 0], j_start + lbm.C_CA[i, 1]]


def test_boundries_bounce():
    eps = 0.0001
    r_mean = 0.5
    u_mean = 0.5
    i_dim, j_dim = 5, 10
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(i_dim=i_dim, j_dim=j_dim, u_mean=u_mean, eps=eps)
    f_cij = lbm.f_eq(u_aij=u_aij, r_ij=r_ij)
    f_cij_old = f_cij.copy()
    f_cij = lbm.stream(f_cij)
    boundries = lbm.make_boundries(i_dim=i_dim, j_dim=j_dim, north=True, east=True, south=True, west=True)
    f_cij = lbm.apply_boundries(f_cij=f_cij, f_cij_old=f_cij_old, boundries=boundries)
    np.testing.assert_array_equal(f_cij[:, boundries], f_cij_old[:, boundries])


def test_sliding_top_boundry():
    eps = 0.0
    velocity = 0.1
    r_mean = 0.5
    u_mean = 0.5
    i_dim, j_dim = 5, 10
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_mean, eps=eps)
    u_aij = lbm.local_avg_velocity_init(i_dim=i_dim, j_dim=j_dim, u_mean=u_mean, eps=eps)
    f_cij = lbm.f_eq(u_aij=u_aij, r_ij=r_ij)
    f_cij_old = f_cij.copy()
    f_cij = lbm.stream(f_cij)
    boundries = lbm.make_boundries(i_dim=i_dim, j_dim=j_dim, north=True, east=True, south=True, west=True)
    f_cij = lbm.apply_boundries(f_cij=f_cij, f_cij_old=f_cij_old, boundries=boundries)
    f_cij = lbm.apply_sliding_top_boundry(f_cij=f_cij, f_cij_old=f_cij_old, velocity=velocity)
    assert f_cij.shape == (9, i_dim, j_dim)


def test_collision():
    eps = 0.01
    r_value = 0.5
    i_dim, j_dim = 5, 10
    r_ij = lbm.rho_init(i_dim=i_dim, j_dim=j_dim, r_mean=r_value, eps=eps)
    u_aij = lbm.local_avg_velocity_init(i_dim=i_dim, j_dim=j_dim, u_mean=0.5, eps=0.01)
    f_cij = lbm.f_eq(u_aij=u_aij, r_ij=r_ij)
    f_cij, u_aij = lbm.collision(f_cij=f_cij, omega=0.5)
    assert f_cij.shape == (9, i_dim, j_dim)
