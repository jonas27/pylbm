import cProfile
import logging

from pylbm import lbm

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()


def test_bench_streaming():
    """use with python -m pytest milestones/m1/m1_test.py"""
    eps = 0.01
    r_value = 1.5
    i_dim, j_dim, v_dim = 500, 1000, 9
    r_ij = lbm.density_init(x_dim=i_dim, y_dim=j_dim, r_mean=r_value, eps=eps)
    f_ijc_init = lbm.f_ijc_init(i_dim=i_dim, j_dim=j_dim, v_dim=v_dim, r_ij=r_ij)
    f_ijc = f_ijc_init.copy()
    log.info(cProfile.runctx("for i in range(100): stream(f_ijc)", {"stream": lbm.stream, "f_ijc": f_ijc}, {}))


def test_streaming():
    eps = 0.01
    r_value = 1.5
    i_dim, j_dim, v_dim = 5, 10, 9
    r_ij = lbm.density_init(x_dim=i_dim, y_dim=j_dim, r_mean=r_value, eps=eps)
    f_ijc_init = lbm.f_ijc_init(i_dim=i_dim, j_dim=j_dim, v_dim=v_dim, r_ij=r_ij)
    f_ijc = lbm.stream(f_cxy=f_ijc_init.copy())

    i_start, j_start = 1, 1
    for i in range(9):
        assert f_ijc_init[i_start, j_start, i] == f_ijc[i_start + lbm.C_CA[i, 0], j_start + lbm.C_CA[i, 1], i]
