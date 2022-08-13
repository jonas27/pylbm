import os

import matplotlib.pyplot as plt
import numpy as np
from pylbm import lbm

import m5_helpers

x_dim = 100
y_dim = 80
epochs = 10000
omega = 0.5
rho_in = 1.01
rho_out = 0.99
path = "./velocities.npy"

path = "./vel_t_100000.npy"
if not os.path.isfile(path):
    v_final = m5_helpers.m5_1_only_final(x_dim=x_dim, y_dim=y_dim, epochs=100000, omega=omega, rho_in=rho_in, rho_out=rho_out)
    np.save(path, v_final)
