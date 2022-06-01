import logging

import numpy as np

from pylbm import lbm


def run(epochs: int):

    pd_func = lbm.f_ijc_init()

    # create walls later

    for e in range(epochs):
        # Step 1: stream for one time period
        for i in range(v_dim):
            pd_func[:, :, i] = np.roll(pd_func[:, :, i], lbm.velocities()[i, 0], axis=0)
            pd_func[:, :, i] = np.roll(pd_func[:, :, i], lbm.velocities()[i, 1], axis=1)

        # Step 2: collision for one time period
        # calc density at time t over x
        rho = np.sum(pd_func, axis=2)
        # calculate momentum
        # print(momentum.shape)
        momentum = np.sum(np.dot(pd_func.T, lbm.velocities().T), 2) / rho
        print(momentum.shape)
        return


if __name__ == "__main__":
    logger = logging.getLogger("pylbm")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    run(epochs=1)
