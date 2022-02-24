from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np


def objective(x):
    gamma = 0.25
    rho = -4
    rho_inv = 1/rho
    return np.power(np.sum(gamma * np.power(x[:4], rho)), rho_inv)

