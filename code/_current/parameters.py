# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np

# import jax.numpy as jnp

# ======================================================================
## Verbosity of print output
vb = False
economic_verbose = True
# ======================================================================
# number of goods of the model
# Number of paths
#No_samples = 10 * n_agt
# ======================================================================
## Control of paths
# To start from scratch, set numstart = 0.
# Otherwise set numstart equal to previous numits. (Equivalently,
# set numstart equal to the last path.)
numstart = 0
# how many i_pths
fthrits = 21
numits = numstart + fthrits

# ======================================================================
# Number of test points to compute the error in postprocessing
# No_samples_postprocess = 20 ##not needed as we use all paths generated now

# ======================================================================
# length_scale_bounds=(10e-9,10e10) ##not needed: for GP

ipopt_tol = 1e-3  ##not needed: for GP
# n_restarts_optimizer=10 ##not needed: for GP
filename = "paths/path_number_"
# folder with the restart/result files

# ================================================================
# Economic Parameters

n_agt = 4  ## for Cai-Judd this is number of regions
Delta = 1
Tstar = 1
n_pth = 1  # Tstar + 1
beta = 0.99
# rho = 0.95
zeta1 = 1
zeta2 = 0.95
phi = 0.5  # adjustment cost multiplier
phik = 0.5  # weight of capital in production # alpha in CJ
phim = 0.5  # weight of intermediate inputs in production
phil = 1 - phik # labour's importancd in production
p_01 = 0.01  # transition probability from state 0 to 1

gamma = 0.5  # power utility exponent
gammahat = 1 - 1 / gamma
tau = np.ones(n_agt)  # regional weights ### change to rho
delta = 0.025  # discount factor
eta = 0.5  # Frisch elasticity of labour supply
etahat = 1 + 1 / eta
A = (1 - (1 - delta) * beta) / (phik * beta)
B = (1 - phik) * A * (A - delta) ** (-1 / gamma)
xi = np.ones(n_agt) * 1 / n_agt
mu = np.ones(n_agt) * 1 / n_agt
