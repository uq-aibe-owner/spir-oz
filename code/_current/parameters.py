# ======================================================================
#
#     sets the parameters and economic functions for the model
#     "Growth Model"
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

import numpy as np

# ======================================================================
## Verbosity of print output
verbose = True
economic_verbose = True
# ======================================================================
# number of goods of the model
n_agt = 2  ## for Cai-Judd this is number of regions
# Number of paths
No_samples = 10 * n_agt
# ======================================================================
## Control of paths
# To start from scratch, set numstart = 0.
# Otherwise set numstart equal to previous numits. (Equivalently,
# set numstart equal to the last path.)
numstart = 0
# how many iterations
fthrits = 4
numits = numstart + fthrits
# ======================================================================
# Number of test points to compute the error in postprocessing
# No_samples_postprocess = 20 ##not needed as we use all paths generated now

# ======================================================================
# length_scale_bounds=(10e-9,10e10) ##not needed: for GP

ipopt_tol = 1e-1 ##not needed: for GP
# n_restarts_optimizer=10 ##not needed: for GP
# filename = "restart/restart_file_step_" ##not needed: for GP
# folder with the restart/result files

# ======================================================================
# Model Parameters

beta = 0.99
# rho = 0.95
# zeta = 0.0
""" phi = {
    "itm": 0.5,
    "kap": 0.5
} """

phik = 0.5
phim = 0.5

gamma = 2.0
delta = 0.1
eta = 1
big_A = 1 / (phim ** phim * phik ** phik)  # * (1-phik-phim)**(1-phik-phim))
xi = np.ones(n_agt) * 1 / n_agt
mu = np.ones(n_agt) * 1 / n_agt

# Ranges For States
kap_L = 2
kap_U = 5
range_cube = kap_U - kap_L  # range of [0..1]^d in 1D

# ======================================================================
