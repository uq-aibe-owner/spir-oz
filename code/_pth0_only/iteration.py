# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from variables import *
from equations import *
import solver

# import cPickle as pickle
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

# from numpy.random import PCG64
from datetime import datetime
import time

# ======================================================================


# def path_gen(i_pth, rng, save_data=True):
def path_gen(i_pth, save_data=True):
    # for i_pth=0,
    # iterate over periods of interest (the last extra period is for error checking only)
    # loop(tt$(ord(tt)<=Tstar+1),
    def call_to_solver(kap, final):
        return solver.ipopt_interface(
            kap, n_polAll, n_cttAll, final=False, verbose=False
        )

    res = dict()
    # solve for s == 0
    res[0] = call_to_solver(k_init, False)
    for s in range(1, Tstar + 1):
        # now solve for s > 0
        if s < Tstar + 1:
            res[s] = call_to_solver(res[s - 1]["knx"], False)
        else:
            res[s] = call_to_solver(res[s - 1]["knx"], True)

    output_file = filename + str(i_pth) + ".pcl"
    print(output_file)
    with open(output_file, "wb") as fd:
        pickle.dump(res, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print("data of path ", i_pth, " written to disk")
    print(" -------------------------------------------")
    fd.close()

    return


# ======================================================================
