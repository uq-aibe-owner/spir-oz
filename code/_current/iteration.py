# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from variables import *
from equations import *
from parameters_compute import k_init  # for k_init
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
            kap, n_pol_all, n_ctt_all, final=False, verbose=False
        )

    info = dict()
    # solve for s == 0
    info[0] = call_to_solver(kap=k_init, final=False)
    print("status_msg0", info[0]["status_msg"])
    print("knx at ", 0, "is ", info[0]["x"][I["knx"]])
    print("total con at ", 0, "is ", sum(info[0]["x"][I["con"]]))
    print("total sav at ", 0, "is ", sum(info[0]["x"][I["sav"]]))
    #print("total out at ", 0, "is ", sum(info[0]["x"][I["out"]]))
    print("lab at ", 0, "is ", info[0]["x"][I["lab"]])
    ### to fix
    for s in range(1, Tstar + 1):
        # now solve for s > 0
        var = info[s - 1]["x"][(s - 1) * n_pol: s * n_pol]
        kap = var[I["knx"]]
        print(kap)
        if s < Tstar:
            info[s] = call_to_solver(kap, final=False)
            print("status_msg1", info[s]["status_msg"])
        else:
            info[s] = call_to_solver(kap, final=True)
            print("status_msg2", info[s]["status_msg"])

    output_file = filename + str(i_pth) + ".pcl"
    print(output_file)
    with open(output_file, "wb") as fd:
        pickle.dump(info, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print("data of path ", i_pth, " written to disk")
    print(" -------------------------------------------")
    fd.close()

    return


# ======================================================================
