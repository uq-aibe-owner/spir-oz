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
from numpy.random import PCG64
from datetime import datetime
import time 
# ======================================================================


#def sceq(i_pth, rng, save_data=True):
def sceq(i_pth, save_data=True):
    # for i_pth=0,  
    # iterate over periods of interest (the last extra period is for error checking only)
    #loop(tt$(ord(tt)<=Tstar+1),

    res = dict()
    # solve for s == 0
    res[0] = solver.ipopt_interface(k_init, n_agt=None, final=False, verbose=False) ###           
    for s in range(1, Tstar+1):
        #now solve for s > 0
        if s < Tstar+1:
            res[s] = solver.ipopt_interface(res[s-1]["knx"], n_agt=None, final=False, verbose=False) ###
        else:
            res[s] = solver.ipopt_interface(res[s-1]["knx"], n_agt=None, final=True, verbose=False) ###

        output_file = filename + str(i_pth) + ".pcl"
        print(output_file)
        with open(output_file, "wb") as fd:
            pickle.dump(res, fd, protocol=pickle.HIGHEST_PROTOCOL)
            print("data of path ", i_pth, "  written to disk")
            print(" -------------------------------------------")
        fd.close()

    return


# ======================================================================
