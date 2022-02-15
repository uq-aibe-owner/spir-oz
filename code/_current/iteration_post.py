# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from variables import *
from equations import *
import solver_post
# import cPickle as pickle
import pickle
import os

#from numpy.random import PCG64
#from datetime import datetime
import time 
# ======================================================================


#def sceq(i_pth, rng, save_data=True):
def sceq(i_pth, save_data=True):
    # i_pth is greater than zero
    s = i_pth - 1
    # unpickle
    infile = open(filename + "0.pcl",'rb')
    res = pickle.load(infile)
    kap = res[s][I["knx"]] 
    infile.close()
    #solve busc_tipped using nlp maximizing obj;
    solver_post.ipopt_interface(kap)
    ###
    output_file = filename + str(i_pth) + ".pcl"
    print(output_file)
    with open(output_file, "wb") as fd:
        pickle.dump(res, fd, protocol=pickle.HIGHEST_PROTOCOL)
        print("data of step ", i_pth, "  written to disk")
        print(" -------------------------------------------")
    fd.close()

    #if i_pth == numits - 1:
    return


# ======================================================================
