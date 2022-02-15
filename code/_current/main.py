
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

import solver as solver
from parameters import *  # parameters of model
from variables import *
from equations import *
import postprocessing as post  # computes the L2 and Linfinity error of the model
import iteration
import iteration_post
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
from numpy.random import PCG64
from datetime import datetime
# ======================================================================
# Start with Value Function i_pth


# Set up a container to save the k/obj pairs as they are sampled & optimised

start = time.time()


for i_pth in range(0, Tstar):
# terminal value function
    if (i_pth==0):
        print("start with untipped path")
        iteration.sceq(i_pth)
    
    else:     
        print("Now we are running the tipped path", i_pth)
        iteration_post.sceq_post(i_pth)






now = datetime.now()
#dt = int(now.strftime("%H%M%S%f"))
#rngm = np.random.default_rng(dt)  # fix seed #move to main so it doesnt re-initialise
#print(rng)

for iter in range(numstart, numits):
    # terminal value function
    # run the model:
    res_fin = i_pth.sceq(iter)

# ======================================================================
print("===============================================================")
print(" ")
print(
    " Computation of a growth model of dimension ",
    n_agt,
    " finished after ",
    numits,
    " steps",
)
print(" ")
print("===============================================================")
# ======================================================================

# ======================================================================
end = time.time()

