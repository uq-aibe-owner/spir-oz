#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

# import solver as solver
from parameters import *  # parameters of model
from parameters_compute import *
from variables import *
from equations import *
import postprocessing as post  # computes the L2 and Linfinity error of the model
import iteration

# import iteration_post
# import numpy as np
# import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib

# from mpl_toolkits.mplot3d import Axes3D
import pickle
import time

# from numpy.random import PCG64
# from datetime import datetime
# ======================================================================
# Start with Value Function i_pth


# Set up a container to save the k/obj pairs as they are sampled & optimised

start = time.time()


for i_pth in range(1):  # range(Tstar):
    # terminal value function
    if i_pth == 0:
        print("start with untipped path")
        iteration.path_gen(i_pth)

    else:
        print("Now we are running the tipped path", i_pth)
    #    iteration_post.path_gen_post(i_pth)

# now = datetime.now()
end = time.time()
minutes_taken = (end - start) / 60
# ======================================================================
print("===============================================================")
print(" ")
print(
    " Computation of this multi-regional model",
    " generated ",
    n_pth,
    " paths, finishing in ",
    minutes_taken,
    " minutes",
    " ",
)
print(" ")
print("===============================================================")
# ======================================================================

# ======================================================================
