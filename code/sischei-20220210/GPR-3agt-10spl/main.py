#======================================================================
#
#     This routine solves an infinite horizon growth model 
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - scikit-learn GPR (https://scikit-learn.org)
#
#     Simon Scheidegger, 01/19 
#
#     Cameron Gordon, 11/21 - conversion to Python3    
#======================================================================

import nonlinear_solver_pre as solver     #solves opt. problems for terminal VF
import nonlinear_solver_post as solvpost   #solves opt. problems during VFI
from parameters import *                      #parameters of model
import iteration_pre as iter_pre              #interface to sparse grid library/terminal VF
import iteration_post as iter_post    #interface to sparse grid library/iteration
import postprocessing as post                 #computes the L2 and Linfinity error of the model
import numpy as np


#======================================================================
# Start with Value Function Iteration


for i in range(numstart, numits):
# terminal value function
    if (i==1):
        print("start with Value Function Iteration")
        iter_pre.GPR_init(i)
    
    else:     
        print("Now, we are in Value Function Iteration step", i)
        iter_post.GPR_iter(i)
    
    
#======================================================================
print("===============================================================")
print(" ")
print(" Computation of a growth model of dimension ", n_agents ," finished after ", numits, " steps")
print(" ")
print("===============================================================")
#======================================================================

# compute errors   
avg_err=post.ls_error(n_agents, numstart, numits, No_samples_postprocess)

#======================================================================
print("===============================================================")
print(" ")
#print " Errors are computed -- see error.txt"
print(" ")
print("===============================================================")
#======================================================================
