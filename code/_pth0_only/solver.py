# ======================================================================
#
#     This routine interfaces with IPOPT
#
#     Based on Simon Scheidegger, 11/16 ; 07/17; 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021

#     Main difference is the shift from pyipopt to cyipopt
#     Involves a class to pass the optimisation problem to ipopt
# ======================================================================

from pickle import TRUE
from parameters import *
from parameters_compute import *
from variables import *
from equations import *
from ipopt_wrapping import cyipopt_class_inst

import numpy as np
import cyipopt


def ipopt_interface(kap, N, M, final=False, verbose=False):

    NELE_JAC = N * M
    NELE_HESS = (N + 1) * N / 2  # number of non-zero entries of Hess matrix

    # Vector for all policy variables
    X = np.empty(N)

    #    LAM = np.empty(M)  # multipliers
    #    G = np.empty(M)  # (in-)equality constraints
    # Vector of lower and upper bounds
    G_L = np.empty(M)
    G_U = np.empty(M)

    X_L = np.empty(N)
    X_U = np.empty(N)
    X_start = np.empty(N)
    # set bounds for policy variables
    for iter in pol_key:
        X_L[I[iter]] = pol_L[iter]
        X_U[I[iter]] = pol_U[iter]
        # initial guesses for first i_pth (aka a warm start)
        # if iter != "sav" and iter != "con":  ### why no warm start?
        #    X[I[iter]] = pol_S[iter]
        # else:
        #    X[I[iter]] = pol_L[iter]
    # Set bounds for the constraints
    for iter in ctt_key:
        G_L[I_ctt[iter]] = ctt_L[iter]
        G_U[I_ctt[iter]] = ctt_U[iter]

    print("hi from before ?")
    SCEQ = cyipopt_class_inst(
        X,
        N,
        M,
        k_init=kap,
        NELE_JAC=NELE_JAC,
        NELE_HESS=NELE_HESS,
        verbose=verbose,
    )
    print(SCEQ.n)
    print(n_pol_all)
    print(SCEQ.m)
    print(n_ctt_all)
    nlp = cyipopt.Problem(
        n=N,
        m=M,
        problem_obj=SCEQ,
        lb=X_L,
        ub=X_U,
        cl=G_L,
        cu=G_U,
    )
    print(X_U - X_L)
    nlp.add_option("obj_scaling_factor", -1.00)  # max function
    # nlp.add_option("mu_strategy", "adaptive")
    nlp.add_option("tol", ipopt_tol)
    nlp.add_option("print_level", 0)
    # nlp.add_option("hessian_approximation", "limited-memory")
    # nlp.add_option("max_iter", 10)
    # solve the model and store related values in a dict called info
    info = nlp.solve(X)[1]
    if final == True:
        nlp.close
    return info


"""
The above dict, info, has the following keys which may be called as info["key"] e.g. info["x"] will return the solution for the primal variables
x: ndarray, shape(n, ) # optimal solution
g: ndarray, shape(m, ) #constraints at the optimal solution
obj_val: float # objective value at optimal solution
mult_g: ndarray, shape(m, ) # final values of the constraint multipliers
mult_x_L: ndarray, shape(n, ) # bound multipliers at the solution
mult_x_U: ndarray, shape(n, ) # bound multipliers at the solution
status: integer # gives the status of the algorithm
status_msg: string # gives the status of the algorithm as a message 
"""
