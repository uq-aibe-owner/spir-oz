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
    var_L = np.ones(n_pol)
    var_U = np.ones(n_pol)
    var_start = np.ones(n_pol)
    short_ctt_L = np.ones(n_ctt)
    short_ctt_U = np.ones(n_ctt)
    for t in range(Delta):
        for iter in pol_key:
            var_L[I[iter]] = pol_L[iter]
            X_L[t * n_pol: (t+1) * n_pol] = var_L
            var_U[I[iter]] = pol_U[iter]
            X_U[t * n_pol: (t+1) * n_pol] = var_U
            # initial guesses for first i_pth (aka a warm start)
            #if iter != "sav" and iter != "con":  ### why no warm start?
            var_start[I[iter]] = pol_S[iter]
            X_start[t * n_pol: (t+1) * n_pol] = var_start
            # else:
            #    X[I[iter]] = pol_L[iter]
    # Set bounds for the constraints
        
        G_L[t * n_ctt: (t+1) * n_ctt] = short_ctt_L
        G_U[t * n_ctt: (t+1) * n_ctt] = short_ctt_U

    #print(X_L[I["con"]])
    #print(kap)
    SCEQ = cyipopt_class_inst(k_init=kap, verbose=vb)
    #print(SCEQ.n)
    #print(n_pol_all)
    #print(SCEQ.m)
    #print(n_ctt_all)
    nlp = cyipopt.Problem(
        n=N,
        m=M,
        problem_obj=SCEQ,
        lb=X_L,
        ub=X_U,
        cl=G_L,
        cu=G_U,
    )
    msg = "upper minus lower bounds on vars =\n"  
    msg += str(X_U - X_L)
    print(msg)
    nlp.add_option("obj_scaling_factor", -1.00)  # max function
    #nlp.add_option("mu_strategy", "adaptive")
    nlp.add_option("tol", 1e-1)
    nlp.add_option("print_level", 0)
    nlp.add_option("check_derivatives_for_naninf", "yes")
    #nlp.add_option("hessian_approximation", "limited-memory")
    #nlp.add_option("max_iter", 10)
    # solve the model and store related values in a dict called info
    x, res = nlp.solve(X_start)
    if final == True:
        nlp.close
    return res


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
