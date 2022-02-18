# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================


from parameters import *
from variables import *
from fcn_economic import V_tail
import equations

import numpy as np
import cyipopt

# for automatic derivatives and just in time compilation
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev

# =======================================================================
#   Objective Function to start VFI (in our case, the value function)


def EV_F(X, kap):
    # extract tail kapital
    var_final = X[(Delta - 1) * n_pol : Delta * n_pol]
    kap_tail = var_final[I["knx"]]
    # extract utilities
    sum_utl = 0.0
    var = []
    for t in range(Delta):
        var.append(X[t * n_pol : (t + 1) * n_pol])
        sum_utl += (beta ** t * utility(var[t][I["con"]], var[t][I["lab"]]))
    #print("utility vector = ", X["utl"])
    #sum_utl = 
    return sum_utl + beta ** Delta * V_tail(kap_tail)


# =======================================================================

# =======================================================================
#   Computation of gradient (first order finite difference) of initial objective function


def EV_GRAD_F(X, kap):

    N = len(X)
    GRAD = np.zeros(N, float)  # Initial Gradient of Objective Function
    h = 1e-4

    for ixN in range(N):
        xAdj = np.copy(X)

        if xAdj[ixN] - h >= 0:
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F(xAdj, kap)

            xAdj[ixN] = X[ixN] - h
            fx1 = EV_F(xAdj, kap)

            GRAD[ixN] = (fx2 - fx1) / (2.0 * h)

        else:
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F(xAdj, kap)

            xAdj[ixN] = X[ixN]
            fx1 = EV_F(xAdj, kap)
            GRAD[ixN] = (fx2 - fx1) / h

    return GRAD


# =======================================================================

# ======================================================================
#   Equality constraints for the first time step of the model


def EV_G(X, kap):
    M = n_ctt_all
    G = np.empty(M, float)

    # I[iter] = slice(prv_ind, prv_ind + n_agt**d_pol[iter])
    # I_ctt[iter] = slice(prv_ind, prv_ind + n_agt**d_ctt[iter])
    ### put time back in
    var = []
    e_ctt = dict()
    e_ctt[0] = equations.f_ctt(X, kap, 0)
    # apply all constraints with this one loop
    for iter in ctt_key:
            G[I_ctt[iter]] = e_ctt[0][iter]
    """ for t in range(Delta):
        var[t] = X[t * n_pol : (t + 1) * n_pol]
        # pull in constraints
        e_ctt[t] = equations.f_ctt(var[t], kap, t)
        # apply all constraints with this one loop
        for iter in ctt_key:
            G[I_ctt[iter]] = e_ctt[t][iter] """

    return G


# ======================================================================

# ======================================================================
#   Computation (finite difference) of Jacobian of equality constraints
#   for first time step


def EV_JAC_G(X, kap, flag):
    N = n_pol_all
    M = n_ctt_all
    # print(N, "  ",M) #testing testing
    NZ = N * M
    A = np.empty(NZ, float)
    ACON = np.empty(
        NZ, int
    )  # its cause int is already a global variable cause i made it
    AVAR = np.empty(NZ, int)

    # Jacobian matrix structure

    if flag:
        for ixM in range(M):
            for ixN in range(N):
                ACON[ixN + (ixM) * N] = ixM
                AVAR[ixN + (ixM) * N] = ixN

        return (ACON, AVAR)

    else:
        # Finite Differences
        h = 1e-3
        gx1 = EV_G(X, kap)

        for ixM in range(M):
            for ixN in range(N):
                xAdj = np.copy(X)
                xAdj[ixN] = xAdj[ixN] + h
                gx2 = EV_G(xAdj, kap)
                A[ixN + ixM * N] = (gx2[ixM] - gx1[ixM]) / h
        return A


# ======================================================================

# ======================================================================


class cyipopt_class_inst:
    """
    Class for the optimization problem to be passed to cyipopt
    Further optimisations may be possible here by including a hessian
    """

    def __init__(self, k_init, verbose=True):
        self.k_init = k_init
        self.verbose = verbose

    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent
    def eval_f(self, x):
        return EV_F(x, self.k_init)

    def eval_grad_f(self, x):
        return EV_GRAD_F(x, self.k_init)

    def eval_g(self, x):
        return EV_G(x, self.k_init)

    def eval_jac_g(self, x, flag):
        return EV_JAC_G(x, self.k_init, flag)

    def objective(self, x):
        # Returns the scalar value of the objective given x.
        return self.eval_f(x)

    def gradient(self, x):
        # Returns the gradient fo the objective with respect to x."""
        return self.eval_grad_f(x)

    def constraints(self, x):
        # Returns the constraints
        return self.eval_g(x)

    def jacobian(self, x):
        # Returns the Jacobian of the constraints with respect to x.
        return self.eval_jac_g(x, False)

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """Prints information at every Ipopt i_pth."""

        if self.verbose:
            msg = "Objective value at step #{:d} of current optimization is - {:g}"
            print(msg.format(iter_count, obj_value))
