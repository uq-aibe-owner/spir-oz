# =======================================================================
#
#     ipopt_wrapper.py : an interface to IPOPT and PYIPOPT
#
#     Simon Scheidegger, 06/17
#
#     Cameron Gordon, 11/21 - adj file name to avoid issues with cyipopt
# =======================================================================

from parameters import *
from econ import *
import numpy as np

# =======================================================================
#   Objective Function to start VFI (in our case, the value function)


def EV_F(X, k_init, n_agents):

    # Extract Variables
    cons = X[0:n_agents]
    lab = X[n_agents : 2 * n_agents]
    inv = X[2 * n_agents : 3 * n_agents]

    knext = (1 - delta) * k_init + inv

    # Compute Value Function
    VT_sum = utility(cons, lab) + beta * V_INFINITY(knext)

    return VT_sum


# V infinity
def V_INFINITY(k=[]):
    e = np.ones(len(k))
    c = output_f(k, e)
    v_infinity = utility(c, e) / (1 - beta)
    return v_infinity


# =======================================================================


# =======================================================================
#   Computation of gradient (first order finite difference) of initial objective function


def EV_GRAD_F(X, k_init, n_agents):

    N = len(X)
    GRAD = np.zeros(N, float)  # Initial Gradient of Objective Function
    h = 1e-4

    for ixN in range(N):
        xAdj = np.copy(X)

        if xAdj[ixN] - h >= 0:
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F(xAdj, k_init, n_agents)

            xAdj[ixN] = X[ixN] - h
            fx1 = EV_F(xAdj, k_init, n_agents)

            GRAD[ixN] = (fx2 - fx1) / (2.0 * h)

        else:
            xAdj[ixN] = X[ixN] + h
            fx2 = EV_F(xAdj, k_init, n_agents)

            xAdj[ixN] = X[ixN]
            fx1 = EV_F(xAdj, k_init, n_agents)
            GRAD[ixN] = (fx2 - fx1) / h

    return GRAD


# =======================================================================
#   Computation of gradient (first order finite difference) of the objective function

# ======================================================================
#   Equality constraints for the first time step of the model


def EV_G(X, k_init, n_agents):
    N = len(X)
    M = 3 * n_agents + 1  # number of constraints
    G = np.empty(M, float)

    # Extract Variables
    cons = X[:n_agents]
    lab = X[n_agents : 2 * n_agents]
    inv = X[2 * n_agents : 3 * n_agents]

    # first n_agents equality constraints
    for i in range(n_agents):
        G[i] = cons[i]
        G[i + n_agents] = lab[i]
        G[i + 2 * n_agents] = inv[i]

    Gam_ad = Gamma_adjust(k_init, inv)
    f_prod = output_f(k_init, lab)
    sectors_sum = cons + inv - delta * k_init - (f_prod - Gam_ad)
    G[3 * n_agents] = np.sum(sectors_sum)  # CJ eq.15

    return G


# ======================================================================
#   Computation (finite difference) of Jacobian of equality constraints
#   for first time step


def EV_JAC_G(X, flag, k_init, n_agents):
    N = len(X)
    M = 3 * n_agents + 1
    NZ = M * N
    A = np.empty(NZ, float)
    ACON = np.empty(NZ, int)
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
        h = 1e-4
        gx1 = EV_G(X, k_init, n_agents)

        for ixM in range(M):
            for ixN in range(N):
                xAdj = np.copy(X)
                xAdj[ixN] = xAdj[ixN] + h
                gx2 = EV_G(xAdj, k_init, n_agents)
                A[ixN + ixM * N] = (gx2[ixM] - gx1[ixM]) / h
        return A


# ======================================================================
# ======================================================================
