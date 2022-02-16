# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

from parameters import *
from variables import *
import equations_post
import ipopt_wrapping

import numpy as np
#import jax.numpy as jnp
#from jax import jit, grad, jacfwd
#from cyipopt import minimize_ipopt


#   Constraints
def EV_G_post(X, kap):
    M = n_ctt * Delta
    G = np.empty(M, float)
    var = []
    e_ctt = dict()
    for t in range(Delta):
        var[t] = X[t * n_pol: (t+1) * n_pol]
        # pull in constraints
        e_ctt[t] = equations_post.f_ctt(var[t], kap, t)
        # apply all constraints with this one loop
        for iter in ctt_key:
            G[I_ctt[iter]] = e_ctt[t][iter]
    return G
  
#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
def EV_JAC_G_post(X, flag, kap):
    N=len(X)
    M=n_ctt
    NZ=M*N
    A=np.empty(NZ, float)
    ACON=np.empty(NZ, int)
    AVAR=np.empty(NZ, int)

    # Jacobian matrix structure

    if (flag):
        for ixM in range(M):
            for ixN in range(N):
                ACON[ixN + (ixM)*N]=ixM
                AVAR[ixN + (ixM)*N]=ixN
        return (ACON, AVAR)

    else:
        # Finite Differences
        h=1e-4
        gx1=EV_G_post(X, kap)

        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G_post(xAdj, kap)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A
    
#======================================================================

class cyipopt_class_inst_post(ipopt_wrapping.ipopt_class_inst): 
    """
    Derived class for the optimization problem to be passed to cyipopt 
    Further optimisations may be possible here by including a hessian (optional param) 
    """

    def eval_g(self, x): 
        return EV_G_post(x, self.k_init)

    def eval_jac_g(self, x, flag):
        return EV_JAC_G_post(x, self.k_init, flag) 

