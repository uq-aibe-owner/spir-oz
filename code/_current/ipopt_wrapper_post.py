#=======================================================================
#
#     ipopt_wrapper.py : an interface to IPOPT and PYIPOPT
#
#     Simon Scheidegger, 06/17
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
#
#=======================================================================

import numpy as np
from parameters import *
from variables import *
from equations import *

#=======================================================================

#=======================================================================
#   Objective Function during VFI (note - we need to interpolate on an "old" GPR)
    
def EV_F_ITER(X, kap, n_agt, gp_old):
    
    """ # initialize correct data format for training point
    s = (1,n_agt)
    kap2 = np.zeros(s)
    kap2[0,:] = X[I["knx"]]
    
    # interpolate the function, and get the point-wise std.
    val = X[I["val"]]
    
    VT_sum = X[I["utl"]] + beta*val """
    
    return X[I["utl"]] + beta*X[I["val"]]
    
#=======================================================================

    
#=======================================================================
#   Computation of gradient (first order finite difference) of the objective function 
    
def EV_GRAD_F_ITER(X, kap, n_agt, gp_old):
    
    N=len(X)
    GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
    h=1e-4
    
    for ixN in range(N):
        xAdj=np.copy(X)
        
        if (xAdj[ixN] - h >= 0):
            xAdj[ixN]=X[ixN] + h            
            fx2=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            
            xAdj[ixN]=X[ixN] - h
            fx1=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            
            GRAD[ixN]=(fx2-fx1)/(2.0*h)
            
        else:
            xAdj[ixN]=X[ixN] + h
            fx2=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            
            xAdj[ixN]=X[ixN]
            fx1=EV_F_ITER(xAdj, kap, n_agt, gp_old)
            GRAD[ixN]=(fx2-fx1)/h
            
    return GRAD
       
#======================================================================

#======================================================================
#   Equality constraints during the VFI of the model
def EV_G_ITER(X, kap, n_agt, gp_old):
    
    M=n_ctt
    G=np.empty(M, float)

    s = (1,n_agt)
    kap2 = np.zeros(s)
    kap2[0,:] = X[I["knx"]]

    """ print("should be the same")
    print(type(X[I["knx"]]))
    print(np.shape(X[I["knx"]]))
    print(type(kap2))
    print(np.shape(kap2)) """

    # pull in constraints
    e_ctt = f_ctt(X, gp_old, kap2, 0, kap)
    # apply all constraints with this one loop
    for iter in ctt_key:
        G[I_ctt[iter]] = e_ctt[iter]

    return G
  
#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
#   during iteration  
  
def EV_JAC_G_ITER(X, flag, kap, n_agt, gp_old):
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
        gx1=EV_G_ITER(X, kap, n_agt, gp_old)

        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G_ITER(xAdj, kap, n_agt, gp_old)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A
    
#======================================================================

class ipopt_obj(): 
    """
    Class for the optimization problem to be passed to cyipopt 
    Further optimisations may be possible here by including a hessian (optional param) 
    Uses the existing instance of the Gaussian Process (GP OLD) 
    """

    def __init__(self, X, n_agents, k_init, NELE_JAC, NELE_HESS, gp_old=None,initial=False, verbose=False): 
        self.x = X 
        self.n_agents = n_agents 
        self.k_init = k_init 
        self.NELE_JAC = NELE_JAC 
        self.NELE_HESS = NELE_HESS 
        self.gp_old = gp_old 
        self.initial = initial 
        self.verbose = verbose 

    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent 
    def eval_f(self, x): 
        return EV_F_ITER(x, self.k_init, self.n_agents, self.gp_old) 

    def eval_grad_f(self, x): 
        return EV_GRAD_F_ITER(x, self.k_init, self.n_agents, self.gp_old) 

    def eval_g(self, x): 
        return EV_G_ITER(x, self.k_init, self.n_agents, self.gp_old) 

    def eval_jac_g(self, x, flag):
        return EV_JAC_G_ITER(x, flag, self.k_init, self.n_agents, self.gp_old) 

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

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        if self.verbose: 
            msg = "Objective value at iteration #{:d} is - {:g}"
            print(msg.format(iter_count, obj_value))
