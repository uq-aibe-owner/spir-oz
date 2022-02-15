# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from variables import *
import equations_post

#=======================================================================

#=======================================================================
#   Objective Function during VFI (note - we need to interpolate on an "old" GPR)
    
def EV_F_post(X, kap, n_agt):
    
    """ # initialize correct data format for training point
    s = (1,n_agt)
    kap2 = np.zeros(s)
    kap2[0,:] = X[I["knx"]]
    
    # interpolate the function, and get the point-wise std.
    val = X[I["val"]]
    
    VT_sum = X[I["utl"]] + beta*val """
       ### value function needs to have the full sum over Dt=30 periods: for each s
    ### thus create a vector of utilities
    u = np.zeros(Delta)
    #u = utility(cons=[], lab=[])
    ### the objective fcn itself
    #VT_sum = sum(u) + beta**Delta * V_tail_post 

    return sum(X[I["utl"]]) + beta**Delta * V_tail
    
#=======================================================================

    
#=======================================================================
#   Computation of gradient (first order finite difference) of the objective function 
    
def EV_GRAD_F_post(X, kap, n_agt):
    
    N=len(X)
    GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
    h=1e-4
    
    for ixN in range(N):
        xAdj=np.copy(X)
        
        if (xAdj[ixN] - h >= 0):
            xAdj[ixN]=X[ixN] + h            
            fx2=EV_F_post(xAdj, kap, n_agt)
            
            xAdj[ixN]=X[ixN] - h
            fx1=EV_F_post(xAdj, kap, n_agt)
            
            GRAD[ixN]=(fx2-fx1)/(2.0*h)
            
        else:
            xAdj[ixN]=X[ixN] + h
            fx2=EV_F_post(xAdj, kap, n_agt)
            
            xAdj[ixN]=X[ixN]
            fx1=EV_F_post(xAdj, kap, n_agt)
            GRAD[ixN]=(fx2-fx1)/h
            
    return GRAD
       
#======================================================================

#======================================================================
#   Equality constraints during the VFI of the model
def EV_G_post(X, kap):
    
    M=n_ctt * Delta
    G=np.empty(M, float)

    #s = (1,n_agt)
    for t in range(Delta):
        var[t] = X[: t * n_agt]
    # pull in constraints
    e_ctt = equations_post.f_ctt(var[t], kap)
    # apply all constraints with this one loop
    for iter in ctt_key:
        G[I_ctt[iter]] = e_ctt[iter]

    return G
  
#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
#   during i_pth  
  
def EV_JAC_G_post(X, flag, kap, t):
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
        gx1=EV_G_post(X, kap, t)

        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G_post(xAdj, kap, t)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A
    
#======================================================================

class ipopt_class_inst(): 
    """
    Class for the optimization problem to be passed to cyipopt 
    Further optimisations may be possible here by including a hessian (optional param) 
    Uses the existing instance of the Gaussian Process (GP OLD) 
    """

    def __init__(self, X, n_agents, k_init, NELE_JAC, NELE_HESS=None, verbose=False): 
        self.x = X 
        self.n_agents = n_agents 
        self.k_init = k_init 
        self.NELE_JAC = NELE_JAC 
        self.NELE_HESS = NELE_HESS 
        self.initial = initial 
        self.verbose = verbose 

    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent 
    def eval_f(self, x): 
        return EV_F_post(x, self.k_init, self.n_agents) 

    def eval_grad_f(self, x): 
        return EV_GRAD_F_post(x, self.k_init, self.n_agents) 

    def eval_g(self, x): 
        return EV_G_post(x, self.k_init, self.n_agents) 

    def eval_jac_g(self, x, flag):
        return EV_JAC_G_post(x, flag, self.k_init, self.n_agents) 

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
        """Prints information at every Ipopt i_pth."""

        if self.verbose: 
            msg = "Objective value at i_pth #{:d} is - {:g}"
            print(msg.format(iter_count, obj_value))
