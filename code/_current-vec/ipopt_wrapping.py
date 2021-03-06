# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from variables import *
import equations

#=======================================================================
#   Objective Function to start VFI (in our case, the value function)
        
def EV_F(X, kap, n_agt):
    
    """ # Extract Variables
    # this loop extracts the variables more expandably than doing them individualy as before
    for iter in i_pol_key:
        # forms the  2d intermediate variables into globals of the same name but in matrix form
        if d_pol[iter] == 2:
            globals()[iter] = np.zeros((n_agt,n_agt))
            for row in range(n_agt):
                for col in range(n_agt):
                    globals()[iter][row,col] = X[I[iter][0]+col+row*n_agt]
        else:
            # forms the 1d intermediate variables into globals of the same name in vector(list) form
            globals()[iter] = [X[ring] for ring in I[iter]] """
    """ val = X[I["val"]]
    V_old1 = V_INFINITY(X[I["knx"]])
    # Compute Value Function
    VT_sum=utility(X[I["con"]], X[I["lab"]]) + beta*V_old1 """
       
    return X[I["utl"]] + beta*X[I["val"]]

#=======================================================================
    
#=======================================================================
#   Computation of gradient (first order finite difference) of initial objective function 

def EV_GRAD_F(X, kap, n_agt):
    
    N=len(X)
    GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
    h=1e-4
    
    for ixN in range(N):
        xAdj=np.copy(X)
        
        if (xAdj[ixN] - h >= 0):
            xAdj[ixN]=X[ixN] + h            
            fx2=EV_F(xAdj, kap, n_agt)
            
            xAdj[ixN]=X[ixN] - h
            fx1=EV_F(xAdj, kap, n_agt)
            
            GRAD[ixN]=(fx2-fx1)/(2.0*h)
            
        else:
            xAdj[ixN]=X[ixN] + h
            fx2=EV_F(xAdj, kap, n_agt)
            
            xAdj[ixN]=X[ixN]
            fx1=EV_F(xAdj, kap, n_agt)
            GRAD[ixN]=(fx2-fx1)/h
            
    return GRAD
    
#=======================================================================
       
#======================================================================
#   Equality constraints for the first time step of the model
      
def EV_G(X, kap, n_agt):
    M=n_ctt
    G=np.empty(M, float)

    s = (1,n_agt)
    kap2 = np.zeros(s)
    kap2[0,:] = X[I["knx"]]

    """ print("should be the same")
    #print(type(X[I["knx"]]))
    print(np.shape(X[I["knx"]]))
    #print(type(kap2))
    print(np.shape(kap2)) """

    # pull in constraints
    e_ctt =  equations.f_ctt(X, kap2, 1, kap)
    # apply all constraints with this one loop
    for iter in ctt_key:
        G[I_ctt[iter]] = e_ctt[iter]

    return G

#======================================================================

#======================================================================
#   Computation (finite difference) of Jacobian of equality constraints 
#   for first time step
    
def EV_JAC_G(X, flag, kap, n_agt):
    N=n_pol
    M=n_ctt
    #print(N, "  ",M) #testing testing
    NZ=n_pol*n_ctt # J - could it be this?
    A=np.empty(NZ, float)
    ACON=np.empty(NZ, int) # its cause int is already a global variable cause i made it
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
        gx1=EV_G(X, kap, n_agt)
        
        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G(xAdj, kap, n_agt)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A
  
#======================================================================

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
        self.gp_old = gp_old 
        self.initial = initial 
        self.verbose = verbose 

    # Create ev_f, eval_f, eval_grad_f, eval_g, eval_jac_g for given k_init and n_agent 
    def eval_f(self, x): 
        return EV_F(x, self.k_init, self.n_agents)

    def eval_grad_f(self, x): 
        return EV_GRAD_F(x, self.k_init, self.n_agents)  
        
    def eval_g(self, x):
        return EV_G(x, self.k_init, self.n_agents, self.gp_old)  
        
    def eval_jac_g(self, x, flag):
        return EV_JAC_G(x, flag, self.k_init, self.n_agents, self.gp_old)

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
