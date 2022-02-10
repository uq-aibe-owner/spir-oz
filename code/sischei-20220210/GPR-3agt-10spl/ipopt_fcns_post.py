#=======================================================================
#
#     ipopt_wrapper.py : an interface to IPOPT and PYIPOPT 
#
#     Simon Scheidegger, 06/17
#
#     Cameron Gordon, 11/21 - adj file name to avoid issues with cyipopt 
#=======================================================================

from parameters import *
from econ import *
import numpy as np

#=======================================================================
#Terminal value function post tip
# V infinity
def V_tail_post(k=[]):
    e=np.ones(len(k))
    c=output_f(k,e)
    v_infinity=utility(c,e)/(1-beta)
    return v_infinity

#=======================================================================
#   Objective Function post tip
    
def EV_F_post(X, k_init, n_agents, gp_old):
    
    # Extract Variables
    cons = X[0:n_agents]
    lab = X[n_agents:2*n_agents]
    inv = X[2*n_agents:3*n_agents]
  
    knext = (1-delta) * k_init + inv

    # initialize correct data format for training point
    s = (1,n_agents)
    Xtest = np.zeros(s)
    Xtest[0,:] = knext
    
    ### value function needs to have the full sum over Dt=30 periods: for each s
    ### thus create a vector of utilities
    u = np.zeros(Tstar)
    u = utility(cons=[], lab=[])
    ### the objective fcn itself
    VT_sum = sum(u) + beta**30 * V_tail_post
    
    return VT_sum
    
#=======================================================================
#   Computation of gradient (first order finite difference) of the objective function 
    
def EV_GRAD_F_post(X, k_init, n_agents, gp_old):
    
    N=len(X)
    GRAD=np.zeros(N, float) # Initial Gradient of Objective Function
    h=1e-4
    
    
    for ixN in range(N):
        xAdj=np.copy(X)
        
        if (xAdj[ixN] - h >= 0):
            xAdj[ixN]=X[ixN] + h            
            fx2=EV_F_post(xAdj, k_init, n_agents, gp_old)
            
            xAdj[ixN]=X[ixN] - h
            fx1=EV_F_post(xAdj, k_init, n_agents, gp_old)
            
            GRAD[ixN]=(fx2-fx1)/(2.0*h)
            
        else:
            xAdj[ixN]=X[ixN] + h
            fx2=EV_F_post(xAdj, k_init, n_agents, gp_old)
            
            xAdj[ixN]=X[ixN]
            fx1=EV_F_post(xAdj, k_init, n_agents, gp_old)
            GRAD[ixN]=(fx2-fx1)/h
            
    return GRAD
           
#======================================================================
#   Equality constraints during the VFI of the model

def EV_G_post(X, k_init, n_agents):
    N=len(X)
    M=3*n_agents+1  # number of constraints
    G=np.empty(M, float)
    
    # Extract Variables
    cons=X[:n_agents]
    lab=X[n_agents:2*n_agents]
    inv=X[2*n_agents:3*n_agents]
    
    
    # first n_agents equality constraints
    for i in range(n_agents):
        G[i]=cons[i]
        G[i + n_agents]=lab[i]
        G[i+2*n_agents]=inv[i]
    
    
    f_prod=output_f(k_init, lab)
    sectors_sum=cons + inv - delta*k_init - (f_prod - Gamma_adjust)
    G[3*n_agents]=np.sum(sectors_sum)
    
    return G

#======================================================================
  
def EV_JAC_G_post(X, flag, k_init, n_agents):
    N=len(X)
    M=3*n_agents+1
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
        gx1=EV_G_post(X, k_init, n_agents)
        
        for ixM in range(M):
            for ixN in range(N):
                xAdj=np.copy(X)
                xAdj[ixN]=xAdj[ixN]+h
                gx2=EV_G_post(xAdj, k_init, n_agents)
                A[ixN + ixM*N]=(gx2[ixM] - gx1[ixM])/h
        return A    
    
#======================================================================

    
    
    
    
    
    
    
    
    
            
            
            
    
    
    
    
    
    
