from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import jit, grad, jacrev, jacfwd
from cyipopt import minimize_ipopt

#-----------basic economic parameters
NREG = 4        # number of regions
NSEC = 6        # number of sectors
NTIM = LFWD = 10# look-forward parameter / time horizon length (Delta_s)
NITR = LPTH = 28# path length (Tstar): number of random steps along given path
NPTH = 1        # number of paths (in basic example Tstar + 1)
BETA = 99e-2    # discount factor
ZETA0 = 1       # output multiplier in status quo state 0
ZETA1 = 95e-2   # output multiplier in tipped state 1
PHIG = 5e-1     # adjustment cost multiplier
PHIK = 5e-1     # weight of capital in production # alpha in CJ
TPT = 1e-2      # transition probability of tipping (from state 0 to 1)
GAMMA = 5e-1    # power utility exponent
DELTA = 25e-3   # depreciation rate
ETA = 5e-1      # Frisch elasticity of labour supply
RHO = np.ones(NREG) # regional weights (population)
TCS=0.75
#-----------suppressed basic parameters
#PHIM = 5e-1    # weight of intermediate inputs in production
#XI = np.ones(NRxS) * 1 / NRxS # importance of kapital input to another
#MU = np.ones(NRxS) * 1 / NRxS # importance of one sector to another
#-----------derived economic parameters
NRxS = NREG * NSEC
GAMMAhat = 1 - 1 / GAMMA    # utility parameter (consumption denominator)
ETAhat = 1 + 1 / ETA        # utility parameter
PHIL = 1 - PHIK             # labour's importance in production
DPT = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # deterministic productivity trend
RWU = (1 - PHIK) * A * (A - DELTA) ** (-1 / GAMMA) # relative weight of con and lab in utility
ZETA = np.array([ZETA0, ZETA1])
#-----------structure of x
#-------first in latex notation:
#---x = [
#        x_{p0, t0, s0, r0}, x_{p0, t0, s0, r1}, x_{p0, t0, s0, r2},
#        x_{p0, t0, s1, r0}, x_{p0, t0, s1, r1}, x_{p0, t0, s1, r2},
#        x_{p0, t1, s0, r0}, x_{p0, t1, s0, r1}, x_{p0, t1, s0, r2},
#        x_{p0, t1, s1, r0}, x_{p0, t1, s1, r1}, x_{p0, t1, s1, r2},
#        x_{p1, t0, s0, r0}, x_{p1, t0, s0, r1}, x_{p1, t0, s0, r2},
#        x_{p1, t0, s1, r0}, x_{p1, t0, s1, r1}, x_{p1, t0, s1, r2},
#        x_{p1, t1, s0, r0}, x_{p1, t1, s0, r1}, x_{p1, t1, s0, r2},
#        x_{p1, t1, s1, r0}, x_{p1, t1, s1, r1}, x_{p1, t1, s1, r2},
#       ]
#-------construct dicts for Policies, Time, Sectors and Regions respectively:
PD = dict()
for p in range(NPOL):
    PD[p] = range(len(x))[p * NREG * NSEC * NTIM : \
                          (p + 1) * NREG * NSEC * NTIM : \
                          1]
TD = dict()
for t in range(NTIM):
    indlist = []
    for p in range(NPOL):
        indlist += range(len(x))[p * NREG * NSEC * NTIM + \
                                 t * NSEC * NREG : \
                                 p * NREG * NSEC * NTIM + \
                                 (t + 1) * NSEC * NREG : \
                                 1]
    TD[t] = indlist
SD = dict()
for s in sectNames: #comment
    indlist = []
    for t in range(NTIM):
        for p in range(NPOL):
            indlist += range(len(x))[p * NREG * NSEC * NTIM + \
                                     t * NREG * NSEC + \
                                     s * NREG : \
                                     p * NREG * NSEC * NTIM + \
                                     t * NREG * NSEC + \
                                     (s + 1) * NREG : \
                                     1]
    SD[t] = indlist
#----the final one can be done with a slicer with stride NREG
RD = dict()
for r in range(NREG)::
    indlist = []
    for s in range(NSECt):
        for t in range(NTIM):
            for p in range(NPOL):
                indlist += range(len(x))[p * NREG * NSEC * NTIM + \
                                         t * NREG * NSEC + \
                                         s * NREG + \
                                         r : \
                                         p * NREG * NSEC * NTIM + \
                                         t * NREG * NSEC + \
                                         s * NREG + \
                                         (r + 1) : \
                                         1]
    RD[r] = indlist


#-----------objective function
#con_weights=GAMMA, elast_par=RHO, inv_elast_par=RHO_inv):
def objective(x,         # full NPOLxLFWDxNSECxNREG vector of variables
              beta=BETA, # discount factor
              lfwd=LFWD, # look-forward parameter
              npol=NPOL, # number of policy-variable types (con, lab, etc)
              ):
    # locate tail kapital in x
    var_fin = x[(LFWD - 1) * NPOL : LFWD * NPOL : 1]
    kap_tail = var_fin[I["knx"]]
    # locate consumption in x
    sum_utl = 0.0
    for t in range(LFWD):
        var = x[t * NPOL : (t + 1) * NPOL]
        sum_utl += (BETA ** t * utility(var[I["con"]], var[I["lab"]]))
    val = sum_utl + BETA ** LFWD * V_tail(kap_tail)
    return

#-----------equality constraints
def eq_constraints(x,
                   state,
                   par=THETA,
                   dicts=DICTS
                   ):
    eqns = np.zeros(4)
        # constraint 1
    eqns = eqns.at[0].set(income - np.inner(kap, x))
        # constraint 2
    for i in range(LFWD):
        eqns  = eqns.at[i].set(np.power(x[i] - i, 2))
    return eqns

#-----------build and jit "objective" derivatives 
obj = jit(objective)
obj_grad = jit(grad(obj))
obj_hess = jit(jacrev(jacfwd(obj)))
#-----------build and jit "eq_constraints" derivatives
eq_ctt = jit(lambda x, state: eq_constraints(x, state))
eq_ctt_jac = jit(jacfwd(eq_ctt))
eq_ctt_hess = jacrev(jacfwd(eq_ctt))
#-----------build and jit "ineq_constraints" derivatives
#con_ineq = jit(ineq_constraints)
#ineq_ctt_jac = jit(jacfwd(ineq_ctt))
#ineq_ctt_hess = jacrev(jacfwd(ineq_ctt))  # hessian
#ineq_ctt_hessvp = jit(lambda x, v: ineq_ctt_hess(x) * v[0]) # hessian vector-product

#-----------define the jitted-state-in functions for the loop
def eq_ctt_js(state):
    return lambda x: eq_ctt(x, state)
def eq_ctt_jac_js(state):
    return lambda x: eq_ctt_jac(x, p)
def eq_ctt_hess_js(state):
    return lambda x: eq_ctt_hess(x, state)

#-----------variable bounds:
    bnds = [(0.1, 100.0) for _ in range(x0.size)]
#-*-*-*-*-*-loop/iteration along path starts here 
res = dict()
for s in range(1, LPTH):
    #-------feed in kapital from starting point s-1
    eq_ctt_fin = eq_ctt_js(state=res[s-1][I["knx"]])
    eq_ctt_jac_fin = eq_ctt_jac_js(state=d[s])
    eq_ctt_hess_fin = eq_ctt_hess_js(state=d[s])
    #!!-----create and jit the hessian vector product-more explanation needed!!
    eq_ctt_hessvp = jit(lambda x, v: eq_ctt_hess(x) * v[0]) # hessian vector-product
    #-------wrap up constraints for cyipopt
    cons = [{'type': 'eq',
             'fun': eq_ctt_fin,
             'jac': eq_ctt_jac,
             'hess': eq_ctt_hessvp,
             }]
    #-----------starting point (absent warm start)
    x0 = np.array([15.0, 14.0, 13.0, 12.0])

    #-----------execute solver
    res[s] = minimize_ipopt(obj,
                            jac=obj_grad,
                            hess=obj_hess,
                            x0=x0,
                            bounds=bnds,
                            constraints=cons,
                            options={'disp': 1, # printout range:0-12, default=5
                                     'obj_scaling_factor': -1.0, # maximize obj
                                     'timing_statistics': 'yes',
                                     'print_timing_statistics': 'yes',
                                     #!!how to warm start? see ipopt options page!!
                                     #'warm_start_init_point': 'yes', 
                                     #!!next one for "multiple problems in one nlp"!!
                                     #'warm_start_same_structure': 'yes',
                                     }
                            )
#-*-*-*-*-*-loop ends here

# ----------- print solution, etc.
for s in range(len(res)):
    print("the solution for iterate ", s, "is ", res[s])
    x_sol[s] = res[s]["x"]
    print("sol=", x_sol[s])
    for i in range(len(x_sol[s])-1):
        print("diff=", x_sol[s][i] - x_sol[s][i+1])
