from jax.config import config
config.update("jax_enable_x64", True)

#-----------global modules
import jax.numpy as np
from jax import jit, grad, jacrev, jacfwd
from cyipopt import minimize_ipopt
#-----------local modules
import economic_functions as efcn

#==============================================================================
#-------------economic parameters
#-----------basic economic parameters
NREG = 4        # number of regions
NSEC = 1        # number of sectors
PHZN = NTIM = LFWD = 10# look-forward parameter / planning horizon (Delta_s)
NPOL = 3        # number of policy types: con, lab, knx, #itm
NITR = LPTH = 28# path length (Tstar): number of random steps along given path
NPTH = 1        # number of paths (in basic example Tstar + 1)
BETA = 99e-2    # discount factor
ZETA0 = 1       # output multiplier in status quo state 0
ZETA1 = 95e-2   # output multiplier in tipped state 1
PHIA = 5e-1     # adjustment cost multiplier
PHIK = 5e-1     # weight of capital in production # alpha in CJ
TPT = 1e-2      # transition probability of tipping (from state 0 to 1)
GAMMA = 5e-1    # power utility exponent
DELTA = 25e-3   # depreciation rate
ETA = 5e-1      # Frisch elasticity of labour supply
RHO = np.ones(NREG) # regional weights (population)
TCS=75e-2       # Tail Consumption Share
#-----------suppressed basic parameters
#PHIM = 5e-1    # weight of intermediate inputs in production
#XI = np.ones(NRxS) * 1 / NRxS # importance of kapital input to another
#MU = np.ones(NRxS) * 1 / NRxS # importance of one sector to another
#-----------derived economic parameters
NRxS = NREG * NSEC
GAMMAhat = 1 - 1 / GAMMA    # utility parameter (consumption denominator)
ETAhat = 1 + 1 / ETA        # utility parameter
PHIL = 1 - PHIK             # labour's importance in production
DPT = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # deterministic prod trend
RWU = (1 - PHIK) * DPT * (DPT - DELTA) ** (-1 / GAMMA) # rel weight: c vs l in u
ZETA = np.array([ZETA0, ZETA1])
# k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
KAP0 = np.ones(NREG) # how about NSEC ????
#for j in range(n_agt):
#    KAP0[j] = np.exp(
#        np.log(kap_L) + (np.log(kap_U) - np.log(kap_L)) * j / (n_agt - 1)
#    )
#-----------suppressed derived economic parameters
NVAR = NPOL * LFWD * NSEC * NREG    # total number of variables
#IVAR = np.arange(0,NVAR)    # index set (as np.array) for all variables

#==============================================================================
#-----------structure of x using latex notation:
#---x = [
#        x_{p0, t0, r0, s0}, x_{p0, t0, r0, s1}, x_{p0, t0, r0, s2},
#
#        x_{p0, t0, r1, s0}, x_{p0, t0, r1, s1}, x_{p0, t0, r1, s2},
#
#        x_{p0, t1, r0, s0}, x_{p0, t1, r0, s1}, x_{p0, t1, r0, s2},
#
#        x_{p0, t1, r1, s0}, x_{p0, t1, r1, s1}, x_{p0, t1, r1, s2},
#
#        x_{p1, t0, r0, s0}, x_{p1, t0, r0, s1}, x_{p0, t0, r0, s2},
#
#        x_{p1, t0, r1, s0}, x_{p1, t0, r1, s1}, x_{p0, t0, r1, s2},
#
#        x_{p1, t1, r0, s0}, x_{p1, t1, r0, s1}, x_{p0, t1, r0, s2},
#
#        x_{p1, t1, r1, s0}, x_{p1, t1, r1, s1}, x_{p0, t1, r1, s2},
#
#
#        x_{p2, t0, r0, s00}, x_{p2, t0, r0, s01}, x_{p2, t0, r0, s02},
#        x_{p2, t0, r0, s10}, x_{p2, t0, r0, s11}, x_{p2, t0, r0, s12},
#        x_{p2, t0, r0, s20}, x_{p2, t0, r0, s21}, x_{p2, t0, r0, s22},
#
#        x_{p2, t0, r1, s00}, x_{p2, t0, r1, s01}, x_{p2, t0, r1, s02},
#        x_{p2, t0, r1, s10}, x_{p2, t0, r1, s11}, x_{p2, t0, r1, s12},
#        x_{p2, t0, r1, s20}, x_{p2, t0, r1, s21}, x_{p2, t0, r1, s22},
#
#        x_{p2, t1, r0, s00}, x_{p2, t1, r0, s01}, x_{p2, t1, r0, s02},
#        x_{p2, t1, r0, s10}, x_{p2, t1, r0, s11}, x_{p2, t1, r0, s12},
#        x_{p2, t1, r0, s20}, x_{p2, t1, r0, s21}, x_{p2, t1, r0, s22},
#
#        x_{p2, t1, r1, s00}, x_{p2, t1, r1, s01}, x_{p2, t1, r1, s02},
#        x_{p2, t1, r1, s10}, x_{p2, t1, r1, s11}, x_{p2, t1, r1, s12},
#        x_{p2, t1, r1, s20}, x_{p2, t1, r1, s21}, x_{p2, t1, r1, s22},
#
#
#        x_{p3, t0, r0, s00}, x_{p3, t0, r0, s01}, x_{p3, t0, r0, s02},
#        x_{p3, t0, r0, s10}, x_{p3, t0, r0, s11}, x_{p3, t0, r0, s12},
#        x_{p3, t0, r0, s20}, x_{p3, t0, r0, s21}, x_{p3, t0, r0, s22},
#
#        x_{p3, t0, r1, s00}, x_{p3, t0, r1, s01}, x_{p3, t0, r1, s02},
#        x_{p3, t0, r1, s10}, x_{p3, t0, r1, s11}, x_{p3, t0, r1, s12},
#        x_{p3, t0, r1, s20}, x_{p3, t0, r1, s21}, x_{p3, t0, r1, s22},
#
#        x_{p3, t1, r0, s00}, x_{p3, t1, r0, s01}, x_{p3, t1, r0, s02},
#        x_{p3, t1, r0, s10}, x_{p3, t1, r0, s11}, x_{p3, t1, r0, s12},
#        x_{p3, t1, r0, s20}, x_{p3, t1, r0, s21}, x_{p3, t1, r0, s22},
#
#        x_{p3, t1, r1, s00}, x_{p3, t1, r1, s01}, x_{p3, t1, r1, s02},
#        x_{p3, t1, r1, s10}, x_{p3, t1, r1, s11}, x_{p3, t1, r1, s12},
#        x_{p3, t1, r1, s20}, x_{p3, t1, r1, s21}, x_{p3, t1, r1, s22},
#       ]
#
#==============================================================================
#---------------dicts
#-----------dimensions for each pol var: 0 : scalar; 1 : vector; 2 : matrix
d_dim = {
    "con": 1,
    "knx": 1,
    "lab": 1,
}
i_pol = {
    "con": 0,
    "knx": 1,
    "lab": 2,
}
i_reg = {
    "aus": 0,
    "qld": 1,
    "wld": 2,
}
i_sec = {
    "agr": 0,
    #"for": 1,
    #"min": 2,
    #"man": 3,
    #"uty": 4,
    #"ctr": 5,
    #"com": 6,
    #"tps": 7,
    #"res": 8,
}
# Warm start
pol_S = {
    "con": 4,
    "lab": 1,
    "knx": KAP0,
    #"sav": 2,
    #"out": 6,
    #    "itm": 10,
    #    "ITM": 10,
    #    "SAV": 10,
    #"utl": 1,
    #    "val": -300,
}
#-----------dicts of index lists for locating variables in x:
#-------Dict for locating every variable for a given policy
d_pol_ind_x = dict()
for p in i_pol.values():
    stride = NTIM * NREG * (NSEC ** d_dim[p])
    start = i_pol[p] * stride
    end = start + stride
    d_pol_ind_x[p] = range(len(x))[start : end : 1]

#-------Dict for locating every variable at a given time
d_tim_ind_x = dict()
for t in range(NTIM):
    indlist = []
    for p in i_pol.values():
        stride = NREG * NSEC ** d_dim[p]
        start = (i_pol[p] * NTIM + t) * stride
        end = start + stride
        indlist += range(len(x))[start : end : 1]
    d_tim_ind_x[t] = sorted(indlist)

#-----------the final one can be done with a slicer with stride NREG
#-------Dict for locating every variable in a given region
d_reg_ind_x = dict()
for r in i_reg.values():
    indlist = []
    for t in range(NTIM):
        for p in i_sec.values():
            stride = NSEC ** d_dim[p]
            start = (p * NTIM * NREG + t * NREG + r) * stride
            end = start + stride
            indlist += range(len(x))[start : end : 1]
    d_reg_ind_x[r] = sorted(indlist)

#-------Dict for locating every variable in a given sector
d_sec_ind_x = dict()
for s in i_sec.values(): #comment
    indlist = []
    for r in i_reg.values():
        for t in range(NTIM):
            for p in i_sec.values():
                stride = 1
                start = (p * NTIM * NREG + t * NREG + r) * NSEC ** d_dim[p] + s
                end = start + stride
                indlist += range(len(x))[start : end : 1]
    d_sec_ind_x[s] = sorted(indlist)
#-----------union of all the "in_x" dicts: those relating to indices of x
d_ind_x = d_pol_ind_x | d_tim_ind_x | d_reg_ind_x | d_sec_ind_x

#==============================================================================
#-----------function for returning index subsets of x for a pair of dict keys
def sub_ind_x(key1,             # any key of d_ind_x
              key2,             # any key of d_ind_x
              d=d_ind_x,   # dict of index categories: pol, time, sec, reg
              ):
    val = np.array(list(set(d['key1']) & set(d['key2'])))
    return val

#-----------function for intersecting two lists: returns indices as np.array
#def f_I2L(list1,list2):
#    return np.array(list(set(list1) & set(list2)))
#==============================================================================
#-----------computational parameters
#dI_CON = dict()
#dI_LAB = dict()
#dI_KNX = dict()
#for t in range(LFWD):
#    dI_CON[t] = f_I2L(dI_P["con"], dI_T[t]) # Index set of con
#    dI_LAB[t] = f_I2L(dI_P["lab"], dI_T[t]) # Index set of lab
#    dI_KNX[t] = f_I2L(dI_P["knx"], dI_T[t]) # Index set of knx
#I_CON = np.array(dI_P["con"])]
#I_LAB = np.array(dI_P["lab"])]
#I_FNLKNX = f_I2L(dI_P["knx"], dI_T[PHZN])  # Index set for final knx in plan

#==============================================================================
#---------------functions
#-----------objective function (purified)
def objective(x,                    # full vector of variables
              beta=par.BETA,        # discount factor
              lfwd=par.LFWD,        # look-forward parameter
              ind=sub_ind_x,    # subindices function for x: req. two keys
              u=efcn.instant_utility,# utility function representing flow per t
              v=efcn.V_tail,         # tail-sum value function
              ):
    # extract/locate knx at the planning horizon in x
    kap_lfwd = x[ind("knx", lfwd)]
    # sum discounted utility over the planning horizon
    sum_disc_utl = 0.0
    for t in range(lfwd):
        CON = x[ind("con", t)]    # locate consumption at t in x
        LAB = x[ind("lab", t)]    # locate labour at t in x
        sum_disc_utl += beta ** t * u(con=CON, lab=LAB)
    val = sum_disc_utl + beta ** lfwd * v(kap=kap_lfwd)
    return val

#==============================================================================
#-----------equality constraints
def eq_constraints(x,
                   state,
                   lfwd=par.LFWD,
                   ind=sub_ind_x,
                   mcl=efcn.market_clearing,
                   ):
    eqns = np.zeros(lfwd)
        # constraint 2
    for t in range(lfwd):
        KAP = state
        KNX = x[ind("knx", t)]
        CON = x[ind("con", t)]
        LAB = x[ind("lab", t)]
        eqns  = eqns.at[t].set(mcl(kap=KAP, knx=KNX, con=CON, lab=LAB, tim=t))
    return eqns

#==============================================================================
#-------------build and jit (just-in-time compile) the derivatives
#-----------first the objective
obj = jit(objective)                                        # obj with one arg
obj_grad = jit(grad(obj))                                   # grad
obj_hess = jit(jacrev(jacfwd(obj)))                         # hessian
#-----------then the equality constraints
eq_ctt = jit(lambda x, state: eq_constraints(x, state))     # eq_ctt two-args
eq_ctt_jac = jit(jacfwd(eq_ctt))                            # jacobian
eq_ctt_hess = jacrev(jacfwd(eq_ctt))                        # hessian
#-----------then the inequality constraints
#con_ineq = jit(ineq_constraints)                           # ineq_ctt two-args 
#ineq_ctt_jac = jit(jacfwd(ineq_ctt))                       # jacobian
#ineq_ctt_hess = jacrev(jacfwd(ineq_ctt))                   # hessian
#ineq_ctt_hessvp = jit(lambda x, v: ineq_ctt_hess(x) * v[0])# hessian vec-prod

#==============================================================================
#-----------define the jitted-state-in functions for the loop
def eq_ctt_js(state):
    return lambda x: eq_ctt(x, state)
def eq_ctt_jac_js(state):
    return lambda x: eq_ctt_jac(x, p)
def eq_ctt_hess_js(state):
    return lambda x: eq_ctt_hess(x, state)

#==============================================================================
#-----------variable bounds:
bnds = [(0.1, 100.0) for _ in range(x0.size)]

#==============================================================================
#-*-*-*-*-*-loop/iteration along path starts here 
res = dict()
for s in range(1, par.LPTH):
    #-------set initial capital for each plan
    if s == 0:
        kap = par.KAP0
    else:
        kap = res[s - 1]["x"][ind("knx", 0)]
    #-----------feed in kapital from starting point s-1 
    eq_ctt_fin = eq_ctt_js(kap)
    eq_ctt_jac_fin = eq_ctt_jac_js(kap)
    eq_ctt_hess_fin = eq_ctt_hess_js(kap)
    #!!-----create and jit the hessian vector product-more explanation needed!!
    eq_ctt_hessvp = jit(lambda x, v: eq_ctt_hess(x) * v[0]) # hessian vec-prod
    #-------wrap up constraints for cyipopt
    cons = [{'type': 'eq',
             'fun': eq_ctt_fin,
             'jac': eq_ctt_jac,
             'hess': eq_ctt_hessvp,
             }]
    #-----------starting point (absent warm start)
    x0 = np.ones(par.NVAR)

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

#==============================================================================
# ----------- print solution, etc.
for s in range(len(res)):
    print("the solution for iterate ", s, "is ", res[s])
    x_sol[s] = res[s]["x"]
    print("sol=", x_sol[s])
    for i in range(len(x_sol[s])-1):
        print("diff=", x_sol[s][i] - x_sol[s][i+1])
