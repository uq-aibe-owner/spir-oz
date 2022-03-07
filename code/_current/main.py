from jax.config import config
config.update("jax_enable_x64", True)

#-----------global modules
import jax.numpy as np
from jax import jit, grad, jacrev, jacfwd
from cyipopt import minimize_ipopt
#-----------local modules
#import economic_functions as efcn

#==============================================================================
#-------------economic parameters
#-----------basic economic parameters
NREG = 2       # number of regions
NSEC = 2        # number of sectors
PHZN = NTIM = LFWD = 2# look-forward parameter / planning horizon (Delta_s)
NPOL = 3        # number of policy types: con, lab, knx, #itm
NITR = LPTH = 2# path length (Tstar): number of random steps along given path
NPTH = 1        # number of paths (in basic example Tstar + 1)
BETA = 95e-2    # discount factor
ZETA0 = 1       # output multiplier in status quo state 0
ZETA1 = 95e-2   # output multiplier in tipped state 1
PHIA = 5e-1     # adjustment cost multiplier
PHIK = 33e-2  # weight of capital in production # alpha in CJ
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
RWU = (1 - PHIK) * DPT * (DPT - DELTA) ** (-1 / GAMMA) # Rel Weight in Utility
ZETA = np.array([ZETA0, ZETA1])
NVAR = NPOL * LFWD * NSEC * NREG    # total number of variables
X0 = np.ones(NVAR)      # our initial warm start 
# k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
KAP0 = np.ones(NREG) # how about NSEC ????
#for j in range(n_agt):
#    KAP0[j] = np.exp(
#        np.log(kap_L) + (np.log(kap_U) - np.log(kap_L)) * j / (n_agt - 1)
#    )
#-----------suppressed derived economic parameters
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
    "for": 1,
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
for pk in i_pol.keys():
    p = i_pol[pk]
    d = d_dim[pk]
    stride = NTIM * NREG * NSEC ** d
    start = p * stride
    end = start + stride
    d_pol_ind_x[pk] = range(NVAR)[start : end : 1]

#-------Dict for locating every variable at a given time
d_tim_ind_x = dict()
for t in range(NTIM):
    indlist = []
    for pk in i_pol.keys():
        p = i_pol[pk]
        d = d_dim[pk]
        stride = NREG * NSEC ** d
        start = (p * NTIM + t) * stride
        end = start + stride
        indlist.extend(range(NVAR)[start : end : 1])
    d_tim_ind_x[t] = sorted(indlist)

#-----------the final one can be done with a slicer with stride NSEC ** d_dim
#-------Dict for locating every variable in a given region
d_reg_ind_x = dict()
for rk in i_reg.keys():
    r = i_reg[rk]
    indlist = []
    for t in range(NTIM):
        for pk in i_pol.keys():
            p = i_pol[pk]
            d = d_dim[pk]
            stride = NSEC ** d
            start = (p * NTIM * NREG + t * NREG + r) * stride
            end = start + stride
            indlist += range(NVAR)[start : end : 1]
    d_reg_ind_x[rk] = sorted(indlist)

#-------Dict for locating every variable in a given sector
d_sec_ind_x = dict()
for sk in i_sec.keys(): #comment
    s = i_sec[sk]
    indlist = []
    for rk in i_reg.keys():
        r = i_reg[rk]
        for t in range(NTIM):
            for pk in i_pol.keys():
                p = i_pol[pk]
                d = d_dim[pk]
                stride = 1
                start = (p * NTIM * NREG + t * NREG + r) * NSEC ** d + s
                end = start + stride
                indlist += range(NVAR)[start : end : 1]
    d_sec_ind_x[s] = sorted(indlist)

#-----------union of all the "in_x" dicts: those relating to indices of x
d_ind_x = d_pol_ind_x | d_tim_ind_x | d_reg_ind_x | d_sec_ind_x

#==============================================================================
#-----------function for returning index subsets of x for a pair of dict keys
def sub_ind_x(key1,             # any key of d_ind_x
              key2,             # any key of d_ind_x
              d=d_ind_x,   # dict of index categories: pol, time, sec, reg
              ):
    val = np.array(list(set(d[key1]) & set(d[key2])))
    return val
j_sub_ind_x = jit(sub_ind_x)
# possible alternative: ind(ind(ind(range(len(X0)), key1),key2), key3)

#-----------function for intersecting two lists: returns indices as np.array
#def f_I2L(list1,list2):
#    return np.array(list(set(list1) & set(list2)))

#==============================================================================
#---------------economic_functions
#------------------------------------------------------------------------------
#-----------instantaneous utility as a pure function
#-----------requires: "import economic_parameters as par"
def instant_utility(con,            # consumption vec of vars at given time
                    lab,            # labour vec of vars at given time
                    B=RWU,      # relative weight of con and lab in util
                    rho=RHO,    # regional-weights vec at given time
                    gh=GAMMAhat,
                    eh=ETAhat,
                    ):
    #-------log utility
    #val = np.sum(rho * (np.log(con) - B * np.log(lab)))
    #-------general power utility:
    val = np.sum(rho * (2 * con ** gh / gh - B * lab ** eh / eh))
    return val
j_instant_utility = jit(instant_utility)
#==============================================================================
#-----------v-tail as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def V_tail(kap,             # kapital vec of vars at time t=LFWD 
           A=DPT,       # deterministic productivity trend
           beta=BETA,   # discount factor
           phik=PHIK,   # weight of capital in production
           tcs=TCS,     # tail consumption share
           u=instant_utility, # utility function
           ):
    #-------tail consumption vec
    tail_con = tcs * A * kap ** phik
    #-------tail labour vec normalised to one
    tail_lab = np.ones(len(kap))
    val = u(tail_con, tail_lab) / (1 - beta)
    return val
j_V_tail = jit(V_tail)
#==============================================================================
#-----------probabity of no tip by time t as a pure function
#-----------requires: "import economic_parameters as par"
def prob_no_tip(tim, # time step along a path
               tpt=TPT, # transition probability of tipping
               ):
    return (1 - tpt) ** tim
j_prob_no_tip = jit(prob_no_tip)
#==============================================================================
#-----------expected output as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def expected_output(kap,                        # kap vector of vars
                    lab,                        # lab vector of vars
                    tim,                          # time step along a path
                    A=DPT,                  # determistic prod trend
                    phik=PHIK,              # weight of kap in prod
                    phil=PHIL,              # weight of lab in prod
                    zeta=ZETA,                  # shock-value vector
                    pnot=prob_no_tip,      # prob no tip by t
                    ):
    y = A * (kap ** phik) * (lab ** phil)       # output
    E_zeta = zeta[1] + pnot(tim) * (zeta[0] - zeta[1]) # expected shock
    val = E_zeta * y
    return val
j_expected_output = jit(expected_output)
#==============================================================================
#-----------adjustment costs of investment as a pure function
#-----------requires: "import economic_parameters as par"
def adjustment_cost(kap,
                    knx,
                    phia=PHIA, # adjustment cost multiplier
                    ):
    # since sav/kap - delta = (knx - (1 - delta) * kap)/kap - delta = ..
    # we can therefore rewrite the adjustment cost as
    val = phia * kap * np.square(knx / kap - 1)
    return 0
j_adjustment_cost = jit(adjustment_cost)
#==============================================================================
#-----------market clearing/budget constraint as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def market_clearing(kap,
                    knx,
                    con,
                    lab,
                    tim,
                    delta=DELTA,
                    adjc=adjustment_cost, # Gamma in Cai-Judd
                    E_f=expected_output,
                    ):
    sav = knx - (1 - delta) * kap
    val = sum(E_f(kap, lab, tim) - con - sav - adjc(kap, sav))
    return val
j_market_clearing = jit(market_clearing)
#==============================================================================
#-----------objective function (purified)
def objective(x,                    # full vector of variables
              beta=BETA,        # discount factor
              lfwd=LFWD,        # look-forward parameter
              ind=sub_ind_x,    # subindices function for x: req. two keys
              u=j_instant_utility,# utility function representing flow per t
              v=V_tail,         # tail-sum value function
              ):
    # extract/locate knx at the planning horizon in x
    kap_tail = x[ind("knx", lfwd - 1)]
    # sum discounted utility over the planning horizon
    sum_disc_utl = 0.0
    for t in range(lfwd):
        CON = x[ind("con", t)]    # locate consumption at t in x
        LAB = x[ind("lab", t)]    # locate labour at t in x
        sum_disc_utl += beta ** t * u(con=CON, lab=LAB)
    val = sum_disc_utl + beta ** lfwd * v(kap=kap_tail)
    return val
j_objective = jit(objective)
#==============================================================================
#-----------equality constraints
def eq_constraints(x,
                   state,
                   lfwd=LFWD,
                   ind=sub_ind_x,
                   mcl=j_market_clearing,
                   ):
    eqns = np.zeros(lfwd)
    for t in range(lfwd):
        if t == 0:
            KAP = state
        else:
            KAP = x[ind("knx", t-1)]
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
eq_ctt_hess = jit(jacrev(jacfwd(eq_ctt)))                        # hessian
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
    return lambda x: eq_ctt_jac(x, state)
def eq_ctt_hess_js(state):
    return lambda x: eq_ctt_hess(x, state)

#==============================================================================
#-----------variable bounds:
bnds = [(1e-3, 1e+3) for _ in range(NVAR)]

#==============================================================================
# Hessian vector product function
#def hvp(f, x, v):
#    return np.jvp(grad(f), primals, tangents)[1]
#def eq_ctt_hvp(x, v):
#    return hvp("constraints function", x, v)

#==============================================================================
#-*-*-*-*-*-loop/iteration along path starts here 
#------------------------------------------------------------------------------
res = dict()
for s in range(LPTH):
    #-------set initial capital for each plan
    if s == 0:
        KAP = KAP0
        x0 = X0
    else:
        X = np.array(res[s - 1]["x"])
        KAP = X[sub_ind_x("knx", 0)]
        x0 = X
    #-------feed in kapital from starting point s-1 
    eq_ctt_fin = jit(eq_ctt_js(state=KAP))
    eq_ctt_jac_fin = jit(eq_ctt_jac_js(state=KAP))
    eq_ctt_hess_fin = jit(eq_ctt_hess_js(state=KAP))
    #-------this returns a hessian for each constraint if v[0] != 0 
    eq_ctt_hessvp = jit(lambda x, v: eq_ctt_hess_fin(x) * v[0]) # hessian vec-prod
    #-------wrap up constraints for cyipopt
    cons = [{'type': 'eq',
             'fun': eq_ctt_fin,
             'jac': eq_ctt_jac_fin,
             'hess': eq_ctt_hessvp,
             }]
    #-------starting point (absent warm start)
    #x0 = X0

    #-----------execute solver
    res[s] = minimize_ipopt(obj,
                            jac=obj_grad,
                            hess=obj_hess,
                            x0=x0,
                            bounds=bnds,
                            constraints=cons,
                            #nele_jac=30,
                            options={'disp': 12, # printout range:0-12, default=5
                                     'obj_scaling_factor': -1.0, # maximize obj
                                     'timing_statistics': 'yes',
                                     'print_timing_statistics': 'yes',
                                     'constr_viol_tol': 5e-2,
                                     'max_iter': 1000,
                                     'acceptable_tol': 1e-4,
                                     #'dual_inf_tol': 0.5,
                                     #!!how to warm start? see ipopt options page!!
                                     #'warm_start_init_point': 'yes', 
                                     #!!next one for "multiple problems in one nlp"!!
                                     #'warm_start_same_structure': 'yes',
                                     }
                            )
    x_sol = np.array(res[s]["x"])
    for pk in d_pol_ind_x.keys():
        print("the solution for", pk, "at step", s, "along path 0 is \n", \
              x_sol[np.array(d_pol_ind_x[pk])])
#-*-*-*-*-*-loop ends here

#==============================================================================
# ----------- print solution, etc.
#for s in range(len(res)):
#    print("the solution for iterate ", s, "is ", res[s])
#    x_sol[s] = res[s]["x"]
#    print("sol=", x_sol[s])
#    for i in range(len(x_sol[s])-1):
#        print("diff=", x_sol[s][i] - x_sol[s][i+1])
