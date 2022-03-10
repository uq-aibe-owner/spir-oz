from casadi import SX, MX, DM, Function, nlpsol, vertcat, sum1
import numpy as np


#==============================================================================
#-----------parameters
#------------------------------------------------------------------------------
#-----------economic parameters
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
TCS = 75e-2       # Tail Consumption Share
#-----------suppressed basic parameters
#PHIM = 5e-1    # weight of intermediate inputs in production
#XI = np.ones(NRxS) * 1 / NRxS # importance of kapital input to another
#MU = np.ones(NRxS) * 1 / NRxS # importance of one sector to another

#------------------------------------------------------------------------------
#-----------derived economic parameters
NRxS = NREG * NSEC
NSxT = NSEC * NTIM
NRxSxT = NREG * NSEC * NTIM
NMCL = NSxT # may add more eg. electricity markets are specific
GAMMAhat = 1 - 1 / GAMMA    # utility parameter (consumption denominator)
ETAhat = 1 + 1 / ETA        # utility parameter
PHIL = 1 - PHIK             # labour's importance in production
DPT = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # deterministic prod trend
RWU = (1 - PHIK) * DPT * (DPT - DELTA) ** (-1 / GAMMA) # Rel Weight in Utility
ZETA = np.array([ZETA0, ZETA1])
NVAR = NPOL * NTIM * NREG * NSEC    # total number of variables
X0 = np.ones(NVAR)          # our initial warm start 
# k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
KAP0 = np.ones(NRxS) # how about NSEC ????
#for j in range(n_agt):
#    KAP0[j] = np.exp(
#        np.log(kap_L) + (np.log(kap_U) - np.log(kap_L)) * j / (n_agt - 1)
#    )
#-----------suppressed derived economic parameters
#IVAR = np.arange(0,NVAR)    # index set (as np.array) for all variables

#------------------------------------------------------------------------------
#-----------probabity of no tip by time t as a pure function
#-----------requires: "import economic_parameters as par"
def prob_no_tip(
    tim, # time step along a path
    tpt=TPT, # transition probability of tipping
):
    return (1 - tpt) ** tim

#-----------prob no tip as a vec of parameters
PNT = np.ones(LPTH)
for t in range(LPTH):
    PNT[t] = prob_no_tip(t)

def E_zeta(
    
):

    val = zeta[1] + pnot(tim) * (zeta[0] - zeta[1]) # expected shock
    return val

#-----------if t were a variable, then, for casadi, we could do:
#t = c.SX.sym('t')
#pnt = c.Function('cpnt', [t], [prob_no_tip(t)], ['t'], ['p0'])
#PNT = pnt.map(LPTH)         # row vector 

#-----------For every look-forward, initial kapital is a parameter. 
#-----------To speed things up, we feed it in in CasADi-symbolic form:
kap = SX.sym('kap', NSEC * NREG)

#==============================================================================
#-----------variables: these are symbolic expressions of casadi type MX or SX
#------------------------------------------------------------------------------
con = SX.sym('con', NRxSxT)
lab = SX.sym('lab', NRxSxT)
knx = SX.sym('knx', NRxSxT)

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
    val = sum1(rho * (2 * con ** gh / gh - B * lab ** eh / eh))
    return val
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

#==============================================================================
#-----------expected output as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def expected_output(kap,                    # kap vector of vars
                    lab,                    # lab vector of vars
                    tim,                    # time step along a path
                    A=DPT,                  # determistic prod trend
                    phik=PHIK,              # weight of kap in prod
                    phil=PHIL,              # weight of lab in prod
                    E_zeta=E_ZETA,          # shock-value vector
                    pnot=prob_no_tip,       # prob no tip by t
                    ):
    y = A * (kap ** phik) * (lab ** phil)   # output
    val = E_zeta * y
    return val
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
g0 = z + x + y - 2
g1 = z + (1 - x) ** p[0] - y
ctt = Function(
    'ctt',
    [x, y, z],
    [g0, g1]
)
G = SX.zeros(NCTT)
G[0] = g0
G[1] = g1
#G = MX(g0, g1)

nlp = {
    'x' : vertcat(x, y, z),
    'f' : p[2] * (x ** p[0] + p[1] * z ** p[0]),
    'g' : G, #[g1, g0], #z + (1 - x) ** p[0] - y, #ctt,
    'p' : p,
}

#==============================================================================
#-----------options for the ipopt (the solver)  and casadi (the frontend)
#------------------------------------------------------------------------------
ipopt_opts = {
    'ipopt.linear_solver' : 'mumps', #default=Mumps
    'ipopt.obj_scaling_factor' : 1.0, #default=1.0
    'ipopt.warm_start_init_point' : 'yes', #default=no
    'ipopt.warm_start_bound_push' : 1e-9,
    'ipopt.warm_start_bound_frac' : 1e-9,
    'ipopt.warm_start_slack_bound_push' : 1e-9,
    'ipopt.warm_start_slack_bound_frac' : 1e-9,
    'ipopt.warm_start_mult_bound_push' : 1e-9,
    'ipopt.fixed_variable_treatment' : 'relax_bounds', #default=
    'ipopt.print_info_string' : 'yes', #default=no
    'ipopt.accept_every_trial_step' : 'no', #default=no
    'ipopt.alpha_for_y' : 'primal', #default=primal, try 'full'?
}
casadi_opts = {
    'calc_lam_p' : False,
}
opts = ipopt_opts | casadi_opts
#-----------when HSL is available, we should also be able to run:
#opts = {"ipopt.linear_solver" : "MA27"}
#-----------or:
#opts = {"ipopt.linear_solver" : "MA57"}
#-----------or:
#opts = {"ipopt.linear_solver" : "HSL_MA86"}
#-----------or:
#opts = {"ipopt.linear_solver" : "HSL_MA97"}

#-----------the following advice comes from https://www.hsl.rl.ac.uk/ipopt/
#-----------"when using HSL_MA86 or HSL_MA97 ensure MeTiS ordering is 
#-----------compiled into Ipopt to maximize parallelism"

solver = nlpsol('solver', 'ipopt', nlp, opts)

X0 = 100 * np.arange(1,4)
LBX = np.zeros(len(X0))
UBX = np.ones(len(X0)) * 1e+3
LBG = np.zeros(NCTT)
UBG = np.zeros(NCTT)
P0 = np.array([2, 100, 1])

arg = dict()

def exclude_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

res = dict()

arg[0] = {
    "x0" : X0,
    "p" : P0,
    'lbx' : LBX,
    'ubx' : UBX,
    'lbg' : LBG,
    'ubg' : UBG,
}
#-----------solve:
res[0] = solver.call(arg[0])

#-----------obvious speed up when we feed in a similar solution:
arg[1] = exclude_keys(arg[0], {'p', 'x0', 'lam_g0'})
arg[1]['x0'] = res[0]['x']
arg[1]['p'] = P0
arg[1]['lam_g0'] = res[0]['lam_g']
#-----------solve:
res[1] = solver.call(arg[1])

#-----------vs when we don't:
arg[2] = exclude_keys(arg[0], {'p', 'x0', 'lam_g0'})
arg[2]['p'] = P0
arg[2]['x0'] = np.repeat(100, len(X0))
#-----------solve:
res[2] = solver.call(arg[2])

#==============================================================================
#-----------print results
for s in range(len(res)):
    print('the full dict of results for step', s, 'is\n', res[s])
