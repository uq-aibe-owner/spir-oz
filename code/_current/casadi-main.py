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

#==============================================================================
#-----------uncertainty
#------------------------------------------------------------------------------
#-----------probabity of no tip by time t as a pure function
#-----------requires: "import economic_parameters as par"
def prob_no_tip(tim, # time step along a path
                tpt=TPT, # transition probability of tipping
                ):
    return (1 - tpt) ** tim

#-----------prob no tip as a vec of parameters
PNT = np.ones(LPTH)
for t in range(LPTH):
    PNT[t] = prob_no_tip(t)

#-----------expected shock
def E_zeta(t,
           zeta=ZETA,
           pnot=prob_no_tip,
           ):
    val = zeta[1] + pnot(t) * (zeta[0] - zeta[1])
    return val

E_ZETA = np.ones(LFWD)
for t in range(LFWD):
    E_ZETA[t] = E_zeta(t)

#-----------if t were a variable, then, for casadi, we could do:
#t = c.SX.sym('t')
#pnt = c.Function('cpnt', [t], [prob_no_tip(t)], ['t'], ['p0'])
#PNT = pnt.map(LPTH)         # row vector 

#-----------For every look-forward, initial kapital is a parameter. 
#-----------To speed things up, we feed it in in CasADi-symbolic form:
kap = SX.sym('kap', NSEC * NREG)
zeta= SX.sym('zeta', LFWD)

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
def instant_utility(
        con,            # consumption vec of vars at given time
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
def V_tail(
        kap,             # kapital vec of vars at time t=LFWD 
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
def E_output(
        kap,                    # kap vector of vars at time t
        lab,                    # lab vector of vars at time t
        E_shock,                # shock or expected shock at time t
        A=DPT,                  # determistic prod trend
        phik=PHIK,              # weight of kap in prod
        phil=PHIL,              # weight of lab in prod
):
    y = A * (kap ** phik) * (lab ** phil)   # output
    val = E_shock * y
    return val

#==============================================================================
#-----------adjustment costs of investment as a pure function
#-----------requires: "import economic_parameters as par"
def adjustment_cost(
        kap,
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
def market_clearing(
        con,
        kap,
        knx,
        lab,
        E_shock,
        delta=DELTA,
        adjc=adjustment_cost, # Gamma in Cai-Judd
        E_f=E_output,
):
    sav = knx - (1 - delta) * kap
    val = sum(E_f(kap, lab, E_shock) - con - sav - adjc(kap, sav))
    return val

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

d_ind_p = {
    'kap'       : range(NRxS),
    'start'     : range(NRxS),
    'shock'     : range(NRxS, NRxS + NTIM),
    t           : [NRxS + t],
}

#==============================================================================
#-----------function for returning index subsets of x for a pair of dict keys
def sub_ind_x(
        key1,             # any key of d_ind_x
        key2,             # any key of d_ind_x
        d=d_ind_x,   # dict of index categories: pol, time, sec, reg
):
    val = np.array(list(set(d[key1]) & set(d[key2])))
    return val
#j_sub_ind_x = jit(sub_ind_x)
# possible alternative: ind(ind(ind(range(len(X0)), key1),key2), key3)

#-----------function for intersecting two lists: returns indices as np.array
#def f_I2L(list1,list2):
#    return np.array(list(set(list1) & set(list2)))

#-----------function for returning index subsets of p for a pair of dict keys
def sub_ind_p(
        key1,             # any key of d_ind_p
        key2,             # any key of d_ind_p
        d=d_ind_p,   # dict of index categories: kap and zeta 
):
    val = np.array(list(set(d[key1]) & set(d[key2])))
    return val
#j_sub_ind_p = jit(sub_ind_p)
# possible alternative: ind(ind(ind(range(len(X0)), key1),key2), key3)

#-----------function for intersecting two lists: returns indices as np.array
#def f_I2L(list1,list2):
#    return np.array(list(set(list1) & set(list2)))

#==============================================================================
#-----------objective function (purified)
def objective(
        x,                  # full vector of variables
        beta=BETA,          # discount factor
        lfwd=LFWD,          # look-forward parameter
        ind=sub_ind_x,      # subindices function for x: req. two keys
        ind_p=sub_ind_p,
        u=instant_utility,# utility function representing flow per t
        v=V_tail,           # tail-sum value function
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
#-----------alt subindex function
# allows you to take a subset from a set you already have using one key
def subset(
        set1, # A set we already have
        key,  # A key we want to use to subset the data further
        d=d_ind_x  # combined dict
):
    val = np.array(list(set(set1) & set(d[key])))
    return val

#----------- another alt subindex function
# could make the 
# allows you to feed a vector of an arbitrary number of keys to get a subset of X
def subset_adapt(
        keys,  # A vector of all the keys we want to use to subset X, can be any length greater than or equal to 1
        d=d_ind_x  # combined dict
):
    inds = d[keys[0]] # get our first indices
    for i in range(1,len(keys)):
        inds = np.array(list(set(inds) & set(d[keys[i]]))) # subset what we already have with the next key we want to subset by
    return inds

#==============================================================================
#-----------equality constraints
def eq_constraints(
        x,                      #casadi vec of variables
        p,                      #casadi vec of parameters
        lfwd=LFWD,
        ind=sub_ind_x,
        ind_p=sub_ind_p,
        mcl=market_clearing,
):
    eqns = np.zeros(lfwd)
    for t in range(lfwd):
        if t == 0:
            KAP = p[ind_p('kap', 'start')]
        else:
            KAP = x[ind("knx", t - 1)]
        KNX = x[ind("knx", t)]
        CON = x[ind("con", t)]
        LAB = x[ind("lab", t)]
        E_SHOCK = p[ind_p("shock", t)]
        eqns[t] = mcl(
                    con=CON,
                    knx=KNX,
                    lab=LAB,
                    kap=KAP,
                    E_shock=E_SHOCK,
        )
    return eqns

#==============================================================================
#-----------dict of arguments for the function casadi.nlpsol
nlp = {
    'x' : vertcat(con, knx, lab),
    'p' : vertcat(kap, zeta),
    'f' : objective(nlp['x']),
    'g' : eq_constraints(nlp['x'], nlp['p']),
}

#==============================================================================
#-----------options for the ipopt (the solver) and casadi (the frontend)
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

#==============================================================================
#-----------A casadi function for us to feed in initial conditions and call:
solver = nlpsol('solver', 'ipopt', nlp, opts)

#==============================================================================
LBX = np.ones(len(X0)) * 1e-3
UBX = np.ones(len(X0)) * 1e+3
LBG = np.zeros(NCTT)
UBG = np.zeros(NCTT)
P0 = np.ones(NRxS + NTIM)
P0 = vertcat(KAP0, E_ZETA)
arg = dict()

#-----------a function for removing elements from a dict
#def exclude_keys(d, keys):
#    return {x: d[x] for x in d if x not in keys}

res = dict()

arg = {
    'lbx' : LBX,
    'ubx' : UBX,
    'lbg' : LBG,
    'ubg' : UBG,
}

#-*-*-*-*-*-loop/iteration along path starts here 
#------------------------------------------------------------------------------
res = dict()
for s in range(LPTH):
    #-------set initial capital and vector of shocks for each plan
    if s == 0:
        arg['p'] = P0
        arg['x0'] = X0
    else:
        arg['x0'] = np.array(res[s - 1]["x"])
        arg['p'] = vertcat(res[s - 1][sub_ind_x("knx", 0)], P0)
        arg['lam_g0'] = res[s - 1]['lam_g']

    #-----------execute solver
    res[s] = solver.call(arg)
#-*-*-*-*-*-loop ends here
#==============================================================================
#-----------print results
for s in range(len(res)):
    print('the full dict of results for step', s, 'is\n', res[s])
