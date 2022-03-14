from casadi import  MX, SX, DM, Function, nlpsol, vertcat, sum1, dot
import numpy as np


#==============================================================================
#-----------parameters
#------------------------------------------------------------------------------
#-----------economic parameters
#-----------basic economic parameters
NREG = 3        # number of regions
NSEC = 1        # number of sectors
PHZN = NTIM = LFWD = 100# look-forward parameter / planning horizon (Delta_s)
NPOL = 3        # number of policy types: con, lab, knx, #itm
NITR = LPTH = 9 # path length (Tstar): number of random steps along given path
NPTH = 1        # number of paths (in basic example Tstar + 1)
BETA = 99e-2    # discount factor
ZETA0 = 1       # output multiplier in status quo state 0
ZETA1 = 95e-2   # output multiplier in tipped state 1
PHIA = 5e-1     # adjustment cost multiplier
PHIK = 33e-2    # weight of capital in production # alpha in CJ
TPT = 1e-2      # transition probability of tipping (from state 0 to 1)
GAMMA = 5e-1    # power utility exponent
DELTA = 25e-3   # depreciation rate
ETA = 5e-1      # Frisch elasticity of labour supply
RHO = DM.ones(NREG) # regional weights (population)
TCS = 75e-2     # Tail Consumption Share
#-----------suppressed basic parameters
#PHIM = 5e-1    # weight of intermediate inputs in production
#XI = np.ones(NRxS) * 1 / NRxS # importance of kapital input to another
#MU = np.ones(NRxS) * 1 / NRxS # importance of one sector to another

#------------------------------------------------------------------------------
#-----------derived economic parameters
NRxS = NREG * NSEC
NSxT = NSEC * NTIM
NRxSxT = NREG * NSEC * NTIM
NCTT = 2 * LFWD  # may add more eg. electricity markets are specific
GAMMA_HAT = 1 - 1 / GAMMA   # utility parameter (consumption denominator)
ETA_HAT = 1 + 1 / ETA       # utility parameter
PHIL = 1 - PHIK             # labour's importance in production
DPT = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # deterministic prod trend
RWU = (1 - PHIK) * DPT * (DPT - DELTA) ** (-1 / GAMMA) # Rel Weight in Utility
ZETA = DM([ZETA0, ZETA1])
NVAR = NPOL * NTIM * NREG * NSEC    # total number of variables
X0 = DM.ones(NVAR)          # our initial warm start 
# k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
KAP0 = DM.ones(NRxS)        # how about NSEC ????
#for j in range(n_agt):
#    KAP0[j] = np.exp(
#        np.log(kap_L) + (np.log(kap_U) - np.log(kap_L)) * j / (n_agt - 1)
#    )
#-----------suppressed derived economic parameters
#IVAR = np.arange(0,NVAR)   # index set (as np.array) for all variables

#==============================================================================
#-----------uncertainty
#------------------------------------------------------------------------------
#-----------probabity of no tip by time t as a pure function
#-----------requires: "import economic_parameters as par"
def prob_no_tip(
        tim, # time step along a path
        tpt=TPT, # transition probability of tipping
):
    return (1 - tpt) ** tim

#-----------prob no tip as a vec of parameters
PNT = DM.ones(LPTH)
for t in range(LPTH):
    PNT[t] = prob_no_tip(t)

#-----------expected shock
def E_zeta(
        t,
        zeta=ZETA,
        pnot=prob_no_tip,
):
    val = zeta[1] + pnot(t) * (zeta[0] - zeta[1])
    return val

E_ZETA = DM.ones(LFWD)
for t in range(LFWD):
    E_ZETA[t] = E_zeta(t)

#-----------if t were a variable, then, for casadi, we could do:
#t = c.MX.sym('t')
#pnt = c.Function('cpnt', [t], [prob_no_tip(t)], ['t'], ['p0'])
#PNT = pnt.map(LPTH)         # row vector 

#-----------For every look-forward, initial kapital is a parameter. 
#-----------To speed things up, we feed it in in CasADi-symbolic form:
par_kap  = MX.sym('kap', NRxS)
par_zeta = MX.sym('zeta', LFWD)

#==============================================================================
#-----------variables: these are symbolic expressions of casadi type MX or SX
#------------------------------------------------------------------------------
var_con = MX.sym('con', NRxSxT)
var_lab = MX.sym('lab', NRxSxT)
var_knx = MX.sym('knx', NRxSxT)

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
    dim = d_dim[pk]
    stride = NTIM * NREG * NSEC ** dim
    start = p * stride
    end = start + stride
    d_pol_ind_x[pk] = range(NVAR)[start : end : 1]

#-------Dict for locating every variable at a given time
d_tim_ind_x = dict()
for t in range(NTIM):
    indlist = []
    for pk in i_pol.keys():
        p = i_pol[pk]
        dim = d_dim[pk]
        stride = NREG * NSEC ** dim
        start = (p * NTIM + t) * stride
        end = start + stride
        indlist.extend(range(NVAR)[start : end : 1])
    d_tim_ind_x[t] = indlist

def tim_ind_pol(
        pol_key,
        tim_key,
        nreg=NREG,
        nsec=NSEC,
        ntim=NTIM,
        d=d_dim
):
    dim = d[pol_key]                #= 2 for 2-d variables
    lpol_t = nreg * nsec ** dim     #length of pol at time t
    val = slice(int(tim_key) * lpol_t, (int(tim_key) + 1) * lpol_t, 1)
    #d_tim_ind_pol = dict()
    #for t in range(NTIM):
    #    indlist = []
    #    d = d_dim[pol_key]
    #    stride = NREG * NSEC ** d
    #    start = t * stride
    #    end = start + stride
    #    indlist += range(NVAR)[start : end : 1]
    #    d_tim_ind_pol[t] = sorted(indlist))
    #val = d_tim_ind_pol[tim_key]
    return val
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
    d_reg_ind_x[rk] = indlist

def reg_ind_pol(
        pol_key,
        reg_key,
        nreg=NREG,
        nsec=NSEC,
        ntim=NTIM,
        nvar=NVAR,
        d=d_dim,
):
    d_reg_ind_pol = dict()
    dim = d_dim[pol_key]
    for rk in i_reg.keys():
        r = i_reg[rk]
        indlist = []
        for t in range(ntim):
            stride = nsec ** dim
            start = (t * nreg + r) * stride
            end = start + stride
            indlist += range(nvar)[start : end : 1]
        d_reg_ind_pol[rk] = indlist
    val = np.array(d_reg_ind_pol[reg_key])
    return val

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
                dim = d_dim[pk]
                stride = 1
                start = (p * NTIM * NREG + t * NREG + r) * NSEC ** dim + s
                end = start + stride
                indlist += range(NVAR)[start : end : 1]
    d_sec_ind_x[sk] = indlist

#-----------union of all the "in_x" dicts: those relating to indices of x
d_ind_x = d_pol_ind_x | d_tim_ind_x | d_reg_ind_x | d_sec_ind_x

d_ind_p = {
    'kap'       : range(NRxS),
    'start'     : range(NRxS),
    'shock'     : range(NRxS, NRxS + NTIM),
}
for t in range(LFWD):
    d_ind_p[t]  = [NRxS + t]

#==============================================================================
#-----------function for returning index subsets of x for a pair of dict keys
def sub_ind_x(
        key1,             # any key of d_ind_x
        key2,             # any key of d_ind_x
        d=d_ind_x,   # dict of index categories: pol, time, sec, reg
):

    val = np.array(list(sorted(set(d[key1]) & set(d[key2]))))
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
    val = np.array(list(sorted(set(d[key1]) & set(d[key2]))))
    return val

#==============================================================================
#---------------economic_functions
#------------------------------------------------------------------------------
#-----------instantaneous utility as a pure function
#-----------requires: "import economic_parameters as par"
def instant_utility(
        con,            # consumption vec of vars at given time
        lab,            # labour vec of vars at given time
        B=RWU,          # relative weight of con and lab in util
        rho=RHO,        # regional-weights vec at given time
        gh=GAMMA_HAT,
        eh=ETA_HAT,
):
    #-------log utility
    #val = np.sum(rho * (np.log(con) - B * np.log(lab)))
    #-------general power utility:
    val = dot(rho, con ** gh / gh - B * lab ** eh / eh)
    return val

#-----------next utility components for efficient computation of objective
#-----------first consumption:
def utility_vec(
        con,
        lab,
        B=RWU,          # relative weight of con and lab in util
        gh=GAMMA_HAT,
        eh=ETA_HAT,
):
    val = con ** gh / gh - B * lab ** eh / eh
    return val

def weights_vec(
        beta=BETA,              # discount factor 
        rho=RHO,                # regional weights vector
        lpol=NRxSxT,            # length of the policy vector
        lfwd=LFWD,              # look forward = NTIM
        nrxs=NRxS,              # 
        r_ind_p=reg_ind_pol,    #
        t_ind_pol=tim_ind_pol,  #
        i_r=i_reg
):
    beta_vec = DM.ones(lpol)
    rho_vec = DM.ones(lpol)
    for t in range(lfwd):
        beta_vec[t_ind_pol(pol_key='con', tim_key=t)] *= beta ** t
    for rk in i_r.keys():
        rho_vec[r_ind_p(pol_key='con', reg_key=rk)] = rho[i_r[rk]]
    val = beta_vec * rho_vec
    return val

WVEC = weights_vec()
#==============================================================================
#-----------v-tail as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def V_tail(
        kap,                # kapital vec of vars at time t = LFWD - 1 
        lab,                # labour vec of vars at time t = LFWD - 1
        A=DPT,              # deterministic productivity trend
        beta=BETA,          # discount factor
        nrxs=NRxS,
        phik=PHIK,          # weight of capital in production
        tcs=TCS,            # tail consumption share
        u=instant_utility,  # utility function: req. con and lab at t
):
    #-------tail consumption vec
    con_tail = tcs * A * kap ** phik
    lab_tail = lab
    #-------tail labour vec normalised to one
    val = u(con=con_tail, lab=lab_tail) / (1 - beta)
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
    val = 1e+0 * (phia / 2) * kap * pow(knx / kap - 1, 2)
    return val
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
        nreg=NREG,
        adjc=adjustment_cost, # Gamma in Cai-Judd
        E_f=E_output,
):
    sav = knx - (1 - delta) * kap
    reg_surplus = E_f(kap, lab, E_shock) - con - sav - adjc(kap, knx)
    val = dot(reg_surplus, DM.ones(nreg))
    return val

#==============================================================================
#-----------objective function (purified)
def objective(
        con=var_con,            #casadi vec of symbolic variables
        knx=var_knx,            #casadi vec of symbolic variables 
        lab=var_lab,            #casadi vec of symbolic variables 
        beta=BETA,              # discount factor
        lfwd=LFWD,              # look-forward parameter
        nrxs=NRxS,
        wvec=WVEC,              # weight vector: across time and regions
        t_ind_pol=tim_ind_pol,  # function for time indices in policy vectors
        u_vec=utility_vec,      # utility function representing flow per t
        v=V_tail,               # tail-sum value function
):
    #-------set tail kapital: extract/locate knx at the planning horizon in knx
    kap_tail = knx[t_ind_pol('knx', lfwd - 1)]
    #-------set tail labour
    lab_tail =  DM.ones(NRxS) # lab[t_ind_pol('lab', lfwd - 1)]
    # sum discounted utility over the planning horizon
    val = dot(wvec, u_vec(con, lab)) + beta ** lfwd * v(kap_tail, lab_tail)
    return val

#==============================================================================
#-----------constraints: both equality and inequality
def constraints(
        kap=par_kap,            #casadi vec of symbolic parameters 
        shk=par_zeta,           #casadi vec of symbolic parameters
        con=var_con,            #casadi vec of symbolic variables
        knx=var_knx,            #casadi vec of symbolic variables 
        lab=var_lab,            #casadi vec of symbolic variables 
        delta=DELTA,
        lfwd=LFWD,
        nrxsxt=NRxSxT,
        ind_p=sub_ind_p,
        i_r=i_reg,
        t_ind_pol=tim_ind_pol,
        mcl=market_clearing,
):
    eqns = MX.zeros(lfwd)
    ineqns = MX.zeros(nrxsxt)
    for t in range(lfwd):
        if t == 0:
            KAP = kap
        else:
            KAP = knx[t_ind_pol("knx", t - 1)]
        E_SHOCK = shk[t]
        CON = con[t_ind_pol('con', t)]
        KNX = knx[t_ind_pol('knx', t)]
        LAB = lab[t_ind_pol('lab', t)]
        eqns[t] = mcl(
                    con=CON,
                    knx=KNX,
                    lab=LAB,
                    kap=KAP,
                    E_shock=E_SHOCK,
        )
        sav = knx[t_ind_pol('knx', t)] - (1 - delta) * KAP
        ineqns[t * NRxS: (t + 1) * NRxS ] = sav
    return vertcat(eqns, ineqns)

#==============================================================================
#-----------dict of arguments for the casadi function nlpsol
nlp = {
    'x' : vertcat(var_con, var_knx, var_lab),
    'p' : vertcat(par_kap, par_zeta),
    'f' : objective(),
    'g' : constraints(),
}

#==============================================================================
#-----------options for the ipopt (the solver) and casadi (the frontend)
#------------------------------------------------------------------------------
ipopt_opts = {
    'ipopt.print_level' : 5,          #default 5
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
LBX = DM.ones(NVAR) * 1e-1
UBX = DM.ones(NVAR) * 1e+1
LBG = DM.zeros(NSxT + NRxSxT)
UBG = vertcat(DM.zeros(NSxT), DM.ones(NRxSxT) * 1e+1)
#P0 = DM.ones(NRxS + NTIM)
P0 = vertcat(KAP0, E_ZETA)

#-----------a function for removing elements from a dict
#def exclude_keys(d, keys):
#    return {x: d[x] for x in d if x not in keys}

arg = dict()
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
        arg['x0'] = res[s - 1]["x"]
        arg['p'] = vertcat(res[s - 1]['x'][sub_ind_x("knx", 0)], E_ZETA)
        arg['lam_g0'] = res[s - 1]['lam_g']

    #-----------execute solver
    res[s] = solver.call(arg)
#-*-*-*-*-*-loop ends here
#==============================================================================
#-----------print results
x_sol = dict()
for pk in d_pol_ind_x.keys():
    print("the solution for", pk, "at steps", range(len(res)), "along path 0 is\n")
    for s in range(len(res)):
        x_sol[s] = np.array(res[s]["x"])
        print(x_sol[s][sub_ind_x(pk, s)], " ")
    print(".\n")
    #print('the full dict of results for step', s, 'is\n', res[s])
#    print('the vector of variable values for step', s, 'is\n', res[s]['x'])
