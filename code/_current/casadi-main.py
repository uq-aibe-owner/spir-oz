import casadi as cas
from casadi import  MX, SX, DM, Function, nlpsol, vertcat, sum1, dot, \
    Sparsity, transpose, mac, external
import numpy as np

#==============================================================================
#-----------parameters
#------------------------------------------------------------------------------
#-----------economic parameters
#-----------basic economic parameters
NREG = 3        # number of regions
NSEC = 1        # number of sectors
PHZN = NTIM = LFWD = 10# look-forward parameter / planning horizon (Delta_s)
NPOL = 4        # number of policy types: con, lab, knx, #itm
NITR = LPTH = 10# path length (Tstar): number of random steps along given path
NPTH = 1        # number of paths (in basic example Tstar + 1)
BETA = 97e-2    # discount factor
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
#NCTT = LFWD # may add more eg. electricity markets are specific
GAMMA_HAT = 1 - 1 / GAMMA   # utility parameter (consumption denominator)
ETA_HAT = 1 + 1 / ETA       # utility parameter
PHIL = 1 - PHIK             # labour's importance in production
DPT = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # deterministic prod trend
RWU = (1 - PHIK) * DPT * (DPT - DELTA) ** (-1 / GAMMA) # Rel Weight in Utility
ZETA = DM([ZETA0, ZETA1])
NVAR = NPOL * NTIM * NREG * NSEC    # total number of variables
X0 = DM.ones(NVAR)          # our initial warm start 

#==============================================================================
#-----------some efficient matrices for constructing constraints, shocks, etc. 
#-----------market clearing matrix
b = np.kron(np.arange(NSxT, dtype=np.uint8), np.ones(NREG, dtype=np.uint8))
s = Sparsity(NSxT, NRxSxT, range(NRxSxT + 1), b)
MCL_MATRIX = DM(s)

#-----------shock matrix (for the case with uniform shocks across reg)
SHK_MATRIX = DM(transpose(s))

#-----------suppressed derived economic parameters
#--GAMS: k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
#IVAR = np.arange(0,NVAR)   # index set (as np.array) for all variables
#for j in range(n_agt):
#    KAP0[j] = np.exp(
#        np.log(kap_L) + (np.log(kap_U) - np.log(kap_L)) * j / (n_agt - 1)
#    )

#==============================================================================
#-----------initial kapital: a casadi parameter 
KAP0 = dict()
KAP0[0] = 3 * DM.ones(NRxS)        # initial kapital (at t=0)
#-----------extend with zeros to a vector of length NRxSxT:
sk = Sparsity(NRxSxT, NREG, range(NREG + 1), range(NREG))
KAP0_MATRIX = DM(sk)
KAP0[0] = MX(KAP0_MATRIX @ KAP0[0])
#==============================================================================
#-----------uncertainty: a casadi parameter
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
        zeta=ZETA, pnot=prob_no_tip,
):
    val = zeta[1] + pnot(t) * (zeta[0] - zeta[1])
    return val

E_ZETA = DM.ones(LFWD)
for t in range(LFWD):
    E_ZETA[t] = E_zeta(t)
E_ZETA = SHK_MATRIX @ E_ZETA
#-----------if t were a variable, then, for casadi, we could do:
#t = c.MX.sym('t')
#pnt = c.Function('cpnt', [t], [prob_no_tip(t)], ['t'], ['p0'])
#PNT = pnt.map(LPTH)         # row vector 

#-----------For every look-forward, initial kapital is a parameter. 
#-----------To speed things up, we feed it in in CasADi-symbolic form:
spar_KAP0 = Sparsity(NRxSxT, 1, [0, NRxS], range(NRxS))
par_kap  = MX.sym('kap', spar_KAP0)
par_zet = MX.sym('zet', NRxSxT)
v_par = vertcat(
            par_kap,
            par_zet,
)
l_par = [
    par_kap,
    par_zet
]
d_par = {
    'kap' : par_kap,
    'zet' : par_zet
}
#==============================================================================
#-----------variables: these are symbolic expressions of casadi type MX or SX
#------------------------------------------------------------------------------
var_con = MX.sym('con', NRxSxT)
var_knx = MX.sym('knx', NRxSxT)
var_lab = MX.sym('lab', NRxSxT)
var_sav = MX.sym('sav', NRxSxT)
v_var = vertcat(var_con, var_lab, var_knx, var_sav)
l_var = [var_con, var_knx, var_lab, var_sav]
d_var = {'con' : var_con, 'knx' : var_knx, 'lab' : var_lab, 'sav' : var_sav}

v_var_par = vertcat(v_var, v_par)
l_var_par = [var_con,
             var_knx,
             var_lab,
             var_sav,
             par_kap,
             par_zet,
]
d_var_par = d_var | d_par
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
    'sav': 1,
}
i_pol = {
    "con": 0,
    "knx": 1,
    "lab": 2,
    'sav': 3,
}
i_reg = {
    "aus": 0,
    "qld": 1,
    "wld": 2,
}
i_sec = {
    "agri": 0,          # agriculture
    "fori": 1,          # forestry
    #"ming": 2,         # mining
    #"manu": 3,         # manufacturing
    #"util": 4,         # utilities
    #"cnst": 5,          # construction
    #"coms": 6,         # commercial services
    #"tpts": 7,          # transport
    #"hhld": 8,          # residential/household
}
# Warm start
pol_S = {
    "con": 4,
    "lab": 1,
    "knx": KAP0,
    "sav": 2,
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

def s_tim(
        tim_key,
        nreg=NREG,
        nsec=NSEC,
        ntim=NTIM,
        d=d_dim,
):
    dim = 1                             #= 2 for 2-d variables
    lpol_t = nreg * nsec ** dim         #length of pol at time t
    val = slice(int(tim_key) * lpol_t, (int(tim_key) + 1) * lpol_t, 1)
    return val
def f_eval(
        vec,
        tim_key,
        nreg=NREG,
        nsec=NSEC,
        ntim=NTIM,
        d=d_dim,
):
    dim = 1                             #= 2 for 2-d variables
    lpol_t = nreg * nsec ** dim         #length of pol at time t
    val = vec[slice(int(tim_key) * lpol_t, (int(tim_key) + 1) * lpol_t, 1)]
    return val
    #d_eval = dict()
    #for t in range(NTIM):
    #    indlist = []
    #    d = d_dim[pol_key]
    #    stride = NREG * NSEC ** d
    #    start = t * stride
    #    end = start + stride
    #    indlist += range(NVAR)[start : end : 1]
    #    d_eval[t] = sorted(indlist))
    #val = d_eval[tim_key]

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
        d=d_ind_p,   # dict of index categories: kap and zet 
):
    val = np.array(list(sorted(set(d[key1]) & set(d[key2]))))
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
        nrxs=NRxS,              # number of regions times number sectors
        r_ind_p=reg_ind_pol,    # function
        sl=s_tim,               # slice for time indices
        i_r=i_reg               # dict for regional indices
):
    beta_vec = np.ones(lpol)
    rho_vec = np.ones(lpol)
    for t in range(lfwd):
        beta_vec[sl(t)] *= beta ** t
    for rk in i_r.keys():
        rho_vec[r_ind_p('con', rk)] = rho[i_r[rk]]
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
        lab,                    # lab vector of vars at time t
        kap,                    # kap vector of vars at time t
        zet,                    # shock or expected shock at time t
        A=DPT,                  # determistic prod trend
        phik=PHIK,              # weight of kap in prod
        phil=PHIL,              # weight of lab in prod
):
    #print(lab, kap, zet)
    y = A * (kap ** phik) * (lab ** phil)   # output
    val = zet * y
    return val

#==============================================================================
#-----------adjustment costs of investment as a pure function
#-----------requires: "import economic_parameters as par"
def adjustment_cost(
        knx,
        kap,
        phia=PHIA, # adjustment cost multiplier
):
    # since sav/kap - delta = (knx - (1 - delta) * kap)/kap - delta = ..
    # we can therefore rewrite the adjustment cost as
    #print(knx, kap)
    val = (phia / 2) * kap * pow(knx / kap - 1, 2)
    return val

#==============================================================================
#-----------market clearing/budget constraint as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def market_clearing(
        con=var_con,
        knx=var_knx,
        lab=var_lab,
        sav=var_sav,
        kap=par_kap,
        zet=par_zet,
        adj_cost=adjustment_cost,# Gamma in Cai-Judd
        E_f=E_output,
):
    #sav = knx - (1 - delta) * kap
    out = E_f(kap=kap, lab=lab, zet=zet)
    adj = adj_cost(knx=knx, kap=kap)
    val = out - (con + sav + adj)
    return val

#==============================================================================
#-----------dynamic equations
def dynamics(
        knx,
        sav,
        kap,
        delta=DELTA,
):
    val = knx - sav - (1 - delta) * kap
    return val

#==============================================================================
#-----------objective function (purified)
def objective(
        con=var_con,                    #casadi vec of symbolic variables
        knx=var_knx,                    #casadi vec of symbolic variables 
        lab=var_lab,                    #casadi vec of symbolic variables 
        beta=BETA,              # discount factor
        lfwd=LFWD,              # look-forward parameter
        nrxs=NRxS,
        wvec=WVEC,              # weight vector: across time and regions
        ev=f_eval,              # evaluate policy vectors at  time t
        u_vec=utility_vec,      # utility function representing flow per t
        v=V_tail,               # tail-sum value function
):
    #-------set tail kapital: extract/locate knx at the planning horizon in knx
    kap_tail = ev(knx, lfwd - 1)
    #-------set tail labour
    lab_tail = DM.ones(NRxS)    # lab[ev('lab', lfwd - 1)] 
    # sum discounted utility over the planning horizon
    val = dot(wvec, u_vec(con, lab)) + beta ** lfwd * v(kap_tail, lab_tail)
    return val #/ 1.0869755e+11

#==============================================================================
#-----------casadi function equivalents of objective():
cas_obj = Function(
        'cas_obj',
        [var_con, var_knx, var_lab],
        [objective(var_con, var_knx, var_lab)],
        ['con', 'knx', 'lab'],
        ['obj'],
)
cas_obj_ext = external('cas_obj', './cas_obj.so')
#==============================================================================
#-----------constraints: both equality and inequality
def constraints(
        con=var_con,            #casadi vec of symbolic variables
        knx=var_knx,            #casadi vec of symbolic var 
        lab=var_lab,            #casadi vec of symbolic var 
        sav=var_sav,            #casadi vec of symbolic var
        kap=par_kap,            #casadi vec of symbolic parameters 
        zet=par_zet,           #casadi vec of symbolic par
        nreg=NREG,
        A=MCL_MATRIX,            # market clearing matrix (pooled across reg)
        mcl=market_clearing,
        dyn=dynamics
):
    #-------generate the vector of current capital for each t:
    full_kap = vertcat(kap[: nreg], knx[: -nreg])  #time shifted knx = kap
    mcl_eqns = mcl(
                con=con,
                knx=knx,
                lab=lab,
                kap=full_kap,
                sav=sav,
                zet=zet,
    )
    mcl_eqns = A @ mcl_eqns
    print(mcl_eqns)
    dyn_eqns = dyn(knx=knx, sav=sav, kap=full_kap)
    print(dyn_eqns)
    return vertcat(mcl_eqns,  dyn_eqns)
ctt = constraints()

#==============================================================================
#-----------casadi function equivalents of contraints()
cas_ctt = Function(
        'cas_ctt',
        l_var_par,
        [constraints(**d_var_par)],
        [key for key in d_var_par.keys()],
        ['ctt'],
)
#==============================================================================
#-----------raw functions:
#------------------------------------------------------------------------------
#-----------current kapital
raw_kap = np.ones(NRxSxT) #vertcat(par_kap[:NRxS], var_knx[:-NRxS])
#-----------tail consumption and tail labour for the value function
raw_tl_con = TCS * DPT * var_knx[(LFWD - 1) * NREG : LFWD * NREG] ** PHIK
raw_tl_lab = np.ones(NRxS)#var_lab[(LFWD - 1) * NREG : LFWD * NREG]

raw_obj = (dot(WVEC, utility_vec(var_con, var_lab)) \
        + BETA ** LFWD * instant_utility(raw_tl_con, raw_tl_lab)) / (1 - BETA)

#raw_obj *= 0

print('raw_obj', raw_obj)
adj = 0 # MX(.25 * raw_kap * pow(var_knx / raw_kap - 1, 2))
out = E_ZETA * DPT * pow(raw_kap, .33) * pow(var_lab, .66)
raw_mcl = mac(
    MCL_MATRIX,
    E_output(kap=raw_kap, lab=var_lab, zet=E_ZETA) - var_con - var_sav \
        - adj, np.ones(10)
)
print(raw_mcl)
#-----------for checking:
#raw_mcl = market_clearing()

raw_dyn = var_knx - (var_sav + (1 - DELTA) * raw_kap)

#==============================================================================
#-----------dict of arguments for the casadi function nlpsol
nlp = {
    'x' : v_var,
    'p' : v_par,
    'f' : objective(),
    'g' : constraints(),
    #'f' : cas_obj(var_con, var_knx, var_lab),
    #-------the following two are seemingly identical:
    #'f' : raw_obj,
    #'g' : vertcat(raw_mcl, raw_dyn)
    #'g' : vertcat(raw_mcl, dynamics(knx=var_knx, sav=var_sav, kap=raw_kap)),
    #-------the following are seemingly identical:
    #'g' : cas_ctt(var_con,
    #              var_knx,
    #              var_lab,
    #              var_sav,
    #              par_kap,
    #              par_zet
    #              ),
}

#==============================================================================
#-----------options for the ipopt (the solver) and casadi (the frontend)
#------------------------------------------------------------------------------
ipopt_opts = {
    'ipopt.print_level' : 1,          #default 5
    'ipopt.linear_solver' : 'mumps', #default=Mumps
    'ipopt.obj_scaling_factor' : -1.0, #default=1.0
    #'ipopt.warm_start_init_point' : 'yes', #default=no
    #'ipopt.warm_start_bound_push' : 1e-9,
    #'ipopt.warm_start_bound_frac' : 1e-9,
    #'ipopt.warm_start_slack_bound_push' : 1e-9,
    #'ipopt.warm_start_slack_bound_frac' : 1e-9,
    #'ipopt.warm_start_mult_bound_push' : 1e-9,
    #'ipopt.fixed_variable_treatment' : 'relax_bounds', #default=
    #'ipopt.print_info_string' : 'yes', #default=no
    #'ipopt.accept_every_trial_step' : 'no', #default=no
    #'ipopt.alpha_for_y' : 'primal', #default=primal, try 'full'?
}
casadi_opts = {
    #'calc_lam_p' : False,
}
opts = casadi_opts # | ipopt_opts  
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
#solver = nlpsol('solver', 'bonmin', nlp, opts)
solver = nlpsol('solver', 'ipopt', nlp, opts)

#==============================================================================
LBX = DM.ones(NVAR) * 1e-6
UBX = DM.ones(NVAR) * 1e+1
LBG = np.zeros(NSxT + NRxSxT)
UBG = np.zeros(NSxT + NRxSxT) #vertcat(DM.zeros(NSxT), DM.ones(NRxSxT) * 1e+1)
#P0 = DM.ones(NRxS + NTIM)

#-----------a function for removing elements from a dict
#def exclude_keys(d, keys):
#    return {x: d[x] for x in d if x not in keys}

#-*-*-*-*-*-loop/iteration along path starts here 
#------------------------------------------------------------------------------
res = dict()
arg = dict()
for s in range(LPTH):
    arg[s] = dict()
    arg[s] = {
        'lbx' : LBX,
        'ubx' : UBX,
        'lbg' : LBG,
        'ubg' : UBG,
    }
    #-------set initial capital and vector of shocks for each plan
    if s == 0:
        arg[s]['x0'] = X0
        P0 = vertcat(
                KAP0[s],
                E_ZETA
        )
        #arg[s]['p'] = P0
    else:
        arg[s]['x0'] = res[s - 1]['x']
        KAP0[s] = KAP0_MATRIX @ res[s - 1]['x'][sub_ind_x('knx', 0)]
        P = vertcat(KAP0[s], E_ZETA)
        #arg[s]['p'] = P
        arg[s]['lam_g0'] = res[s - 1]['lam_g']
    print('Initial kapital at the next step', s, 'is', KAP0[s][range(NRxS)])
    #-----------execute solver
    res[s] = solver.call(arg[s])
#-*-*-*-*-*-loop ends here
#==============================================================================
#-----------print results
x_sol = dict()
g_sol = dict()
polt_sol = dict()
pol_sol = dict()
for s in range(len(res)):
    x_sol[s] = res[s]['x']
    g_sol[s] = res[s]['g']
    print('max of absolute values of constraints', max(abs(np.array(g_sol[s]))))
    polt_sol[s] = dict()
for pk in d_pol_ind_x.keys():
    pol_sol[pk] = dict()
    print("the solution for", pk, "at steps", range(len(res)), "along path 0 is\n")
    for s in range(len(res)):
        polt_sol[s][pk] = x_sol[s][sub_ind_x(pk, s)]
        print(polt_sol[s][pk])
        pol_sol[pk][s] = polt_sol[s][pk]
    print(".\n")

#print('the full dict of results for step', s, 'is\n', res[s])
#    print('the vector of variable values for step', s, 'is\n', res[s]['x'])

con_sol = np.concatenate([np.concatenate(np.array(pol_sol['con'][i])) \
                          for i in pol_sol['con'].keys()])
knx_sol = np.concatenate([np.concatenate(np.array(pol_sol['knx'][i])) \
                          for i in pol_sol['knx'].keys()])
lab_sol = np.concatenate([np.concatenate(np.array(pol_sol['lab'][i])) \
                          for i in pol_sol['lab'].keys()])
sav_sol = np.concatenate([np.concatenate(np.array(pol_sol['sav'][i])) \
                          for i in pol_sol['sav'].keys()])
kap_sol = np.concatenate([[3, 3, 3], knx_sol[:-NREG]])
out_sol = E_output(lab=lab_sol, kap=kap_sol, zet=E_ZETA)
out_sol_sec = MCL_MATRIX @ out_sol
mcl_sol = MCL_MATRIX @ (out_sol - con_sol - sav_sol \
                             - adjustment_cost(knx=knx_sol, kap=kap_sol))
dyn_sol = np.reshape(dynamics(knx=knx_sol, sav=sav_sol, kap=kap_sol), (10, 3)) 
print('out_sol_sec:\n', np.transpose(out_sol_sec))

