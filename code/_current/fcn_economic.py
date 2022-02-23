#!/usr/bin/env python3

# ======================================================================
#-----------utility as a pure function
#----requirements: "import parameters as par"

def utility(con, # consumption vector variable
            lab, # labour vector variable
            rho=par.RHO, # vector of regional weights
            nreg=par.NREG, # number of regions
            ):
    #-------log utility
    util = np.sum(rho * (np.log(con) -  np.log(lab)))
    #-------general power utility:
    #util = BETA ** t * np.sum(RHO[j] * (con[j] ** GAMMAhat / GAMMAhat \
    #        - B * lab[j] ** ETAhat / ETAhat) for j in range(NREG))
    return util

# ======================================================================
#-----------v-tail as a pure function
#----required: "import parameters as par"
#----required: "import economic_functions as efcn"
def V_tail(kap, # kapital vector variable
           tcs=par.TCS # tail consumption share
           A=par.DPT, # deterministic productivity trend
           phik=par.PHIK, # weight of capital in production # alpha in CJ
           beta=par.BETA, # discount factor
           u=efcn.utility, # utility function
           ):
    #-------tail consumption vector (as kap is a vector)
    tail_con = tcs * A * kap ** phik
    #-------tail labour vector normalised to one
    tail_lab = np.ones(len(kap))
    return u(tail_con, tail_lab) / (1 - beta)

# ======================================================================
def Pr_noTip(t, # time step along a path
             tpt=TPT, # transition probability of tipping
             ):
    return (1 - tpt) ** (t)

# ======================================================================
# output
def output(kap, lab, t):
    return (zeta2 + Pr_noTip(t) * (zeta1 - zeta2)) * A * (kap ** phik) * (lab ** phil)

# ======================================================================
# adjustment costs for investment
def Gamma_adjust(kap, sav, t):
    return 0.5 * phi * kap * np.square(sav / kap - delta)

# ======================================================================
# budget constraint
def budget(kap, con, sav, lab, t):
    return sum(con + sav + Gamma_adjust(kap, sav, t) - output(kap, lab, t))
    # need to work with striding and slices to negate need to convert c to matrix

# ======================================================================
# objective
#def objective(kap, )
#    sum(utility(con[t], lab[t], t) for t in range(Delta)) + V_tail(kap[Delta])
# ======================================================================
