#!/usr/bin/env python3
from parameters import *

# ======================================================================
# utility function
def utility(con, lab):
    #util = beta ** t * sum(tau[j] * (con[j] ** gammahat / gammahat - B * lab[j] ** etahat / etahat) for j in range(n_agt))
    util = sum(tau[j] * (np.log(con[j]) -  np.log(lab[j])) for j in range(n_agt))
    #print("util=", util)
    return util

# ======================================================================
# v-tail
def V_tail(kap):
    con = 0.75 * A * kap ** phik
    #print("con_tail =", con)
    lab = np.ones(len(kap))
    #print("V_tail =", utility(con, lab) / (1 - beta))
    return utility(con, lab) / (1 - beta)

# ======================================================================
def Pr_noTip(t):
    return (1 - p_01) ** (t)

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
