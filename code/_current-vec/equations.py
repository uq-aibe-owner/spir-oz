# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

from cmath import sqrt
import numpy as np
from parameters import *
from variables import * 

# ======================================================================
# utility function u(c,l)
def utility(cons, lab, t):
    sum_util=0.0
    for i in range(n_agt):
        nom1=(cons[t,i])**gammahat -1.0 
        den1=gammahat
        
        nom2= B * lab[t,i]**etahat -1.0
        den2= etahat
        
        sum_util+=tau[i]*(nom1/den1 - nom2/den2)
    
    return sum_util

# ======================================================================
#v-tail
def V_tail(k, s):
    return sum(tau[j] * beta**Delta * (0.75 * A * k[s + Delta, j]**phik)**gammahat /gammahat - B / (1-beta) for j in range(n_agt))

# ======================================================================
def Pr_noTip(t):
    return (1 - p_01)**(t)

# ======================================================================
# output
def output(k, l, t):
    return (zeta2 + Pr_noTip(t) * (zeta1-zeta2)) * A * (k[t,:]**phik) * (l[t,:]**(1 - phik))

# ======================================================================
# adjustment costs for investment 
def Gamma_adjust(kap, inv, t):
    return 0.5 * phi * kap[t,:] * sqrt(inv[t,:] / kap[t,:] - delta)

# ======================================================================
# budget constraint
def budget(kap, c, Inv, l, t):
    return sum(c[t,:] + Inv[t,:] + Gamma_adjust(kap, Inv, t) - output(kap, l, t))
    # need to work with striding and slices to negate need to convert c to matrix
# ======================================================================
# Constraints

def f_ctt(X, kap, t): 
    # f_prod=output_f(kap, lab, itm)
    e_ctt = dict()
    # canonical market clearing constraint
    e_ctt["mclt"] = budget(kap, X[I["con"]], X[I["sav"]], X[I["lab"]], t)
    # capital next period constraint
    e_ctt["knxt"] = (1 - delta) * kap + X[I["sav"]] - X[I["knx"]] 
    # intermediate sum constraints
    #e_ctt["savt"] = SAV_com - X[I["sav"]]
    # e_ctt["itmt"] = ITM_com - X[I["itm"]] ## for Cai-Judd rep
    # output constraint
    e_ctt["outt"] = X[I["out"]] - output(kap, X[I["lab"]], t)
    # utility constraint
    e_ctt["utlt"] = X[I["utl"]] - utility(X[I["con"]], X[I["lab"]], t)
    # e_ctt["blah blah blah"] = constraint rearranged into form that can be equated to zero

    # Check dicts are all same length
    if not len(d_ctt) == len(ctt_U) == len(ctt_L) == len(e_ctt):
        raise ValueError(
            "Constraint-related Dicts are not all the same length, check f_cct in parameters.py"
        )

    return e_ctt

        # Summing the 2d policy variables
    # SAV_com = np.ones(n_agt, float)
    # SAV_add = np.zeros(n_agt, float)
    # ITM_com = np.ones(n_agt, float)
    # ITM_add = np.zeros(n_agt, float)
    # for iter in range(n_agt):
    #     for ring in range(n_agt):
    #         #  SAV_com[iter] *= X[I["SAV"]][iter+n_agt*ring]**xi[ring] ## for Cai-Judd rep
    #         #  ITM_com[iter] *= X[I["ITM"]][iter+n_agt*ring]**mu[ring] ## for Cai-Judd rep
    #         #  SAV_add[iter] += X[I["SAV"]][iter*n_agt+ring] ## for Cai-Judd rep
    #         #SAV_add[iter] += X[I["SAV"]][iter * n_agt + ring]  ### to add Gamma?
    #     #  ITM_add[iter] += X[I["ITM"]][iter*n_agt+ring] ## for Cai-Judd rep