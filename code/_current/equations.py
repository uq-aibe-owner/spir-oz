# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

from cmath import sqrt
import numpy as np
from parameters import *
from variables import * 

# ======================================================================
# utility function u(c,l)
def utility(con, lab, t):
    return beta**t * sum(tau[j] * (con[t, j])**gammahat / gammahat 
                            - B * lab[t, j]**etahat / etahat 
                            for j in range(n_agt)
                        )

# ======================================================================
#v-tail
def V_tail(kap):
    con = 0.75 * A * kap**phik
    lab = np.ones(len(kap))
    t = Delta
    return utility(con, lab, t)/ (1 - beta)

# ======================================================================
def Pr_noTip(t):
    return (1 - p_01)**(t)

# ======================================================================
# output
def output(k, l, t):
    return (zeta2 + Pr_noTip(t) * (zeta1 - zeta2)) * A * (k[t,:]**phik) * (l[t,:]**(1 - phik))

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

def f_ctt(var, kap, t): 
    # f_prod=output_f(kap, lab, itm)
    e_ctt = dict()
    # canonical market clearing constraint
    e_ctt["mclt"] = budget(kap, var[I["con"]], var[I["sav"]], var[I["lab"]], t)
    # capital next period constraint
    e_ctt["knxt"] = (1 - delta) * kap + var[I["sav"]] - var[I["knx"]] 
    # intermediate sum constraints
    #e_ctt["savt"] = SAV_com - var[I["sav"]]
    # e_ctt["itmt"] = ITM_com - var[I["itm"]] ## for Cai-Judd rep
    # output constraint
    e_ctt["outt"] = var[I["out"]] - output(kap, var[I["lab"]], t)
    # utility constraint
    e_ctt["utlt"] = var[I["utl"]] - utility(var[I["con"]], var[I["lab"]], t)
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
    #         #  SAV_com[iter] *= var[I["SAV"]][iter+n_agt*ring]**xi[ring] ## for Cai-Judd rep
    #         #  ITM_com[iter] *= var[I["ITM"]][iter+n_agt*ring]**mu[ring] ## for Cai-Judd rep
    #         #  SAV_add[iter] += var[I["SAV"]][iter*n_agt+ring] ## for Cai-Judd rep
    #         #SAV_add[iter] += var[I["SAV"]][iter * n_agt + ring]  ### to add Gamma?
    #     #  ITM_add[iter] += var[I["ITM"]][iter*n_agt+ring] ## for Cai-Judd rep