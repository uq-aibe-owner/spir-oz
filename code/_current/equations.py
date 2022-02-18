# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

from cmath import sqrt
import numpy as np
from parameters import *
from variables import *
from fcn_economic import (
    Gamma_adjust,
    Pr_noTip,
    V_tail,
    output,
    utility,
    budget,
    # objective,
)
# Constraints
# var is a single time period's variables (of length n_pol)
def f_ctt(X, initial_kap=k_init):
    # f_prod=output_f(kap, lab, itm)
    e_ctt = dict()
    # canonical market clearing constraint
    # capital next period constraint
    for t in range(Delta):
        if t == 0:
            e_ctt["kapt"][t] = X[I["kap"]][t] - initial_kap
        else:
            e_ctt["kapt"][t] = X[I["kap"]][t] - X[I["knx"]][t - 1]
    e_ctt["mclt"] = budget(X[I["kap"]], X[I["con"]], X[I["sav"]], X[I["lab"]], t)
    #for t in range(Delta):
    e_ctt["knxt"] = (1 - delta) * X[I["kap"]] + X[I["sav"]] -  X[I["knx"]]
    #e_ctt["outt"] = X[I["out"]] - output(kap, X[I["lab"]], t)
    # intermediate sum constraints
    # e_ctt["savt"] = SAV_com - var[I["sav"]]
    # e_ctt["itmt"] = ITM_com - var[I["itm"]] ## for Cai-Judd rep
    # output constraint
    # utility constraint
    #e_ctt["utlt"] = var[I["utl"]] - utility(var[I["con"]], var[I["lab"]], t)
    # e_ctt["blah blah blah"] = constraint rearranged into form that can be equated to zero

    # Check dicts are all same length
    if not len(d_ctt) == len(ctt_U) == len(ctt_L) == len(e_ctt):
        raise ValueError(
            "Constraint-related Dicts are not all the same length, check f_cct in equations.py"
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
