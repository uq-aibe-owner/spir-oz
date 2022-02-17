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
    objective,
)

# Constraints


def f_ctt(var, kap, t):
    # f_prod=output_f(kap, lab, itm)
    e_ctt = dict()
    # canonical market clearing constraint
    e_ctt["mclt"] = budget(kap, var[I["con"]], var[I["sav"]], var[I["lab"]], t)
    # capital next period constraint
    e_ctt["knxt"] = (1 - delta) * kap + var[I["sav"]] - var[I["knx"]]
    # intermediate sum constraints
    # e_ctt["savt"] = SAV_com - var[I["sav"]]
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
