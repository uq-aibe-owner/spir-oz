# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from parameters_compute import *

# ======================================================================
# dimensions of each policy variable: 0 for a scalar; 1 for a vector; 2 for a matrix
d_pol = {
    "con": 1,
    "lab": 1,
    "knx": 1,
    "sav": 1,
    #"out": 1,
    #    "itm": 1,
    #    "ITM": 2,
    #    "SAV": 2,
    #"utl": 0
    #    "val": 0,
}

# dimensions of the dual variables (i.e. number of constraints)
d_ctt = {
    "mclt": 0,
    "knxt": 1,
    #"outt": 1,
    #         "savt": 1,
    #        "itmt": 1,
    #"utlt": 0,
    #         "valt": 0,
}
# ======================================================================m

# Lower policy variables bounds
pol_L = {
    "con": con_L,
    "lab": lab_L,
    "knx": kap_L,
    "sav": sav_L,
    #    "ITM": pL,
    #    "SAV": pL,
    #    "itm": pL,
    #"out": out_L,
    #"utl": utl_L,
    #    "val": -pU,
}
# Upper policy variables bounds
pol_U = {
    "con": con_U,
    "lab": lab_U,
    "knx": kap_U,
    "sav": sav_U,
    #    "ITM": pU,
    #    "SAV": pU,
    #    "itm": pU,
    #"out": out_U,
    #"utl": utl_U,
    #    "val": pU,
}
# Warm start
pol_S = {
    "con": 4,
    "lab": 1,
    "knx": k_init,
    "sav": 2,
    #    "itm": 10,
    #    "ITM": 10,
    #    "SAV": 10,
    #"out": 6,
    #"utl": 1,
    #    "val": -300,
}

if not len(d_pol) == len(pol_U) == len(pol_L) == len(pol_S):
    raise ValueError(
        "Policy-variable-related Dicts are not all the same length, check variables.py"
    )

# Constraint variables bounds
ctt_L = {
    "mclt": c_L,
    "knxt": c_L,
    #    "savt": ctt_L,
    #    "itmt": ctt_L,
    #"utlt": ctt_L,
    #"outt": c_L,
    #    "valt": ctt_L,
}
ctt_U = {
    "mclt": c_U,
    "knxt": c_U,
    #    "savt": ctt_U,
    #    "itmt": ctt_U,
    #"utlt": ctt_U,
    #"outt": c_U,
    #    "valt": ctt_U,
}

# Check dicts are all same length
if not len(d_ctt) == len(ctt_U) == len(ctt_L):
    raise ValueError(
        "Constraint-related dicts are not all the same length, check parameters.py and variables.py"
    )

# =================================================================# =========================
# Slicing the ipopt vector for the policy variables
# =========================
# creating list of the dict keys
pol_key = list(d_pol.keys())
# instantiate the number of policy variables at a given time
n_pol = 0
# temporary variable to keep track of previous highest index
prv_ind = 0
# dict for indices of each policy variable in X/x
I = dict()
for iter in pol_key:
    n_pol += n_agt ** d_pol[iter]
    # allocating slices of indices to each policy variable as a key
    I[iter] = slice(prv_ind, prv_ind + n_agt ** d_pol[iter])
    prv_ind += n_agt ** d_pol[iter]

# total number of variables across time
n_pol_all = Delta * n_pol
# =========================
# Slicing the ipopt vector for the policy variables
# =========================
# for use in running through loops
ctt_key = list(d_ctt.keys())
# number of constraints at a given point in time
n_ctt = 0
# dict for indices of each constraint variable in G/g
I_ctt = dict()
# temporary variable to keep track of previous highest index
prv_ind = 0
for iter in ctt_key:
    # add to number of total constraint values
    n_ctt += n_agt ** d_ctt[iter]
    # allocating slicess of indices to each constraint variable as a key
    I_ctt[iter] = slice(prv_ind, prv_ind + n_agt ** d_ctt[iter])
    prv_ind += n_agt ** d_ctt[iter]

# total number of constraints across time
n_ctt_all = Delta * n_ctt
