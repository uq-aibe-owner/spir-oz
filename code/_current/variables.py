# ======================================================================
#
#     sets the parameters and economic functions for the model
#     "Growth Model"
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

import numpy as np
from parameters import * 
# ======================================================================
# dimensions of each policy variable: 0 for a scalar; 1 for a vector; 2 for a matrix
d_pol = {
    "con": 1,
    #    "lab": 1,
    "sav": 1,
    "knx": 1,
    "ITM": 2,
    "SAV": 2,
    "itm": 1,
    "val": 0,
    "utl": 0,
    "out": 1,
}

# dimensions of each constraint variable
d_ctt = {"mclt": 1, "knxt": 1, "savt": 1, "itmt": 1, "valt": 0, "utlt": 0, "outt": 1}
# ======================================================================

# Ranges for Controls
pL = 1e-1
pU = 1e3
# Lower policy variables bounds
pol_L = {
    "con": 1.1,
    #    "lab": pL,
    "sav": pL,
    "knx": pL,
    "ITM": pL,
    "SAV": pL,
    "itm": pL,
    "val": -pU,
    "utl": pL,
    "out": pL,
}
# Upper policy variables bounds
pol_U = {
    "con": pU,
    #    "lab": pU,
    "sav": pU,
    "knx": pU,
    "ITM": pU,
    "SAV": pU,
    "itm": pU,
    "val": pU,
    "utl": pU,
    "out": pU,
}
# Warm start
pol_S = {
    "con": 10,
    #    "lab": 10,
    "sav": 10,
    "knx": 10,
    "ITM": 10,
    "SAV": 10,
    "itm": 10,
    "val": -300,
    "utl": 10,
    "out": 10,
}

if not len(d_pol) == len(pol_U) == len(pol_L) == len(pol_S):
    raise ValueError(
        "Policy-variable-related Dicts are not all the same length, check parameters.py"
    )

# Constraint variables bounds
cL = 0 * -1e-5
cU = 0 * 1e-5
ctt_L = {
    "mclt": cL,
    "knxt": cL,
    "savt": cL,
    "itmt": cL,
    "valt": cL,
    "utlt": cL,
    "outt": cL,
}
ctt_U = {
    "mclt": cU,
    "knxt": cU,
    "savt": cU,
    "itmt": cU,
    "valt": cU,
    "utlt": cU,
    "outt": cU,
}

# Check dicts are all same length
if not len(d_ctt) == len(ctt_U) == len(ctt_L):
    raise ValueError(
        "Constraint-related Dicts are not all the same length, check parameters.py"
    )

# ======================================================================
# Automated stuff, for indexing etc, shouldnt need to be altered if we are just altering economics

# creating list of the dict keys
pol_key = list(d_pol.keys())
# number of policy variables in total, to be used for lengths of X/x vectors
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

# for use in running through loops
ctt_key = list(d_ctt.keys())
# number of constraints
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
