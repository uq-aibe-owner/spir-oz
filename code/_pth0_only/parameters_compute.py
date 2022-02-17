#!/usr/bin/env python3
import numpy as np
from parameters import *
from fcn_economic import *

# ================================================================
# Computational parameters
# Ranges for state variables
kap_L = 0.1
kap_U = 10
# Ranges for policy variables
lab_L = 0.1
lab_U = 2

vec_kap_L = kap_L * ones(n_agt)
vec_kap_U = kap_U * ones(n_agt)
vec_lab_L = lab_L * ones(n_agt)
vec_lab_U = lab_U * ones(n_agt)

out_L = output(vec_kap_L, vec_lab_L, 0)[0]
out_U = output(vec_kap_U, vec_lab_U, 0)[0]
con_L = 0.1 * output(vec_kap_L, vec_lab_L, 0)[0]
con_U = 0.99 * output(vec_kap_U, vec_lab_U, 0)[0]
utl_L = utility(vec_con_L, vec_lab_U, 0)
utl_U = utility(vec_con_U, vec_lab_L, 0)

# warm start for the policy variables

###

# k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
k_init = np.ones(n_agt)
for j in range(n_agt):
    k_init[j] = np.exp(
        np.log(kap_L) + (np.log(kap_U) - np.log(kap_L)) * j / (n_agt - 1)
    )
# ======================================================================
# constraint upper and lower bounds
ctt_L = -1e-6
ctt_U = 1e-6
