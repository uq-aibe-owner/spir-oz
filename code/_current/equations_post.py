# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

from cmath import sqrt
import numpy as np
from parameters import *
from variables import * 
from equations import * 
del globals()[output]
#utility, V_tail, Pr_noTip, output, Gamma_adjust, budget, f_ctt

# ======================================================================
# output_f
def output(k, l, t):
    return zeta2 * A * (k[t,:]**phik) * (l[t,:]**(1-phik))
# ======================================================================

