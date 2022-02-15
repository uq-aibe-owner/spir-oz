# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

from cmath import sqrt
import numpy as np
from parameters import *
from variables import * 
from equations import *

# ======================================================================
# output_f
def output(zeta, k, l):
    return zeta2*A*(k[t,:]**alpha) * (l[t,:]**(1-alpha))
# ======================================================================

