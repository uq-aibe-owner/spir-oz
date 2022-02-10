#======================================================================
# 
#     sets the economic functions for the "Growth Model", i.e., 
#     the production function, the utility function
#     
#
#     Simon Scheidegger, 11/16 ; 07/17
#====================================================================== 

from parameters import *
import numpy as np


#====================================================================== 
#utility function u(c,l) 

def utility(cons=[], lab=[]):
    sum_util=0.0
    n=len(cons)
    for i in range(n):
        nom1=(cons[i]/big_A)**(1.0-gamma) -1.0 #CJ swap A for B (multiplying labour)
        den1=1.0-gamma
        
        nom2=(1.0-psi)*((lab[i]**(1.0+eta)) -1.0)
        den2=1.0+eta
        
        sum_util+=(nom1/den1 - nom2/den2)
    
    util=sum_util
    
    return util 


#====================================================================== 
# output_f 

def output_f(kap=[], lab=[]):
    fun_val = big_A*(kap**psi)*(lab**(1.0 - psi)) ###CJ A = big_A
    ### put in mean zeta as errors as actual zetas only enter g_t
    return fun_val

#======================================================================

# transformation to comp domain -- range of [k_bar, k_up]

def box_to_cube(knext=[]):
    # transformation onto cube [0,1]^d      
    knext_box = np.clip(knext, k_bar, k_up)

    return knext_box

# adjustment costs for investment 
def Gamma_adjust(kap=[], inv=[]) 
    fun_val = 0.5*phi*kap*((inv/kap - delta)**2.0) #CJ change zeta to phi
    return fun_val
#======================================================================  