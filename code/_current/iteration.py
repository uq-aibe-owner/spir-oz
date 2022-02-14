# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from variables import *
from equations import *
import solver

# import cPickle as pickle
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from numpy.random import PCG64
from datetime import datetime
import time 
# ======================================================================


#def sceq(i_pth, rng, save_data=True):
def sceq(i_pth, save_data=True):
    # for i_pth=0,  
    # iterate over periods of interest (the last extra period is for error checking only)
    #loop(tt$(ord(tt)<=Tstar+1),
        for tt in range(0,Tstar+1):
            s = tt
            
            for t in range(s, Delta_s + s):
                def Pr_noTip(t):
                    return (1 - p_01)**(t-s) ###
            #now solve for that tt
            solver.ipopt_interface(...) ###
            pickle for each tt ###
        
        """
        print(res['ITM'])
        SAV_add = np.zeros(n_agt, float)
        ITM_add = np.zeros(n_agt, float)
        for iter in range(n_agt):
            SAV_add[iter] = np.add(res["SAV"][iter*n_agt], res["SAV"][iter*n_agt+1])
            ITM_add[iter] = res["ITM"][iter*n_agt] + res["ITM"][iter*n_agt+1]
        print(ITM_add)
        res['kap'] = Xtraining[iI]
        res['itr'] = i_pth
        y[iI] = res['obj']
        ctt = res['ctt']
        msg = "Constraint values: " + str(ctt) + os.linesep
        msg += "a quick check using output_f - con - SAV_add - ITM_add" + os.linesep
        msg += (
            str(output_f(Xtraining[iI], res["itm"]) - np.add(res['con'], SAV_add, ITM_add)) + os.linesep
        )
        print(ITM_add)
        msg += (
            "and consumption, labor, investment and intermediate inputs are, respectively,"
            + os.linesep
            + str(res['con'])
            #+ os.linesep
            #+ str(res['lab'])
            + os.linesep
            + str(res['SAV'])
            + str(res['sav'])
            + str(SAV_add)
            + os.linesep
            + str(res['ITM'])
            + str(res['itm'])
            + str(ITM_add)
        )
        if economic_verbose:
            print("{}".format(msg))
        if i_pth == numits - 1:
            ctnr.append(res)
        """
            end_nlp = time.time()
   
            output_file = filename + str(i_pth) + ".pcl"
            print(output_file)
            with open(output_file, "wb") as fd:
                pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
                print("data of step ", i_pth, "  written to disk")
                print(" -------------------------------------------")
            fd.close()

    return #ctnr


# ======================================================================
