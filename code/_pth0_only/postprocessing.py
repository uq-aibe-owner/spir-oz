# ======================================================================
#     Created by Josh Aberdeen, Cameron Gordon, Patrick O'Callaghan 11/2021
# ======================================================================

import numpy as np
from parameters import *
from variables import *
from equations import *
#import cPickle as pickle
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import solver as solver
import os
#======================================================================    
# Routine compute the errors
def ls_error(n_agents, t1, t2, num_points):
    i=0
    while os.path.exists("errors%s.txt" % i):
      i += 1
    file=open('errors%s.txt' % i, 'w')
    
    np.random.seed(0)
    dim = n_agents
    k_test = np.random.uniform(kap_L, kap_U, (num_points, dim))
    # test target container
    y_test = np.zeros(num_points, float)
    to_print=np.empty((1,5))

    if (t1 == 0):
        t1+=1
    for i in range(t1-1, t2-1):
        sum_diffs=0
        diff = 0
    
        # Load the model from the previous i_pth step
        restart_data = filename + str(i) + ".pcl"
        with open(restart_data, 'rb') as fd_old:
            gp_old = pickle.load(fd_old)
            print("data from i_pth step ", i , "loaded from disk")
        fd_old.close()      

        # Load the model from the previous i_pth step
        restart_data = filename + str(i+1) + ".pcl"
        with open(restart_data, 'rb') as fd:
            gp = pickle.load(fd)
            print("data from i_pth step ", i+1 , "loaded from disk")
        fd.close()
      
        mean_old, sigma_old = gp_old.predict(k_test, return_std=True)
        mean, sigma = gp.predict(k_test, return_std=True)

        gp_old = gp
        # solve bellman equations at test points
        for j in range(len(k_test)):
            y_test[j] = solver.ipopt_interface(k_test[j], n_agents)["obj"]

        targ_new = y_test
        # plot predictive mean and 95% quantiles
        #for j in range(num_points):
            #print k_test[j], " ",y_pred_new[j], " ",y_pred_new[j] + 1.96*sigma_new[j]," ",y_pred_new[j] - 1.96*sigma_new[j]

        diff_mean = mean_old - mean
        max_diff_mean = np.amax(np.fabs(diff_mean))
        avg_diff_mean = np.average(np.fabs(diff_mean))

        diff_targ = mean - targ_new
        max_diff_targ = np.amax(np.fabs(diff_targ))
        avg_diff_targ = np.average(np.fabs(diff_targ))

        to_print[0,0]= i+1
        to_print[0,1]= max_diff_mean
        to_print[0,2]= avg_diff_mean
        to_print[0,3]= max_diff_targ
        to_print[0,4]= avg_diff_targ
        msg="with k_test varying across ipopt_interfaces:"
#        msg+="alphaSK=" + str(alphaSK) + ",tolIpopt=" + str(alphaSK)
#        msg+=",n_restarts_optimizer=" +str(n_restarts_optimizer)
        msg+=",numstart=" + str(numstart)
        msg+=",No_samples=" + str(No_samples)
        np.savetxt(file, to_print, header=msg, fmt='%2.16f')
        msg = "Cauchy:" + str(diff_mean) + ", max = " + str(round(max_diff_mean,3))
        msg += os.linesep
        msg += "Absolute:" + str(diff_targ) + ", max = " + str(round(max_diff_targ,3))
        print(msg)
        print("===================================")

        
    file.close()
    
    return 
        
#======================================================================
