
#     Simon Scheidegger, 01/19
#     edited by Patrick O'Callaghan, with Cameron Gordon and Josh Aberdeen, 11/2021
# ======================================================================

import solver as solver
from parameters import *  # parameters of model
from variables import *
from equations import *
import postprocessing as post  # computes the L2 and Linfinity error of the model
import iteration
import iteration_post
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
from numpy.random import PCG64
from datetime import datetime
# ======================================================================
# Start with Value Function i_pth


# Set up a container to save the k/obj pairs as they are sampled & optimised

start = time.time()


for i_pth in range(0, Tstar):
# terminal value function
    if (i_pth==0):
        print("start with untipped path")
        iteration.sceq(i_pth)
    
    else:     
        print("Now we are running the tipped path", i_pth)
        iteration_post.sceq_post(i_pth)






now = datetime.now()
dt = int(now.strftime("%H%M%S%f"))
rngm = np.random.default_rng(dt)  # fix seed #move to main so it doesnt re-initialise
#print(rng)

for iter in range(numstart, numits):

    # terminal value function
    # run the model:
    res_fin = i_pth.sceq(iter)

# ======================================================================
print("===============================================================")
print(" ")
print(
    " Computation of a growth model of dimension ",
    n_agt,
    " finished after ",
    numits,
    " steps",
)
print(" ")
print("===============================================================")
# ======================================================================

# compute errors
#avg_err = post.ls_error(n_agt, numstart, numits, No_samples_postprocess)

# ======================================================================
#print("===============================================================")
#print(" ")
# print " Errors are computed -- see error.txt"
#print(" ")
#print("===============================================================")
# ======================================================================
end = time.time()

#def plot_scatterplot():
#
#    # for all sampled points (not all will have converged, but will give an approximate view of the surface)
#    sample_container["kap"] = np.array(sample_container["kap"])
#    sample_container["value"] = np.array(sample_container["value"])
#
#    matplotlib.use("tkagg")
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection="3d")  # (projection='3d')
#    ax.set_xlabel("k (sector 1)")
#    ax.set_ylabel("k (sector 2)")
#    ax.set_zlabel("Value")
#
#    # colormap = matplotlib.cm(sample_container['i_pth'])
#
#    img = ax.scatter(
#        sample_container["kap"][:, 0],
#        sample_container["kap"][:, 1],
#        sample_container["value"],
#        c=sample_container["i_pth"],
#    )
#    plt.colorbar(img)
#
#    # plt.show()
#    plt.show()

def get_gaussian_process():
    with open("./restart/restart_file_step_" + str(numits - 1) + ".pcl", "rb") as fd:
        gp_old = pickle.load(fd)

    fd.close()
    return gp_old

def get_values(kap):
    Gaussian_Process = get_gaussian_process()
    values = Gaussian_Process.predict(kap, return_std=False)
    return values

def generate_random_k_vals(): 
    return rngm.uniform(kap_L+0.2, kap_U-0.2, (No_samples, n_agt)) 

def solve_for_kvals(kap, n_agt, gp_old): 

    result = np.empty((kap.shape[0]))
    for iter in range(kap.shape[0]): 
        result[iter] = solver.ipopt_interface(k_init=kap[iter], n_agt=n_agt, gp_old=gp_old,initial=False, verbose=verbose)['obj']

    return result

#def extract_variables(default=True, k_vals=None):
#    # extract the consumption, investment, labour variables (from the final i_pth if default=True)
#    # if false, specify random points and calculate
#
#    kap_tst = []
#    val_tst = []
#    consumption = []
#    investment = []
#    labor = []
#
#    if default:
#        for iter in ctnr:
#            kap_tst.append(i[0])
#            val_tst.append(i[1])
#            consumption.append(i[3])
#            investment.append(i[4])
#            labor.append(i[5])
#
#    kap_tst = np.array(kap_tst)
#    val_tst = np.array(val_tst)
#    consumption = np.array(consumption)
#    investment = np.array(investment)
#    labor = np.array(labor)
#
#    return kap_tst, val_tst, consumption, investment, labor



#kap_tst, val_tst, consumption, investment, labor = extract_variables()
# print(consumption)
# print(investment)
# print(labor)
def help():
    print(" ========== Finished ==========")
    print("Time elapsed: ", round(end - start, 2))

    print("Call variables kap_tst, val_tst, consumption, investment, labor")
    print("Use plot_scatterplot() to visualise")
    print("Use get_gaussian_process() to get the Gaussian Process")
    print(
        "Predict values for a given level of capital k with get_values(k), e.g. values = get_values(kap_tst)"
    )
    print("For options / prompts type help()")

help()

avg_err = post.ls_error(n_agt, numstart, numits, No_samples_postprocess)


#plot_scatterplot()


"""

"""

# test and debug
