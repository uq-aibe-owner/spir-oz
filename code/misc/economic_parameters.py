from jax.config import config config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import jit, grad, jacrev, jacfwd
from cyipopt import minimize_ipopt

#--------------- economic parameters
NREG = 4    # number of regions
NSEC = 6    # number of sectors
LHZN = 1    # horizon length (Delta_s) in paper: forward-looking parameter
LPTH = 1    # path length (Tstar) in the paper
NPTH = 1    # number of paths Tstar + 1
BETA = 99e-2  # discount factor
ZETA1 = 1   # output multiplier in status quo state 0
ZETA2 = 95e-2 # output multiplier in tipped state 1
PHIA = 5e-1  # adjustment cost multiplier
PHIK = 5e-1  # weight of capital in production # alpha in CJ
#PHIM = 5e-1  # weight of intermediate inputs in production
TP01 = 1e-2 # transition probability from state 0 to 1
GAMMA = 5e-1 # power utility exponent
DELTA = 25e-3   # depreciation rate
ETA = 5e-1   # Frisch elasticity of labour supply
NRxS = NREG * NSEC
RHO = np.ones(NREG) # regional weights (population)
GAMMAhat = 1 - 1 / GAMMA # utility parameter (consumption denominator)
ETAhat = 1 + 1 / ETA # utility parameter
PHIL = 1 - PHIK # labour's importance in production
A = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # output multiplier
B = (1 - PHIK) * A * (A - DELTA) ** (-1 / GAMMA) # utility parameter (multiplies lab)
#XI = np.ones(NRxS) * 1 / NRxS # importance of kapital input to another
#MU = np.ones(NRxS) * 1 / NRxS # importance of one sector to another
#--------------- objective
#con_weights=GAMMA, elast_par=RHO, inv_elast_par=RHO_inv):
def objective(x, par=THETA, dicts=DICTS):
    # tail kapital
    var_final = X[(LHZN - 1) * NPOL : LHZN * NPOL]
    kap_tail = var_final[I["knx"]]
    # extract utilities
    sum_utl = 0.0
    for t in range(LHZN):
        var = X[t * NPOL : (t + 1) * NPOL]
        sum_utl += (BETA ** t * utility(var[I["con"]], var[I["lab"]]))
    return sum_utl + BETA ** LHZN * V_tail(kap_tail)


#--------------- equality constraints
def eq_constraints(x, state, par=THETA, dicts=DICTS):
    #income.at[0].set(10)
    eqns = np.zeros(4)
        # constraint 1
    eqns = eqns.at[0].set(income - np.inner(kap, x))
        # constraint 2
    for i in range(LHZN):
        eqns  = eqns.at[i].set(np.power(x[i] - i, 2))
    return eqns


#--------------- inequality constraints
#def ineq_constraints(x):
#    return #np.prod(x) - 25

from jax.config import config
config.update("jax_enable_x64", True)


#--------------- jit the functions
obj_jit = jit(objective)

res = dict()
#G(x) = eq_constraints(x, state=res[s-1][I["knx"]])
con_eq_jit = jit(lambda x, state: eq_constraints(x, state))
#con_ineq_jit = jit(ineq_constraints)

#------------ build and jit "objective" derivatives 
obj_grad = jit(grad(obj_jit))
obj_hess = jit(jacrev(jacfwd(obj_jit)))
#------------ build and jit "eq_constraints" derivatives
con_eq_jac = jit(jacfwd(con_eq_jit))
#con_ineq_jac = jit(jacfwd(con_ineq_jit))
con_eq_hess = jacrev(jacfwd(con_eq_jit))
con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
#------------ build and jit "ineq_constraints" derivatives
#con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
#con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

def WK_con_eq_jit(state): 
    return lambda x: con_eq_jit(x, state)

fin_con_eq_jit = WK_con_eq_jit(state=res[s-1][I["knx"]])

# ---------------variable bounds: 0.0 <= x[i] <= 100
    bnds = [(0.1, 100.0) for _ in range(x0.size)]

# -*-*-*-*-*-*-*-*iteration starts here 
for s in range(1, Tstar):

    # --------------- constraints
    cons = [{'type': 'eq',
         'fun': fin_con_eq_jit,
         'jac': con_eq_jac,
         'hess': con_eq_hessvp
         }
        ]
#,
#{'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}]

# --------------- starting point
    x0 = x[s] np.array([15.0, 14.0, 13.0, 12.0])



# ----------------- execute solver
    res[s] = minimize_ipopt(
        obj_jit,
        jac=obj_grad,
        hess=obj_hess,
        x0=x0,
        bounds=bnds,
        constraints=cons,
        options={'disp': 5,
                 'obj_scaling_factor': -1.0
                }
)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-* loop ends here

# --------------- print solution, etc.
x_sol = res["x"]
print("sol=", x_sol)
diff_x_sol = np.zeros(len(x_sol) - 1)
#for i in range(len(x_sol) - 1):#
#    diff_x_sol.at[i].set(x_sol[i] - x_sol[i+1])
for i in range(len(x_sol)-1):
    print("diff=", x_sol[i] - x_sol[i+1])
