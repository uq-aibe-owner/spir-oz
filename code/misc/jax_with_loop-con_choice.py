from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import jit, grad, jacrev, jacfwd
from cyipopt import minimize_ipopt



def objective(x):
    gamma = 0.25
    rho = -4
    rho_inv = 1/rho
    return np.power(np.sum(gamma * np.power(x[:4], rho)), rho_inv)

def eq_constraints(x,
                   p,
                   ):
    y = 100.0
    eqns = np.array([
        # constraint 1
        y - np.inner(p, x),
        # constraint 2
        np.power(x[1], 2) - 12,
                     ]
                    )
    return eqns

#def ineq_constraints(x):
#    return #np.prod(x) - 25

#-----------build and jit "objective" derivatives 
obj = jit(objective)
obj_grad = jit(grad(obj))
obj_hess = jit(jacrev(jacfwd(obj)))
#-----------build and jit "eq_constraints" derivatives
eq_ctt = jit(lambda x, p: eq_constraints(x, p))
eq_ctt_jac = jit(jacfwd(eq_ctt))
eq_ctt_hess = jacrev(jacfwd(eq_ctt))
#-----------build and jit "ineq_constraints" derivatives
#con_ineq = jit(ineq_constraints)
#ineq_ctt_jac = jit(jacfwd(ineq_ctt))
#ineq_ctt_hess = jacrev(jacfwd(ineq_ctt))  # hessian
#ineq_ctt_hessvp = jit(lambda x, v: ineq_ctt_hess(x) * v[0]) # hessian vector-product

#-----------define the jitted-with-price functions for the loop
def eq_ctt_jp(p):
    return lambda x: eq_ctt(x, p)
def eq_ctt_jac_jp(p):
    return lambda x: eq_ctt_jac(x, p)
def eq_ctt_hess_jp(p):
    return lambda x: eq_ctt_hess(x, p)

d = dict()

d[0] = np.array([1.0, 2.0, 3.0, 4.0])
d[1] = np.array([9.0, 7.0, 5.0, 3.0])
d[2] = np.array([1.0, 2.0, 3.0, 4.0])

# --------------- starting point
x0 =  np.array([15.0, 14.0, 13.0, 12.0])

# variable bounds: 0.0 <= x[i] <= 100
bnds = [(0.1, 100.0) for _ in range(x0.size)]
res = dict()
#-*-*-*-*-*-iteration over LPTH starts here 
for s in range(3):
    #-------feed in price vectors
    eq_ctt_fin = eq_ctt_jp(d[s])
    eq_ctt_jac_fin = eq_ctt_jac_jp(d[s])
    eq_ctt_hess_fin = eq_ctt_hess_jp(d[s])
    eq_ctt_hessvp_fin = jit(lambda x, v: eq_ctt_hess_fin(x) * v[0]) # hessian vector-product
    #-------wrap up constraints for cyipopt
    cons = [{'type': 'eq',
             'fun': eq_ctt_fin,
             'jac': eq_ctt_jac_fin,
             'hess': eq_ctt_hessvp_fin,
             }]

    #-------starting point
    x0 = np.array([15.0, 14.0, 13.0, 12.0])



    #-------execute solver
    res[s] = minimize_ipopt(obj,
                            jac=obj_grad,
                            hess=obj_hess,
                            x0=x0,
                            bounds=bnds,
                            constraints=cons,
                            options={'disp': 5,
                                     'obj_scaling_factor': -1.0,
                                    # 'warm_start_init_point': 'yes',
                                    # 'warm_start_same_structure': 'yes',
                                     'timing_statistics': 'yes',
                                     'print_timing_statistics': 'yes',
                                     }
                            )
#-*-*-*-*-*-loop ends here

# --------------- print solution, etc.
for s in range(3):
    print("the solution for iterate ", s, "is ", res[s])
#print("sol=", x_sol)
#diff_x_sol = np.zeros(len(x_sol) - 1)
#for i in range(len(x_sol) - 1):#
#    diff_x_sol.at[i].set(x_sol[i] - x_sol[i+1])
#for i in range(len(x_sol)-1):
#    print("diff=", x_sol[i] - x_sol[i+1])
