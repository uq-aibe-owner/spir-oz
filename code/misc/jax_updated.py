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

def eq_constraints(x):
    y = 100.0
    p = np.array([1.0, 2.0, 3.0, 4.0])
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

#--------------- jit the functions
obj_jit = jit(objective)
con_eq_jit = jit(eq_constraints)
#con_ineq_jit = jit(ineq_constraints)

#--------------- build and jit "objective" derivatives 
obj_grad = jit(grad(obj_jit))
obj_hess = jit(jacrev(jacfwd(obj_jit)))
#--------------- build and jit "eq_constraints" derivatives
con_eq_jac = jit(jacfwd(con_eq_jit))
#con_ineq_jac = jit(jacfwd(con_ineq_jit))
con_eq_hess = jacrev(jacfwd(con_eq_jit))
con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
#--------------- build and jit "ineq_constraints" derivatives
#con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
#con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

# --------------- constraints
cons = [{'type': 'eq',
         'fun': con_eq_jit,
         'jac': con_eq_jac,
         'hess': con_eq_hessvp
         }
        ]
#,
#{'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}]

# --------------- starting point
x0 =  np.array([15.0, 14.0, 13.0, 12.0])

# variable bounds: 0.0 <= x[i] <= 100
bnds = [(0.1, 100.0) for _ in range(x0.size)]

# ----------------- execute solver
res = minimize_ipopt(
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

# --------------- print solution, etc.
x_sol = res["x"]
print("sol=", x_sol)
diff_x_sol = np.zeros(len(x_sol) - 1)
#for i in range(len(x_sol) - 1):#
#    diff_x_sol.at[i].set(x_sol[i] - x_sol[i+1])
for i in range(len(x_sol)-1):
    print("diff=", x_sol[i] - x_sol[i+1])
