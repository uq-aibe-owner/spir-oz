from casadi import SX, MX, DM, Function, nlpsol, vertcat
import numpy as np

NCTT = 2
p = SX.sym('p', 3)

x = SX.sym('x')
y = SX.sym('y')
z = SX.sym('z')

g0 = z + x + y - 2
g1 = z + (1 - x) ** p[0] - y
ctt = Function(
    'ctt',
    [x, y, z],
    [g0, g1]
)
G = SX.zeros(NCTT)
G[0] = g0
G[1] = g1
#G = MX(g0, g1)

nlp = {
    'x' : vertcat(x, y, z),
    'f' : p[2] * (x ** p[0] + p[1] * z ** p[0]),
    'g' : G, #[g1, g0], #z + (1 - x) ** p[0] - y, #ctt,
    'p' : p,
}

#==============================================================================
#-----------options for the ipopt (the solver)  and casadi (the frontend)
#------------------------------------------------------------------------------
ipopt_opts = {
    'ipopt.linear_solver' : 'mumps', #default=Mumps
    'ipopt.obj_scaling_factor' : 1.0, #default=1.0
    'ipopt.warm_start_init_point' : 'yes', #default=no
    'ipopt.warm_start_bound_push' : 1e-9,
    'ipopt.warm_start_bound_frac' : 1e-9,
    'ipopt.warm_start_slack_bound_push' : 1e-9,
    'ipopt.warm_start_slack_bound_frac' : 1e-9,
    'ipopt.warm_start_mult_bound_push' : 1e-9,
    'ipopt.fixed_variable_treatment' : 'relax_bounds', #default=
    'ipopt.print_info_string' : 'yes', #default=no
    'ipopt.accept_every_trial_step' : 'no', #default=no
    'ipopt.alpha_for_y' : 'primal', #default=primal, try 'full'?
}
casadi_opts = {
    'calc_lam_p' : False,
}
opts = ipopt_opts | casadi_opts
#-----------when HSL is available, we should also be able to run:
#opts = {"ipopt.linear_solver" : "MA27"}
#-----------or:
#opts = {"ipopt.linear_solver" : "MA57"}
#-----------or:
#opts = {"ipopt.linear_solver" : "HSL_MA86"}
#-----------or:
#opts = {"ipopt.linear_solver" : "HSL_MA97"}

#-----------the following advice comes from https://www.hsl.rl.ac.uk/ipopt/
#-----------"when using HSL_MA86 or HSL_MA97 ensure MeTiS ordering is 
#-----------compiled into Ipopt to maximize parallelism"

solver = nlpsol('solver', 'ipopt', nlp, opts)

X0 = 100 * np.arange(1,4)
LBX = np.zeros(len(X0))
UBX = np.ones(len(X0)) * 1e+3
LBG = np.zeros(NCTT)
UBG = np.zeros(NCTT)
P0 = np.array([2, 100, 1])

arg = dict()

def exclude_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

res = dict()

arg[0] = {
    "x0" : X0,
    "p" : P0,
    'lbx' : LBX,
    'ubx' : UBX,
    'lbg' : LBG,
    'ubg' : UBG,
}
#-----------solve:
res[0] = solver.call(arg[0])

#-----------obvious speed up when we feed in a similar solution:
arg[1] = exclude_keys(arg[0], {'p', 'x0', 'lam_g0'})
arg[1]['x0'] = res[0]['x']
arg[1]['p'] = P0
arg[1]['lam_g0'] = res[0]['lam_g']
#-----------solve:
res[1] = solver.call(arg[1])

#-----------vs when we don't:
arg[2] = exclude_keys(arg[0], {'p', 'x0', 'lam_g0'})
arg[2]['p'] = P0
arg[2]['x0'] = np.repeat(100, len(X0))
#-----------solve:
res[2] = solver.call(arg[2])

#==============================================================================
#-----------print results
for s in range(len(res)):
    print('the full dict of results for step', s, 'is\n', res[s])
