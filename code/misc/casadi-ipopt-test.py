from casadi import *

p = MX.sym('p', 3)

x = MX.sym('x')
y = MX.sym('y')
z = MX.sym('z')

nlp = nlp = {'x' : vertcat(x, y, z),
             'f' : p[2] * (x ** p[0] + p[1] * z ** p[0]),
             'g' : z + (1 - x) ** p[0] - y,
             'p' : p
             }

#==============================================================================
#-----------options for the ipopt (the solver)  and casadi (the frontend)
#------------------------------------------------------------------------------
ipopt_opts = {'ipopt.linear_solver' : 'mumps', #default=Mumps
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
casadi_opts = {'calc_lam_p' : False,
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

res = dict()
res[0] = solver(x0=DM([1, 2, 3]),
                p = DM([2.5, 100, 1]),
                )
#-----------obvious speed up when we feed in a similar solution:
res[1] = solver(x0=res[0]["x"] + [.01, .01, .01],
                p=DM([2, 100, 1]),
                lam_g0=res[0]["lam_g"],
                )
#-----------vs when we don't:
res[2] = solver(x0=[10, 11, 12], p = DM([2, 100, 1]))
for s in range(len(res)):
    print('the dict of results for step', s, 'is', res[s])

