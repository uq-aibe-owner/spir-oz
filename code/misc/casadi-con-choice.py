import numpy as np
from casadi import *

opti = casadi.Opti()

#==============================================================================
#-----------parameters
#------------------------------------------------------------------------------
RHO = -4
RHOinv = 1/RHO
p = opti.parameter(3)
#==============================================================================
#-----------variables: these are symbolic expressions of casadi type MX
#------------------------------------------------------------------------------
x = opti.variable()
y = opti.variable()
#z = opti.variable()

#==============================================================================
#-----------options for the ipopt (the solver)  and casadi (the frontend)
#------------------------------------------------------------------------------
ipopt_opts = {'ipopt.linear_solver' : 'mumps', #default=Mumps
              'ipopt.obj_scaling_factor' : -1.0, #default=1.0
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
#==============================================================================
#-----------instantaneous utility as a pure function
#-----------requires: "import economic_parameters as par"
def instant_utility(con,            # consumption vec of vars at given time
                    lab,            # labour vec of vars at given time
                    #B=RWU,      # relative weight of con and lab in util
                    rho=RHO,    # regional-weights vec at given time
                    rho_inv=RHOinv
                    #gh=GAMMAhat,
                    #eh=ETAhat,
                    ):
    #-------log utility
    #val = np.sum(rho * (np.log(con) - B * np.log(lab)))
    #-------general power utility:
    #val = np.sum(rho * (2 * con ** gh / gh - B * lab ** eh / eh))
    #-------CES utility with labour as a good
    val = np.power(.5 * np.power(con, rho) + .5 * np.power(lab, rho),  rho_inv)
    return val
opti.minimize(np.power(.5 * np.power(x, RHO) + .5 * np.power(y, RHO),  RHOinv))
#opti.minimize(instant_utility(con=x, lab=y))
opti.subject_to(p[0] * x + p[1] * y - p[2] == 0)
opti.subject_to(x >= 0)
opti.subject_to(y >= 0)


opti.solver('ipopt', opts)

opti.set_value(p, [5, 5, 100])
opti.set_initial(vertcat(x, y), vertcat(100, 200))
#opti.set_initial(y, 2)
#opti.set_initial(z, 3)

sol = dict()
sol[0] = opti.solve()

#-----------Below, opti.x is the full ipopt vector:
#print(sol[0].value(opti.x))
#print(sol[0].value(opti.lam_g))
#-----------Here it is one of the three variables:
#print(sol[0].value(vertcat(x, y)))
#-----------The following will not work:
#print(sol[0].value([x, y , z]))

#-----------Next run: we need to set new values.
#-----------for parameters:
opti.set_value(p, [5, 50, 100])

#-----------for variables: (opti.x is the full ipopt vector!!)
opti.set_initial(vertcat(x, y), sol[0].value(opti.x))
#opti.set_initial(x, sol[0].value(x))
#opti.set_initial(y, sol[0].value(y))
#opti.set_initial(z, sol[0].value(z))

#-----------and a warm start for lam_g (prices)
opti.set_initial(opti.lam_g, sol[0].value(opti.lam_g))

#-----------The following will not work:
#opti.set_initial(opti.g, sol[0].value(opti.g))

sol[1] = opti.solve()

for s in range(len(sol)):
    print(sol[s].value(opti.x))
    print(sol[s].value(opti.g))
    print(sol[s].value(opti.lam_g))

