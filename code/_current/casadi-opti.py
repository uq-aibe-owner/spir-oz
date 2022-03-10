import numpy as np
from casadi import *

#==============================================================================
#-----------instantiate problem class
#------------------------------------------------------------------------------
opti = casadi.Opti()

#==============================================================================
#-----------parameters
#------------------------------------------------------------------------------
#-----------economic parameters
#-----------basic economic parameters
NREG = 2       # number of regions
NSEC = 2        # number of sectors
PHZN = NTIM = LFWD = 2# look-forward parameter / planning horizon (Delta_s)
NPOL = 3        # number of policy types: con, lab, knx, #itm
NITR = LPTH = 2# path length (Tstar): number of random steps along given path
NPTH = 1        # number of paths (in basic example Tstar + 1)
BETA = 95e-2    # discount factor
ZETA0 = 1       # output multiplier in status quo state 0
ZETA1 = 95e-2   # output multiplier in tipped state 1
PHIA = 5e-1     # adjustment cost multiplier
PHIK = 33e-2  # weight of capital in production # alpha in CJ
TPT = 1e-2      # transition probability of tipping (from state 0 to 1)
GAMMA = 5e-1    # power utility exponent
DELTA = 25e-3   # depreciation rate
ETA = 5e-1      # Frisch elasticity of labour supply
RHO = .5#np.ones(NREG) # regional weights (population)
TCS=75e-2       # Tail Consumption Share
#-----------suppressed basic parameters
#PHIM = 5e-1    # weight of intermediate inputs in production
#XI = np.ones(NRxS) * 1 / NRxS # importance of kapital input to another
#MU = np.ones(NRxS) * 1 / NRxS # importance of one sector to another

#------------------------------------------------------------------------------
#-----------derived economic parameters
NRxS = NREG * NSEC
GAMMAhat = 1 - 1 / GAMMA    # utility parameter (consumption denominator)
ETAhat = 1 + 1 / ETA        # utility parameter
PHIL = 1 - PHIK             # labour's importance in production
DPT = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # deterministic prod trend
RWU = (1 - PHIK) * DPT * (DPT - DELTA) ** (-1 / GAMMA) # Rel Weight in Utility
ZETA = np.array([ZETA0, ZETA1])
NVAR = NPOL * LFWD * NSEC * NREG    # total number of variables
X0 = np.ones(NVAR)      # our initial warm start 
# k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
KAP0 = np.ones(NREG) # how about NSEC ????
#for j in range(n_agt):
#    KAP0[j] = np.exp(
#        np.log(kap_L) + (np.log(kap_U) - np.log(kap_L)) * j / (n_agt - 1)
#    )
#-----------suppressed derived economic parameters
#IVAR = np.arange(0,NVAR)    # index set (as np.array) for all variables

#------------------------------------------------------------------------------
#-----------probabity of no tip by time t as a pure function
#-----------requires: "import economic_parameters as par"
def prob_no_tip(tim, # time step along a path
               tpt=TPT, # transition probability of tipping
               ):
    return (1 - tpt) ** tim

#-----------prob no tip as a vec of parameters
PNT = np.ones(LPTH)
for t in range(LPTH):
    PNT[t] = prob_no_tip(t)

#-----------if t were a variable, then, for casadi, we could do:
#t = c.SX.sym('t')
#pnt = c.Function('cpnt', [t], [prob_no_tip(t)], ['t'], ['p0'])
#PNT = pnt.map(LPTH)         # row vector 

#-----------casadi wrapper for parameters
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
RHOinv=1/RHO
#-----------instantaneous utility as a pure function
#-----------requires: "import economic_parameters as par"
def instant_utility(con,            # consumption vec of vars at given time
                    lab,            # labour vec of vars at given time
                    #B=RWU,      # relative weight of con and lab in util
                    rho=0.5,    # regional-weights vec at given time
                    rho_inv=1/RHO
                    #gh=GAMMAhat,
                    #eh=ETAhat,
                    ):
    #-------log utility
    #val = np.sum(rho * (np.log(con) - B * np.log(lab)))
    #-------general power utility:
    #val = np.sum(rho * (2 * con ** gh / gh - B * lab ** eh / eh))
    #-------CES utility with labour as a good
    val = np.ones(3) \
    * np.power(.5 * np.power(con, rho) + .5 * np.power(lab, rho),  rho_inv)
    return sum1(val)
#fh = MX(mac(DM.ones(1, 3), pow((0.5 * pow(x_1, 0.5) + 0.5 * pow(x_2,0.5)), 2) * DM.ones(3, 1),0))
#inst_util_cas = Function('iuc', [x, y], instant_utility(con=x, lab=y)
#opti.minimize(np.power(.5 * np.power(x, RHO) + .5 * np.power(y, RHO),  RHOinv))
opti.minimize(instant_utility(con=x, lab=y))#.fold(N))
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

