#!/usr/bin/env python3

# ======================================================================
#-----------instantaneous utility as a pure function
#-----------requires: "import economic_parameters as par"
def instant_utility(con,            # consumption vec of vars at given time
                    lab,            # labour vec of vars at given time
                    B=par.RWU,      # relative weight of con and lab in util
                    rho=par.RHO,    # regional-weights vec at given time
                    ):
    #-------log utility
    val = np.sum(rho * (np.log(con) - B * np.log(lab)))
    #-------general power utility:
    #val = np.sum(RHO * (con ** GAMMAhat / GAMMAhat - \
    #           B * lab ** ETAhat / ETAhat))
    return val

# ======================================================================
#-----------v-tail as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def V_tail(kap,             # kapital vec of vars at time t=LFWD 
           A=par.DPT,       # deterministic productivity trend
           beta=par.BETA,   # discount factor
           phik=par.PHIK,   # weight of capital in production
           tcs=par.TCS,     # tail consumption share
           u=efcn.instant_utility, # utility function
           ):
    #-------tail consumption vec
    tail_con = tcs * A * kap ** phik
    #-------tail labour vec normalised to one
    tail_lab = np.ones(len(kap))
    val = u(tail_con, tail_lab) / (1 - beta)
    return val

# ======================================================================
#-----------probabity of no tip by time t as a pure function
#-----------requires: "import economic_parameters as par"
def prob_noTip(t, # time step along a path
               tpt=par.TPT, # transition probability of tipping
               ):
    return (1 - tpt) ** (t)

# ======================================================================
#-----------expected output as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def expect_output(kap, # kapital vector of variables
                  lab, # labour vector of variables
                  t, # time step along a path
                  A=par.DPT, # deterministic productivity trend
                  phik=par.PHIK, # weight of capital in production # alpha in CJ
                  phil=par.PHIL, # weight of labour in production # 1-alpha in CJ
                  zeta=ZETA, # shock-value vector
                  pnt=efcn.prob_noTip, # probability of no tip by time t
                  ):
    val = (zeta[1] + pnt(t) * (zeta[0] - zeta[1])) * \
                A * (kap ** phik) * (lab ** phil)
    return val
# ======================================================================
#-----------adjustment costs of investment as a pure function
#-----------requires: "import economic_parameters as par"
def adjustment_cost(kap,
                    sav,
                    delta=par.DELTA, # depreciation rate
                    phig=par.PHIG, # adjustment cost multiplier
                    ):
    val = phig * kap * np.square(sav / kap - delta)
    return val

# ======================================================================
#-----------market clearing/budget constraint as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def market_clearing(kap,
                    con,
                    sav,
                    lab,
                    t,
                    a_c=efcn.adjustment_cost,
                    e_o=efcn.expect_output,
                    ):
    val = sum(con + sav + a_c(kap, sav) - e_o(kap, lab, t))
    return val

# ======================================================================
#-----------end-of-file
# ======================================================================
