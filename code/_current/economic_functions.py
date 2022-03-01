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
#-----------full utility (summing over time) as a pure fcn


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
def prob_no_tip(tim, # time step along a path
               tpt=par.TPT, # transition probability of tipping
               ):
    return (1 - tpt) ** tim

# ======================================================================
#-----------expected output as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def expected_output(kap,                        # kap vector of vars
                    lab,                        # lab vector of vars
                    tim,                          # time step along a path
                    A=par.DPT,                  # determistic prod trend
                    phik=par.PHIK,              # weight of kap in prod
                    phil=par.PHIL,              # weight of lab in prod
                    zeta=ZETA,                  # shock-value vector
                    pnot=efcn.prob_no_tip,      # prob no tip by t
                    ):
    y = A * (kap ** phik) * (lab ** phil)       # output
    E_zeta = zeta[1] + pnot(tim) * (zeta[0] - zeta[1]) # expected shock
    val = E_zeta * y
    return val
# ======================================================================
#-----------adjustment costs of investment as a pure function
#-----------requires: "import economic_parameters as par"
def adjustment_cost(kap,
                    knx,
                    phia=par.PHIA, # adjustment cost multiplier
                    ):
    # since sav/kap - delta = (knx - (1 - delta) * kap)/kap - delta = ..
    # we can therefore rewrite the adjustment cost as
    val = phia * kap * np.square(knx / kap - 1)
    return val

# ======================================================================
#-----------market clearing/budget constraint as a pure function
#-----------requires: "import economic_parameters as par"
#-----------requires: "import economic_functions as efcn"
def market_clearing(kap,
                    knx,
                    con,
                    lab,
                    tim,
                    adjc=efcn.adjustment_cost, # Gamma in Cai-Judd
                    E_f=efcn.expected_output,
                    ):
    sav = knx - (1 - delta) * kap
    val = sum(E_f(kap, lab, tim) - con - sav - adjc(kap, sav))
    return val

# ======================================================================
#-----------end-of-file
# ======================================================================
