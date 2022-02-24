#---------------basic economic parameters
NREG = 4        # number of regions
NSEC = 6        # number of sectors
LFWD = 1        # look-forward parameter / time horizon length (Delta_s) in paper 
LPTH = 1        # path length (Tstar): number of random steps along a given path
NPTH = 1        # number of paths Tstar + 1
BETA = 99e-2    # discount factor
ZETA0 = 1       # output multiplier in status quo state 0
ZETA1 = 95e-2   # output multiplier in tipped state 1
PHIG = 5e-1     # adjustment cost multiplier
PHIK = 5e-1     # weight of capital in production # alpha in CJ
TPT = 1e-2      # transition probability of tipping (from state 0 to 1)
GAMMA = 5e-1    # power utility exponent
DELTA = 25e-3   # depreciation rate
ETA = 5e-1      # Frisch elasticity of labour supply
RHO = np.ones(NREG) # regional weights (population)
TCS=0.75
#---------------suppressed basic parameters
#PHIM = 5e-1    # weight of intermediate inputs in production
#XI = np.ones(NRxS) * 1 / NRxS # importance of kapital input to another
#MU = np.ones(NRxS) * 1 / NRxS # importance of one sector to another
#---------------derived economic parameters
NRxS = NREG * NSEC
GAMMAhat = 1 - 1 / GAMMA    # utility parameter (consumption denominator)
ETAhat = 1 + 1 / ETA        # utility parameter
PHIL = 1 - PHIK             # labour's importance in production
DPT = (1 - (1 - DELTA) * BETA) / (PHIK * BETA) # deterministic productivity trend
B = (1 - PHIK) * A * (A - DELTA) ** (-1 / GAMMA) # relative weight of con and lab in utility
ZETA = np.array([ZETA0, ZETA1])
