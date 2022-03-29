** This is a code of SCEQ to solve a multi-country real business cycle problem *
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

scalar starttime;
starttime = jnow;

*==============================================================================
*-----------sets
*==============================================================================
set r regions /Aus, Qld/;
alias(rr, r)
;
set j sectors /1*3/;
alias(i, j)
;
*-----------number of different paths + 1 (the extra one is for error checking
*-----------at the last period of interest)
set p /a, b, c, d, e, f, g, h, i, j /;
*set p /a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r /;

*==============================================================================
*-----------basic economic parameters
*==============================================================================
parameters
    BETA                     discount rate /0.98/
    ALPHA                    capital cost share /0.33/
    DELTA                    capital stock depreciation /0.025/
    PHI_ADJ                  adjustment cost parameter /0.5/
    GAMMA                    intertemporal elasticity of substitution /0.5/
    GAMMA_HAT                utility parameter
    ETA                      Frisch elasticity of labor supply /0.5/
    ETA_HAT                  utility parameter
    A                        technology parameter
    B                        relative weight of consumption and leisure
    LFWD                     look-forward parameter / 15/
    s                        starting period for each look forward
    T_STAR                    number of periods of interest 
    REG_WGHT(r)              regional weights (eg population)
    EoS_KAP                  elasticity of substitution for kapital /0.4/
    RHO                      exponent of the ces function
    RHO_HAT                  inverse of RHO
    CON_SHR(i)               share of each commodity in consumption
    LAB_SHR(i)               share of each type of labour in utility
    INV_SHR(i, j)               share of each commodity in saving
    INV_MIN                  lower bound of investment 
    KAP0(r, i)               initial capital
    kmin                     smallest capital    / 0.1 /
    kmax                     largest capital     / 10 /
    ZETA1                    TFP before shock   / 1 /
    ZETA2                    TFP after shock   / 1e-0 /
    PROB1                    one period probability of jump of TFP / 0.01 /
    TL_CON_SHR               tail consumption share (of output) / 0.45 /
    CON_SHR(i)               consumption share for each sector
    T_ALL                    for the set of all time periods
;
*==============================================================================
*-----------derived economic parameters
*==============================================================================
T_STAR = card(p) - 1;
T_ALL = card(p) + LFWD;
A = (1 - (1 - DELTA) * BETA) / (ALPHA * BETA);
GAMMA_HAT = 1 - (1 / GAMMA);
B = (1 - ALPHA) * A * (A - DELTA) ** (-1 / GAMMA);
ETA_HAT = 1 + (1 / ETA);
RHO = (EoS_KAP - 1) / EoS_KAP; 
RHO_HAT = 1 / RHO;

*-----------the following is the vector of population weights: 
*-----------it enters the objective function and determines demand
loop(r,
  REG_WGHT(r) = 1 / ord(r);
);
loop(i,
  CON_SHR(i) = ord(i) / ((card(i) * (card(i) + 1)) / 2);
  LAB_SHR(i) = ord(i) / ((card(i) * (card(i) + 1)) / 2);
  INV_SHR(i, j) = ord(i) / ((card(i) * (card(i) + 1)) / 2);
*-----------alternative, less symmetric parametrisation for INV_SHR
* loop(j,
*   INV_SHR(i, j) = ord(i) / ((card(i) * (card(i) - 1)) / 2) 
*    / (ord(j) / ((card(j) * (card(j) - 1)) / 2))
* ;
* );
);
*-----------
INV_MIN = 0.9 * DELTA;
*-----------the following is initial kapital: it will vary across regions
*KAP0(r) = kmin + (kmax-kmin)*(ord(r)-1)/(card(r)-1);
*KAP0(r) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(r)-1)/(card(r)-1));
KAP0(r, i) = 1;

display A, B, ETA_HAT, KAP0;
*-----------the following won't work as s is a parameter not a set
*interval(s) = yes$(s < ord(t) and ord(t) < LFWD + s)

*==============================================================================
*-----------derived set for timeline that transcends LFWD and T_STAR
*==============================================================================
set t time / 1 * 30 /;
alias(tt,t);
*-----------
parameter
  probs(t)                probability of jump of TFP
  E_shk(r, i, t)           expected shock (exogenous)
;
*==============================================================================
*-----------basic (economic) variables
*==============================================================================
Variables
    obj                       objective criterion
    inv(r, i, j, t)           investment
    utility(r, t)             instantaneous utility
;
*------------------------------------------------------------------------------
Positive variables

    con(r, i, t)              consumption
    kap(r, i, t)              kapital stock
    lab(r, i, t)              labor supply
*-----------intermediate variables
    con_sec(r, t)             consumption aggregate (across sectors)
    inv_sec(r, j, t)             kapital aggregate (across sectors)
    lab_sec(r, t)             labour aggregate (across sectors)
    adj(r, i, t)              kapital adjustment costs
    out(r, i, t)              output
;

*============================================================================== 
*-----------Bound Constraints
*============================================================================== 
kap.lo(r, i, t) = 0.001;
kap.up(r, i, t) = 1000;
con.lo(r, i, t) = 0.001;
con.up(r, i, t) = 1000;
lab.lo(r, i, t) = 0.001;
lab.up(r, i, t) = 1000;
inv.lo(r, i, j, t) = INV_MIN;
con_sec.lo(r, t) = 0.001;
*============================================================================== 
*-----------Initial Guess
*============================================================================== 
s = 1;
inv.L(r, i, j, t) = DELTA;
kap.L(r, i, t) = 1e+0;
lab.L(r, i, t) = 1e+0;
con.L(r, i, t) = A-DELTA;
*==============================================================================
*-----------equation declarations (over entire sets)
*==============================================================================
*-----------declarations are applied globally (eg on the whole of 1..71)
Equations
*-----------declarations for intermediate-variables
    con_sec_eq(r, t)           consumption aggregate (across sectors)
    inv_sec_eq(r, j, t)           kapital aggregate (across sectors)
    lab_sec_eq(r, t)           labour aggregate (across sectors)
    adj_eq(r, i, t)            kapital adjustment costs
    out_eq(r, i, t)            output 
    utility_eq(r, t)           the utility function
    tail_utility_eq(r, t)       the tail utility function
    obj_eq                     Objective function
*-----------canonical equations
    dynamics_eq(r, i, t)        Law of Motion for Capital Stock
    market_clearing_eq(i, t)            budget constraint before jump
*-----------other states
    tipped_market_clearing_eq(i, t)      budget constraint after jump
;
*==============================================================================
*-----------equation definitions for each s along the path
*==============================================================================
*-----------definitions for some intermediate-variables
con_sec_eq(r, t) $ (s <= ord(t) and ord(t) < LFWD + s)..
  con_sec(r, t) =e= prod(i, con(r, i, t) ** CON_SHR(i))
;
lab_sec_eq(r, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  lab_sec(r, t) =e= prod(i, lab(r, i, t) ** LAB_SHR(i))
;
inv_sec_eq(r, j, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  inv_sec(r, j, t)
    =e= (sum(i, INV_SHR(i, j) * inv(r, i, j, t) ** RHO)) ** RHO_HAT
;
adj_eq(r, i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  adj(r, i, t)
    =e= (PHI_ADJ/2) * kap(r, i, t) * sqr(kap(r, i, t + 1) / kap(r, i, t) - 1)
;
out_eq(r, i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  out(r, i, t) =e= A * kap(r, i, t) ** ALPHA * lab(r, i, t) ** (1 - ALPHA)
;
*------------------------------------------------------------------------------
*-----------canonical equations
*------------------------------------------------------------------------------
dynamics_eq(r, i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  kap(r, i, t + 1) =e= (1 - DELTA) * kap(r, i, t) + inv_sec(r, i, t)
;
market_clearing_eq(i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  sum(r,
    E_shk(r, i, t) * out(r, i, t)
    - con(r, i, t)
    - adj(r, i, t)
    - sum(j, inv(r, i, j, t))
  )
  =e= 0
;
*------------------------------------------------------------------------------
*-----------other states
*------------------------------------------------------------------------------
tipped_market_clearing_eq(i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  sum(r,
    ZETA2 * out(r, i, t) 
    - con(r, i, t)
    - adj(r, i, t)
    - sum(j, inv(r, i, j, t))
  )
  =e= 0
;
*------------------------------------------------------------------------------
*-----------the sequence of utility flows per region and time period
*------------------------------------------------------------------------------
utility_eq(r, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  utility(r, t) =e= 
*-----------the consumption part:
    con_sec(r, t) ** GAMMA_HAT / GAMMA_HAT
*-----------the labour part:
    - B * lab_sec(r, t) ** ETA_HAT / ETA_HAT
;
tail_utility_eq(r, t) $ (ord(t) = LFWD + s)..
*-----------tail/continuation utility, where tail labour is normalised to one:
  utility(r, t) =e= 
*-----------in the consumption part, a fixed share of output is consumed
    (prod(i, 
      (TL_CON_SHR * A * kap(r, i, t)) ** CON_SHR(i)) ** ALPHA) ** GAMMA_HAT / GAMMA_HAT 
*-----------and the labour part:
    - B / (1 - BETA)
;
*------------------------------------------------------------------------------
*-----------the objective function
*------------------------------------------------------------------------------
obj_eq..
  obj =e= 
    sum(r, REG_WGHT(r) * sum(t $ (s <= ord(t) and ord(t) <= LFWD),
      BETA ** (ord(t) - s) * utility(r, t)))
;
*==============================================================================
*-----------solver options
options limrow = 0, limcol = 0;
option reslim = 10000;
option iterlim = 10000;
option solprint = off;
*-----------which solver to use, comment out one of the following:
*option nlp = ipopt;
option nlp = conopt;

*==============================================================================
*-----------instantiate models with corresponding equations
model busc /
        obj_eq,
        dynamics_eq,
        market_clearing_eq,
        con_sec_eq,
        inv_sec_eq,
        lab_sec_eq,
        adj_eq,
        out_eq,
        utility_eq
        /;
model busc_tipped /
        obj_eq,
        dynamics_eq,
        tipped_market_clearing_eq,
        con_sec_eq,
        inv_sec_eq,
        lab_sec_eq,
        adj_eq,
        out_eq,
        utility_eq
        /;

