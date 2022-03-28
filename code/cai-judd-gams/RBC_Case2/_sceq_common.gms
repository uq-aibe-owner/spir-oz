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
set r regions /1*2/;
alias(rr, r)
;
set j sectors /1*2/;
alias(i, j)
;
sets
    t time /1*71/
;
alias(tt,t);
*-----------number of different paths + 1 (the extra one is for error checking
*-----------at the last period of interest)
set p /1*10/;

*==============================================================================
*-----------basic economic parameters
*==============================================================================
parameters
    BETA                     discount rate /0.98/
    ALPHA                    capital cost share /0.33/
    DELTA                    capital stock depreciation /0.5/
    PHI_ADJ                      adjustment cost parameter /0.5/
    GAMM                    intertemporal elasticity of substitution /0.5/
    GAMM_HAT                 utility parameter
    ETA                      Frisch elasticity of labor supply /0.5/
    ETA_HAT                   utility parameter
    A                        technology parameter
    B                        relative weight of consumption and leisure
    LFWD                        look-forward parameter / 10 /
    s                        starting period for each look forward
    Tstar                    number of periods of interest 
    REG_WGHT(r)                 regional weights (eg population)
    EoS_KAP                     elasticity of substitution for kapital /0.4/
    RHO                     exponent of the ces function
    RHO_INV                 inverse of RHO
    CON_SHR(i)               share of each commodity in consumption
    LAB_SHR(i)               share of each type of labour in utility
    INV_MIN                     lower bound of investment 
    KAP0(r, i)                    initial capital
    kmin                     smallest capital    / 0.1 /
    kmax                     largest capital     / 10 /
    ZETA1                    TFP before shock   / 1 /
    ZETA2                    TFP after shock   / 0.95 /
    PROB1                    one period probability of jump of TFP / 0.01 /
    probs(t)                 probability of jump of TFP
    E_shk(r, i, t)                   expected shock (exogenous) 
    TL_CON_SHR                      tail consumption share (of output) / .45 /
    CON_SHR(i)             consumption share for each sector
;

*==============================================================================
*-----------derived economic parameters
*==============================================================================
Tstar = card(p)-1;
A = (1 - (1 - DELTA) * BETA) / (ALPHA * BETA);
GAMM_HAT = 1 - (1 / GAMM);
B = (1 - ALPHA) * A * (A - DELTA) ** (-1 / GAMM);
ETA_HAT = 1 + (1 / ETA);
RHO = (EoS_KAP - 1) / EoS_KAP; 
RHO_INV = 1 / RHO;

*-----------the following is the vector of population weights: 
*-----------it enters the objective function and determines demand
REG_WGHT(r) = 1;
CON_SHR(i) = 1 / card(i);
LAB_SHR(i) = 1 / card(i);
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
*-----------basic (economic) variables
*==============================================================================
Variables
    obj                        objective criterion
    inv(r, i, t)              investment
    utility(r, t)            instantaneous utility
;
*------------------------------------------------------------------------------
Positive variables

    con(r, i, t)              consumption
    kap(r, i, t)              kapital stock
    lab(r, i, t)              labor supply
*-----------intermediate variables
    con_sec(r, t)            consumption aggregate (across sectors)
    kap_sec(r, t)            kapital aggregate (across sectors)
    lab_sec(r, t)            labour aggregate (across sectors)
    adj(r, i, t)             kapital adjustment costs
    out(r, i, t)             output
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
inv.lo(r, i, t) = INV_MIN;

*============================================================================== 
*-----------Initial Guess
*============================================================================== 
s = 1;
inv.L(r, i, t) = DELTA;
kap.L(r, i, t) = 1;
lab.L(r, i, t) = 1;
con.L(r, i, t) = A-DELTA;
*==============================================================================
*-----------equation declarations (over entire sets)
*==============================================================================
*-----------declarations are applied globally (eg on the whole of 1..71)
Equations
*-----------declarations for intermediate-variables
    con_sec_eq(r, t)           consumption aggregate (across sectors)
    kap_sec_eq(r, t)           kapital aggregate (across sectors)
    lab_sec_eq(r, t)           labour aggregate (across sectors)
    adj_eq(r, i, t)            kapital adjustment costs
    out_eq(r, i, t)            output 
    utility_eq(r, t)            the utility function
    obj_eq                              Objective function
*-----------canonical equations
    dynamics_eq(r, i, t)        Law of Motion for Capital Stock
    market_clearing_eq(i, t)            budget constraint before jump
*-----------other states
    tipped_market_clearing_eq(i, t)      budget constraint after jump
;
*==============================================================================
*-----------equation definitions for each s along the path
*==============================================================================
If (((s <= ord(t)) and (ord(t) <= LFWD + s)),
  If ((ord(t) < LFWD + s),
*-----------definitions for some intermediate-variables
con_sec_eq(r, t).. 
      con_sec(r, t) =e= prod(i, con(r, i, t) ** CON_SHR(i))
    ;
    lab_sec_eq(r, t).. 
      lab_sec(r, t) =e= prod(i, lab(r, i, t) ** LAB_SHR(i))
    ;
    kap_sec_eq(r, t).. 
      kap_sec(r, t) =e= (sum(i, KAP_SHR(i) * kap(r, i, t) ** RHO)) ** RHO_INV
    ;
    adj_eq(r, i, t)..
      adj(r, i, t) =e= (PHI_ADJ/2) * k(r, i, t) 
        * sqr(inv(r, i, t) / k(r,t) - DELTA)
    ;
    out_eq(r, i, t).. 
      out(r, i, t) =e= A * kap(r, i, t) ** ALPHA * lab(r, i, t) ** (1 - ALPHA)
    ;
*------------------------------------------------------------------------------
*-----------the sequence of utility flows per region and time period
*------------------------------------------------------------------------------
    utility_eq(r, t)..
      utility(r, t) =e=
*-----------the consumption part:
        con_sec(r, t) ** GAMM_HAT / GAMM_HAT
*-----------the labour part:
        - B * lab_sec(r, t) ** ETA_HAT / ETA_HAT
    ;
*------------------------------------------------------------------------------
*-----------canonical equations
*------------------------------------------------------------------------------
    dynamics_eq(r, t)..
      kap(r, i, t + 1) =e= (1 - DELTA) * kap(r, i, t) + inv(r, i, t)
    ;
    market_clearing_eq(t)..
      sum(r, E_shk(r, i, t) * out(r, i, t)
        - c(r, i, t) - inv(r, i, t) - adj(r, i, t)) =e= 0
    ;
*------------------------------------------------------------------------------
*-----------other states
    tipped_market_clearing_eq(t)..
      sum(r, ZETA2 * out(r, i, t) 
        - c(r, i, t) - inv(r, i, t) - adj(r, i, t)) =e= 0
    ;
*-----------tail/continuation utility, where tail labour is normalised to one:
  else (ord(t) = LFWD + s),
    utility_eq(r, t)..
      utility(r, t) =e=
*-----------in the consumption part, a fixed share of output is consumed
        ((TL_CON_SH * A * kap_sec(r, t)) ** ALPHA) ** GAMM_HAT / GAMM_HAT 
        - B / (1 - BETA)
    ;
  );
*------------------------------------------------------------------------------
*-----------the objective function
*------------------------------------------------------------------------------
    obj_eq.. 
      obj =e= 
        sum(r, REG_WGHT(r) * sum(t, BETA ** (ord(t) - s) * utility(r, t)))
    ;
);

*==============================================================================
*-----------solver options
options limrow = 0, limcol = 0;
option reslim = 10000;
option iterlim = 10000;
option solprint = off;
option nlp = conopt;

*==============================================================================
*-----------instantiate models with corresponding equations
model busc /objfun,
        dynamics_eq,
        market_clearing_eq,
        con_sec_eq,
        kap_sec_eq,
        lab_sec_eq,
        adj_eq,
        out_eq,
        utility_eq,
        /;
model busc_tipped /objfun,
        dynamics_eq,
        tipped_market_clearing_eq,
        con_sec_eq,
        kap_sec_eq,
        lab_sec_eq,
        adj_eq,
        out_eq,
        utility_eq,
        /;

