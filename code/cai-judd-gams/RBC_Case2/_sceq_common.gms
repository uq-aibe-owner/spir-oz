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
set npath /1*10/;

*==============================================================================
*-----------basic economic parameters
*==============================================================================
parameters
    BETA                     discount rate /0.98/
    ALPH                    capital cost share /0.33/
    DELT                    capital stock depreciation /0.5/
    PHI_ADJ                      adjustment cost parameter /0.5/
    GAMM                    intertemporal elasticity of substitution /0.5/
    GAMM_HAT                 utility parameter
    ETA                      Frisch elasticity of labor supply /0.5/
    ETA_HAT                   utility parameter
    A                        technology parameter
    B                        relative weight of consumption and leisure
    s                        starting period
    DT                       number of periods for optimization in SCEQ / 10 /
    Tstar                    number of periods of interest 
    REG_WGHT(r)                 regional weights (eg population)
    EoS_KAP                     elasticity of substitution for kapital
    Imin                     lower bound of investment 
    KAP0(r)                    initial capital
    kmin                     smallest capital    / 0.1 /
    kmax                     largest capital     / 10 /
    ZETA1                    TFP before shock   / 1 /
    ZETA2                    TFP after shock   / 0.95 /
    prob1                    one period probability of jump of TFP / 0.01 /
    probs(t)                 probability of jump of TFP
    E_shk                   expected shock (exogenous) 
    TL_CON_SHR                      tail consumption share (of output) / .45 /
    CON_SHR(i)             consumption share for each sector
;

*==============================================================================
*-----------derived economic parameters
*==============================================================================
Tstar = card(npath)-1;
A = (1 - (1 - DELT) * BETA) / (ALPH * BETA);
GAMM_HAT = 1 - (1 / GAMM);
B = (1 - ALPH) * A * (A - DELT) ** (-1 / GAMM);
ETA_HAT = 1 + (1 / ETA);
RHO = (EoS_KAP - 1) / EoS_KAP 
RHO_INV = 1 / RHO;

*-----------the following is the vector of population weights: 
*-----------it enters the objective function and determines demand
REG_WGHT(r) = 1;
*-----------
Imin = 0.9 * DELT;
*-----------the following is initial kapital: it will vary across regions
*KAP0(r) = kmin + (kmax-kmin)*(ord(r)-1)/(card(r)-1);
*KAP0(r) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(r)-1)/(card(r)-1));
KAP0(r, i) = 1

E_shk(r, i, t) = ZETA2 + Probs(t) * (ZETA1 - ZETA2)

display A, B, ETA_HAT, KAP0;
*-----------the following won't work as s is a parameter not a set
*interval(s) = yes$(s < ord(t) and ord(t) < DT + s)

*==============================================================================
*-----------basic (economic) variables
*==============================================================================
Variables
    obj                        objective criterion
    inv(r, i, t)              investment
;
*------------------------------------------------------------------------------
Positive variables

    con(r, i, t)              consumption
    kap(r, i, t)              kapital stock
    lab(r, i, t)              labor supply
*-----------intermediate variables
    con_sec(r, t)           consumption aggregate (across sectors)
    kap_sec(r, t)           kapital aggregate (across sectors)
    lab_sec(r, t)           labour aggregate (across sectors)
    utility(r, t)                      instantaneous utility
;

*==============================================================================
*-----------equation declarations (over entire sets)
*==============================================================================
*-----------declarations are applied globally (eg on the whole of 1..71)
Equations
*-----------declarations for intermediate-variables
    con_sec_eq(r, t)           consumption aggregate (across sectors)
    kap_sec_eq(r, t)           kapital aggregate (across sectors)
    lab_sec_eq(r, t)           labour aggregate (across sectors)
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
If ((s <= ord(t) <= DT + s),
*-----------definitions for intermediate-variables
  con_sec_eq(r, t).. 
    con_sec(r, t) =e= prod(i, con(r, i, t) ** CON_SHR(i))
  ;
  lab_sec_eq(r, t).. 
    lab_sec(r, t) =e= prod(i, lab(r, i, t) ** LAB_SHR(i))
  ;
  kap_sec_eq(r, t).. 
    kap_sec(r, t) =e= (sum(i, KAP_SHR(i) * kap(r, i, t) ** RHO)) ** RHO_INV
  ;
  out_eq(r, i, t).. 
    out(r, i, t) =e= A * kap(r, i, t) ** ALPH * lab(r, i, t) ** (1 - ALPH)
  ;
  adj_eq(r, i, t)..
    adj(r, i, t) =e= (PHI_ADJ/2) * k(r, i, t) 
      * sqr(inv(r, i, t) / k(r,t) - DELT)
  ;
*------------------------------------------------------------------------------
*-----------the sequence of utility flows per region and time period
*------------------------------------------------------------------------------
  utility_eq(r, t)..
    If ((ord(t) < DT + s),
      utility(r, t) =e=
*-----------the consumption part:
        con_sec(r, t) ** GAMM_HAT / GAMM_HAT
*-----------the labour part:
        - B * lab_sec(r, t) ** ETA_HAT / ETA_HAT
    ;
*-----------tail/continuation utility, where tail labour is normalised to one:
    else (ord(t) = DT + s),
      utility(r, t) =e=
*-----------in the consumption part, a fixed share of output is consumed
        ((TL_CON_SH * A * kap_sec(r, t)) ** ALPH) ** GAMM_HAT / GAMM_HAT 
        - B / (1 - BETA)
    ;
    );
  ;
);
*------------------------------------------------------------------------------
*-----------the objective function
*------------------------------------------------------------------------------
  obj_eq.. 
    obj =e= 
        sum(r, REG_WGHT(r) * sum(t, BETA ** (ord(t) - s) * utility(r, t)))
  ;
*------------------------------------------------------------------------------
*-----------canonical equations
*------------------------------------------------------------------------------
  dynamics_eq(r, t) $ (ord(t) < s + DT)..
    kap(r, i, t + 1) =e= (1 - DELT) * kap(r, i, t) + inv(r, i, t)
  ;
  market_clearing_eq(t) $ (ord(t) < s + DT)..
    sum(r, E_shk(r, i, t) * out(r, i, t)
        - c(r, i, t) - inv(r, i, t) - adj(r, i, t)) =e= 0
  ;
*------------------------------------------------------------------------------
*-----------other states
  tipped_market_clearing_eq(t) $ (ord(t) < s + DT)..
    sum(r, ZETA2 * out(r, i, t) 
        - c(r, i, t) - inv(r, i, t) - adj(r, i, t)) =e= 0
  ;
);

*==============================================================================
*-----------Bound Constraints
con.lo(r,t) = 0.001;
con.up(r,t) = 1000;
kap.lo(r,t) = 0.001;
kap.up(r,t) = 1000;
lab.lo(r,t) = 0.001;
lab.up(r,t) = 1000;
inv.lo(r,t) = Imin;

*==============================================================================
*-----------Initial Guess
s = 1;
con.l(r,t) = A - DELT;
inv.l(r,t) = DELT;
kap.l(r,t) = 1;
lab.l(r,t) = 1;
obj.l = sum(r, REG_WGHT(r) * sum(t$ (s <= ord(t) and ord(t) < s+DT),
    BETA**(ord(t)-s)*((c.l(r,t)**GAMM_HAT)/GAMM_HAT - B * (l.l(r,t)**ETA_HAT)/ETA_HAT))) +
  sum(r, REG_WGHT(r) * sum(t$(ord(t)=s+DT), BETA**(ord(t)-s)*((((0.75*A*(k.l(r,t)**ALPH))**GAMM_HAT)/GAMM_HAT-B)/(1-BETA))));

*==============================================================================
*-----------solver options
options limrow = 0, limcol = 0;
option reslim = 10000;
option iterlim = 10000;
option solprint = off;
option nlp = conopt;

*==============================================================================
*-----------instantiate models with corresponding equations
model busc /objfun, dynamics_eq, market_clearing_eq/;
model busc_tipped /objfun, dynamics_eq, tipped_market_clearing_eq/;

