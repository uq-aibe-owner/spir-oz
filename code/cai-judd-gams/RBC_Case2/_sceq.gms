*==============================================================================
*-----------This is NAIM (the New Australian Inter-* Model). Where * stands for
*-----------regional, sectoral, generational and state-of-the-world. It is 
*-----------an adaptation of Cai--Judd (2021) to include multiple sectors among
*-----------among other things.
*==============================================================================
*-----------"The model" was originally in a separate file and included using
*-----------"$include". But this makes it difficult to debug errors (in GAMS) as
*-----------line numbers are aggregated across the two files.
*$include _sceq_common.gms
*==============================================================================
*-----------"THE MODEL" starts here
*==============================================================================
scalar starttime;
starttime = jnow;
*==============================================================================
*-----------sets
*==============================================================================
set r regions /Aus, Qld/;
alias(rr, r)
;
set j sectors /1*2/;
alias(i, j)
alias(ii, j)
alias(jj, j)
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
    LFWD                     look-forward parameter / 50/
    s                        starting period for each look forward
    T_STAR                    number of periods of interest 
    REG_WGHT(r)              regional weights (eg population)
    EoS_KAP                  elasticity of substitution for kapital /0.4/
    RHO                      exponent of the ces function
    RHO_INV                  inverse of RHO
    CON_SHR(i)               share of each commodity in consumption
    LAB_SHR(i)               share of each type of labour in utility
    INV_SHR(i, j)               share of each commodity in saving
    INV_MIN                  lower bound of investment 
    KAP0(r, i)               initial capital
    kmin                     smallest capital    / 0.1 /
    kmax                     largest capital     / 10 /
    ZETA1                    TFP before shock   / 1 /
    ZETA2                    TFP after shock   /95e-2 /
    PROB1                    one period probability of jump of TFP / 0.01 /
    TL_CON_SHR               tail consumption share (of output) / 0.75 /
    CON_SHR(i)               consumption share for each sector
    T_ALL                    for the set of all time periods
    num                      temporary allocation
    temp_sum_1               temporary allocation / 0/
    temp_sum_2                 temporary allocation / 0/
;
*==============================================================================
*-----------derived economic parameters
*==============================================================================
T_STAR = card(p) - 1;
A = (1 - (1 - DELTA) * BETA) / (ALPHA * BETA);
GAMMA_HAT = 1 - (1 / GAMMA);
B = (1 - ALPHA) * A * (A - DELTA) ** (-1 / GAMMA);
ETA_HAT = 1 + (1 / ETA);
RHO = (EoS_KAP - 1) / EoS_KAP; 
RHO_INV = 1 / RHO;

*-----------set the seed for the random number generator that we use in weights
execseed = 12345
*-----------vector of population weights: 
*-----------it enters the objective function and determines demand
loop(r,
  REG_WGHT(r) = 1 / ord(r);
);
loop(i,
  num = uniform(3e-1, 7e-1);
  temp_sum_1 = temp_sum_1 + num;
  CON_SHR(i) = num;
  num = uniform(3e-1, 7e-1);
  temp_sum_2 = temp_sum_2 + num;
  LAB_SHR(i) = num;
*  INV_SHR(i, j) = ord(i) / ((card(i) * (card(i) + 1)) / 2);
);
CON_SHR(i) = CON_SHR(i) / temp_sum_1;
LAB_SHR(i) = LAB_SHR(i) / temp_sum_2;
*-----------alternative, less symmetric parametrisation for INV_SHR
loop(j,
  temp_sum_1 = 0;
  loop(i,
    num = uniform(4e-1, 6e-1);
    temp_sum_1 = temp_sum_1 + num;
    INV_SHR(i, j) = num;
  ;
  );
  INV_SHR(i, j) = INV_SHR(i, j) / temp_sum_1;
);
display CON_SHR LAB_SHR INV_SHR;
*-----------the following is initial kapital: it will vary across regions
*KAP0(r) = kmin + (kmax-kmin)*(ord(r)-1)/(card(r)-1);
*KAP0(r) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(r)-1)/(card(r)-1));

display A, B, ETA_HAT;
*-----------the following won't work as s is a parameter not a set
*interval(s) = yes$(s < ord(t) and ord(t) < LFWD + s)

*==============================================================================
*-----------derived set for timeline that transcends LFWD and T_STAR
*==============================================================================
set t time / 1 * 71 /;
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
    utility(r, t)             instantaneous utility
;
*------------------------------------------------------------------------------
Positive variables

    inv(r, i, j, t)           investment
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
kap.up(r, i, t) = 1e+2;
con.up(r, i, t) = 1e+2;
lab.up(r, i, t) = 1e+0;
con.lo(r, i, t) = 1e-2;
kap.lo(r, i, t) = 1e-2;
inv.lo(r, i, j, t) = 1e-2;
lab.lo(r, i, t) = 1e-2;
utility.up(r, t) = 1e+3;
utility.lo(r, t) = -1e+3;
*============================================================================== 
*-----------Initial Guess
*============================================================================== 
s = 1;
KAP0(r, i) = 1e+0;
kap.L(r, i, t) = KAP0(r, i);
inv.L(r, i, j, t) = DELTA * KAP0(r, i);
lab.L(r, i, t) = 1e+0;
con.L(r, i, t) = 1e+0;
con_sec.L(r, t) = 
  prod(i, con.L(r, i, t) ** CON_SHR(i))
*  sum(i, CON_SHR(i) * con.L(r, i, t) ** RHO)
;
lab_sec.L(r, t) = sum(i, LAB_SHR(i) * lab.L(r, i, t) ** RHO);
inv_sec.L(r, j, t) = sum(i, INV_SHR(i, j) * inv.L(r, i, j, t) ** RHO);
adj.L(r, i, t) = 0;
out.L(r, i, t) = A * kap.L(r, i, t) ** ALPHA * lab.L(r, i, t) ** (1 - ALPHA);
*==============================================================================
*-----------equation declarations (over entire sets)
*==============================================================================
*-----------declarations are applied globally (eg on the whole of 1..71)
Equations
*-----------declarations for intermediate-variables
    con_sec_eq(r, t)           consumption aggregate (across sectors)
    inv_sec_eq(r, j, t)        kapital aggregate (across sectors)
    lab_sec_eq(r, t)           labour aggregate (across sectors)
    adj_eq(r, i, t)            kapital adjustment costs
    out_eq(r, i, t)            output 
    utility_eq(r, t)           the utility function
    tail_utility_eq(r, t)      the tail utility function
    obj_eq                     Objective function
*-----------canonical equations
    dynamics_eq(r, i, t)                   Law of Motion for Capital Stock
    market_clearing_eq(i, t)               budget constraint before jump
    jacobi_identities(r, i, j, t, ii)  optimal investment distribution
*-----------other states
    tipped_market_clearing_eq(i, t)      budget constraint after jump
;
*==============================================================================
*-----------equation definitions for each s along the path
*==============================================================================
*-----------definitions for some intermediate-variables
con_sec_eq(r, t) $ (s <= ord(t) and ord(t) < LFWD + s)..
  con_sec(r, t) =e= sum(i, con(r, i, t) ** CON_SHR(i))
*  con_sec(r, t) =e= prod(i, con(r, i, t) ** CON_SHR(i))
*  con_sec(r, t) =e= sum(i, CON_SHR(i) * con(r, i, t) ** RHO)
;
lab_sec_eq(r, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  lab_sec(r, t) =e= sum(i, lab(r, i, t) ** 2)
*  lab_sec(r, t) =e= prod(i, lab(r, i, t) ** LAB_SHR(i))
;
inv_sec_eq(r, j, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  inv_sec(r, j, t) =e= prod(i, inv(r, i, j, t) ** INV_SHR(i, j))
*  inv_sec(r, j, t) =e= sum(i, INV_SHR(i, j) * inv(r, i, j, t) ** RHO)
;
adj_eq(r, i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  adj(r, i, t)
    =e= (PHI_ADJ) * kap(r, i, t) 
      * sqr(kap(r, i, t + 1) / kap(r, i, t) - 1)
;
out_eq(r, i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  out(r, i, t) =e= kap(r, i, t) ** ALPHA * lab(r, i, t) ** (1 - ALPHA)
;
jacobi_identities(r, i, j, t, ii) $ (
  (s <= ord(t) and ord(t) < LFWD + s)
  and (1 < ord(j) and ord(i) <> ord(j))
  )..
    inv(r, i, j, t)
      =e= (inv(r, i, ii, t) / INV_SHR(i, ii))
        / (inv(r, ii, ii, t) / INV_SHR(ii, ii))
        * (inv(r, ii, j, t) / INV_SHR(ii, j))
        * INV_SHR(i, j)
;
*------------------------------------------------------------------------------
*-----------canonical equations
*------------------------------------------------------------------------------
dynamics_eq(r, i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  kap(r, i, t + 1) =e= (1 - DELTA) * kap(r, i, t) + inv_sec(r, i, t)
;
market_clearing_eq(i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  0 =e= sum(r,
    E_shk(r, i, t) * out(r, i, t)
    - con(r, i, t)
    - adj(r, i, t)
    - sum(j, inv(r, i, j, t))
  )
;
*------------------------------------------------------------------------------
*-----------other states
*------------------------------------------------------------------------------
tipped_market_clearing_eq(i, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  0 =e= sum(r,
    ZETA2 * out(r, i, t) 
    - con(r, i, t)
    - adj(r, i, t)
    - sum(j, inv(r, i, j, t))
  )
;
*------------------------------------------------------------------------------
*-----------the sequence of utility flows per region and time period
*------------------------------------------------------------------------------
utility_eq(r, t) $ (s <= ord(t) and ord(t) < LFWD + s).. 
  utility(r, t) =e= 1e-4
*-----------the consumption part:
    + con_sec(r, t)
*    + con_sec(r, t) ** GAMMA_HAT / GAMMA_HAT
*-----------the labour part:
*    - 1 * lab_sec(r, t) ** RHO_INV
*    - B * lab_sec(r, t) ** ETA_HAT / ETA_HAT
    - lab_sec(r, t)
;
tail_utility_eq(r, t) $ (ord(t) = LFWD + s)..
*-----------tail/continuation utility, where tail labour is normalised to one:
  utility(r, t) =e= 1e-4
*-----------in the consumption part, a fixed share of output is consumed
    + prod(i, TL_CON_SHR * kap(r, i, t) ** ALPHA) / (1 - BETA)
*    + sum(i, 
*        CON_SHR(i) 
*          * (TL_CON_SHR * (A * kap(r, i, t)) ** ALPHA) ** RHO)
*-----------and the labour part:
    - B / (1 - BETA)
;
*------------------------------------------------------------------------------
*-----------the objective function
*------------------------------------------------------------------------------
obj_eq..
  obj =e= 
    sum(r, REG_WGHT(r) * sum(t $ (s <= ord(t) and ord(t) <= LFWD + s),
      BETA ** (ord(t) - s) * utility(r, t)))
;
*==============================================================================
*-----------solver options
options limrow = 0, limcol = 0;
option reslim = 1e+4;
option iterlim = 1e+4;
option solprint = off;
*-----------which solver to use, comment out one of the following:
*option nlp = conopt;
*option nlp = bonminh;
*linear_solver = pardisomkl;
option nlp = conopt;
*option nlp = knitro;
*option nlp = pardiso;
*option nlp = ipopth;
*option nlp = minos;
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

*==============================================================================
*-----------"THE MODEL" ends here.
*==============================================================================
*==============================================================================
*-----------The algorithm for solving the model along paths, for storing
*-----------its results, and for generating the Euler errors begins here.
*==============================================================================

parameter 
    path_con(r, i, tt, p) simulated consumption paths
    path_inv(r, i, j, tt, p) simulated investment paths
    path_inv_sec(r, j, tt, p) simulated aggregate investment paths
    path_lab(r, i, tt, p) simulated labor supply paths
    path_kap(r, i, tt, p) simulated capital paths
    path_lam(r, i, tt, p) shadow prices for capital transition
    path_mu(i, tt, p) shadow prices for budget constraint
;

path_con(r, i, tt, p) $ (ord(tt) <= T_STAR + 1) = 1;
path_inv(r, i, j, tt, p) $ (ord(tt) <= T_STAR + 1) = 1;
path_inv_sec(r, j, tt, p) $ (ord(tt) <= T_STAR + 1) = 1;
path_lab(r, i, tt, p) $ (ord(tt) <= T_STAR + 1) = 1;
path_kap(r, i, tt, p) $ (ord(tt) <= T_STAR + 1) = 1;
path_lam(r, i, tt, p) $ (ord(tt) <= T_STAR + 1) = 1;
path_mu(i, tt, p) $ (ord(tt) <= T_STAR + 1) = 1;
                  
set niter / 1 * 10 /;

path_kap(r, i, '1', p) = KAP0(r, i);

*==============================================================================
*-----------Loop for solving the model along the no-tipping path (1st path)
*==============================================================================
loop(p $ (ord(p) = 1),
*-----------iterate over time periods along the path upto T_STAR
*-----------(last period for error checking)
  loop(tt $ (ord(p) <= ord(tt) and ord(tt) <= card(p)),
*-----------starting period
    s = ord(tt);    
*------------------------------------------------------------------------------
*-----------On the path with no tipping, we face uncertainty.
*-----------The probability of tipping for each future time:
    probs(t) $ (s <= ord(t) and ord(t) <= LFWD + s) 
      = (1 - PROB1) ** (ord(t) - s);
*-----------Expected productivity is:
    E_shk(r, i, t) $ (s <= ord(t) and ord(t) <= LFWD + s)
      = ZETA2 + probs(t) * (ZETA1 - ZETA2);
*-----------fix the state variable at s
*------------------------------------------------------------------------------
    kap.fx(r, i, tt) = path_kap(r, i, tt, p);
    
*-----------solve the model:
    solve busc using nlp maximizing obj;
*-----------stop if any run is terminating abnormally:
*    if((busc.MODELSTAT > 2 or busc.SOlVESTAT > 1),
*    abort "Abnormal termination: check MODELSTAT or SOLVESTAT!"
*    );
*-----------save results to the path_ parameters:
    path_con(r, i, tt, p) = con.L(r, i, tt);
    path_inv(r, i, j, tt, p) = inv.L(r, i, j, tt);
    path_inv_sec(r, j, tt, p) = inv_sec.L(r, j, tt);
    path_lab(r, i, tt, p) = lab.L(r, i, tt);
    path_lam(r, i, tt, p) = dynamics_eq.m(r, i, tt);
    path_mu(i, tt, p) = market_clearing_eq.m(i, tt);

* simulation step
    path_kap(r, j, tt + 1, p) = (1 - delta) * path_kap(r, j, tt, p)
      + path_inv_sec(r, j, tt, p);
  );
);

*==============================================================================
* solve the tipped paths

loop(p $ (ord(p) > 1),
* starting period is also the period that the tipped event happens
  loop(tt $ (ord(p) <= ord(tt) and ord(tt) <= card(p)),
    s = ord(tt);
*-----------fix the state variable at s: the tipping event happens at s, but
*-----------the capital at s has not been impacted 
    kap.fx(r, i, tt) $ (ord(tt) = s) = path_kap(r, i, tt,'a');
    
    solve busc_tipped using nlp maximizing obj;

    path_con(r, i, tt, p) = con.L(r, i, tt);
    path_inv(r, i, j, tt, p) = inv.L(r, i, j, tt);
    path_inv_sec(r, j, tt, p) = inv_sec.L(r, j, tt);
    path_lab(r, i, tt, p) = lab.L(r, i, tt);
    path_kap(r, i, tt, p) = kap.L(r, i, tt);
    path_lam(r, i, tt, p) = dynamics_eq.m(r, i, tt);
    path_mu(i, tt, p) = tipped_market_clearing_eq.m(i, tt);
  );
*  loop(tt $ (ord(tt) < ord(p)),
*    path_con(r, i, tt, p) = path_con(r, i, tt, 'a');
*    path_inv(r, i, j, tt, p) = path_inv(r, i, j, tt, 'a');
*    path_inv_sec(r, i, tt, p) = path_inv_sec(r, i, tt, 'a');
*    path_lab(r, i, tt, p) = path_lab(r, i, tt, 'a');
*    path_kap(r, i, tt, p) = path_kap(r, i, tt, 'a');
*    path_lam(r, i, tt, p) = path_lam(r, i, tt, 'a');
*    path_mu(i, tt, p) = path_mu(i, tt, 'a');
*  );
);
*==============================================================================

*display con.L, inv.L, inv_sec.L, kap.L, lab.L, out.L, adj.L;
*display path_inv_sec;

parameter
jac_id(r, i, j, ii, jj, t, p)  check on Jacobi identity
;
jac_id(r, i, j, ii, jj, t, p)
  $ (ord(p) <= ord(t) and ord(t) <= card(p))
    = path_inv(r, i, j, t, p) / INV_SHR(i, j)
      - (path_inv(r, i, ii, t, p) / INV_SHR(i, ii))
        / (path_inv(r, jj, ii, t, p) / INV_SHR(jj, ii))
        * (path_inv(r, jj, j, t, p) / INV_SHR(jj, j))
;

parameter max_jac_id maximum of jac_id;

max_jac_id = 
  smax((r, i, j, ii, jj, t, p), 
    abs(jac_id(r, i, j, ii, jj, t, p)) 
      $ (ord(p) <= ord(t) and ord(t) <= card(p))
  )
;
display max_jac_id;
*==============================================================================
*-----------compute Euler errors at the pre-tipping path
*==============================================================================

parameters 
  integrand(r, i, tt, p)
  errs(r, i, tt)
;

* integrand(r, i, tt, p) $ (ord(tt) <= T_STAR + 1)
*   = path_lam(r, i, tt, p) * (1 - delta) + path_mu(tt, p) 
*     * (A * ALPHA
*     * ((path_kap(r, i, tt, p) / path_lab(r, i, tt, p)) ** (ALPHA - 1))
*     - PHI_ADJ / 2 * sqr(path_inv(r, i, tt, p) / path_kap(r, i, tt, p) - delta)
*     + PHI_ADJ*(path_inv(r, i, tt, p) / path_kap(r, i, tt, p) - delta)
*     * path_inv(r, i, tt,  p) / path_kap(r, i, tt, p)
*     )
* ;
integrand(r, i, tt, p) $ (ord(tt) <= T_STAR + 1) 
  = path_lam(r, i, tt, p) * (1 - delta) 
    + path_mu(i, tt, p) * (
      A * ALPHA
        * (path_kap(r, i, tt, p) / path_lab(r, i, tt, p)) ** (ALPHA - 1)
      - PHI_ADJ / 2
        * sqr(path_kap(r, i, tt + 1, p) / path_kap(r, i, tt, p) - 1)
      + PHI_ADJ
        * (path_kap(r, i, tt + 1, p) / path_kap(r, i, tt, p) - 1)
        * path_kap(r, i, tt + 1, p) / path_kap(r, i, tt, p)
    )
;
   
errs(r, i, tt) $ (ord(tt) <= T_STAR)
  = abs(1
    - BETA * (
      (1 - PROB1) * integrand(r, i, tt + 1, 'a')
      + PROB1 * sum(p $ (ord(p) = ord(tt) + 1), integrand(r, i, tt + 1, p))
    ) / path_lam(r, i, tt,'a')
  )
;
    
*==============================================================================
*-----------Export solutions to file 
*==============================================================================
option savepoint=2

$gdxOut errs


*File sol_SCEQ_RBC_con /sol_SCEQ_RBC_con.csv/;
*sol_SCEQ_RBC_con.pc=5;
*sol_SCEQ_RBC_con.pw=4000;

*Put sol_SCEQ_RBC_con;

*loop(p,
*  loop(tt $ (ord(tt) <= Tstar),
*    put tt.tl::4;    
*    loop(r,
*      loop(i,
*        put path_con(r, i, tt, p)::6;
*      );
*    );
*    put /;
*  );
*);

*File sol_SCEQ_RBC_kap /sol_SCEQ_RBC_kap.csv/;
*sol_SCEQ_RBC_kap.pc=5;
*sol_SCEQ_RBC_kap.pw=4000;

*Put sol_SCEQ_RBC_kap;

*loop(p,
*  loop(tt$(ord(tt)<=Tstar),
*    put tt.tl::4;
*    loop(r,
*      loop(i,
*        put path_kap(r, i, tt, p)::6;
*      ) ;
*    );    
*    put /;
*  );
*);	
*
*File sol_SCEQ_RBC_inv /sol_SCEQ_RBC_inv.csv/;
*sol_SCEQ_RBC_inv.pc=5;
*sol_SCEQ_RBC_inv.pw=4000;
*
*Put sol_SCEQ_RBC_inv;
*
*loop(p,
*  loop(tt$(ord(tt)<=Tstar),
*    put tt.tl::4;
*    loop(r,
*      loop(i,
*        put path_inv(r, i, tt, p)::6;
*      );
*    );
*    put /;
*  );
*);
*
*File sol_SCEQ_RBC_lab /sol_SCEQ_RBC_lab.csv/;
*sol_SCEQ_RBC_lab.pc=5;
*sol_SCEQ_RBC_lab.pw=4000;
*
*Put sol_SCEQ_RBC_lab;
*
*loop(p,
*  loop(tt$(ord(tt)<=Tstar),
*    put tt.tl::4;
*    loop(r,
*      loop(i,
*        put path_lab(r, i, tt, p)::6;
*      );
*    );    
*    put /;
*  );
*);

*File sol_SCEQ_RBC_err /sol_SCEQ_RBC_err.csv/;
*sol_SCEQ_RBC_err.pc=5;
*sol_SCEQ_RBC_err.pw=4000;
*
*Put sol_SCEQ_RBC_err;
*
*loop(tt$(ord(tt)<=Tstar),
*    put tt.tl::4;
*    loop(r,
*      loop(i,
*        put errs(r, i, tt)::6;
*      );
*    ); 
*    put /;
*);
*
* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;
