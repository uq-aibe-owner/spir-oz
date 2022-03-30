** This is a code of SCEQ to solve a multi-country real business cycle problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include _sceq_common.gms

*********************************************
* run SCEQ

parameter 
    con_path(r, i, tt, p) simulated consumption paths
    inv_sec_path(r, j, tt, p) simulated investment paths
    lab_path(r, i, tt, p) simulated labor supply paths
    kap_path(r, i, tt, p) simulated capital paths
    lam_path(r, i, tt, p) shadow prices for capital transition
    mu_path(i, tt, p) shadow prices for budget constraint
;

con_path(r, i, tt, p) = 1;
inv_sec_path(r, j, tt, p) = 1;
lab_path(r, i, tt, p) = 1;
kap_path(r, i, tt, p) = 1;
lam_path(r, i, tt, p) = 1;
mu_path(i, tt, p) = 1;
                  
set niter / 1 * 10 /;

kap_path(r, i, '1', p) = KAP0(r, i);

*==============================================================================
*-----------solve the pre-tipping path
loop(p $ (ord(p) = 1),
*-----------iterate over time periods, tt, of interest
*-----------(last period for error checking)
  loop(tt $ (ord(tt) <= T_STAR + 1),
* starting period
    s = ord(tt);    

* optimization step

* fix the state variable at s
    kap.fx(r, i, tt) = kap_path(r, i, tt, p);
    
* if tipping event has not happened by the beginning of the current period
    probs(t) $ (s <= ord(t)) = (1 - PROB1) ** (ord(t) - s);
    E_shk(r, i, t) $ (s <= ord(t)) = ZETA2 + probs(t) * (ZETA1 - ZETA2);

    solve busc using nlp maximizing obj;

    con_path(r, i, tt, p) = con.L(r, i, tt);
    inv_sec_path(r, j, tt, p) = inv_sec.L(r, j, tt);
    lab_path(r, i, tt, p) = lab.L(r, i, tt);
    lam_path(r, i, tt, p) = dynamics_eq.m(r, i, tt);
    mu_path(i, tt, p) = market_clearing_eq.m(i, tt);

* simulation step
    kap_path(r, j, tt + 1, p) = (1 - delta) * kap_path(r, j, tt, p)
      + inv_sec_path(r, j, tt, p);
  );
);

*==============================================================================
* solve the tipped paths

loop(p $ (ord(p) > 1),
* starting period is also the period that the tipped event happens
  loop(tt $ (ord(tt) <= T_STAR + 1),
    s = ord(tt);

*-----------fix the state variable at s: the tipping event happens at s, but
*-----------the capital at s has not been impacted 
    kap.fx(r, i, tt) $ (ord(tt) = s) = kap_path(r, i, tt,'a');
    
    solve busc_tipped using nlp maximizing obj;

    con_path(r, i, tt, p) = con.L(r, i, tt);
    inv_sec_path(r, j, tt, p) = inv_sec.L(r, j, tt);
    lab_path(r, i, tt, p) = lab.L(r, i, tt);
    kap_path(r, i, tt, p) = kap.L(r, i, tt);
    lam_path(r, i, tt, p) = dynamics_eq.m(r, i, tt);
    mu_path(i, tt, p) = market_clearing_eq.m(i, tt);
  );
);
*==============================================================================

display con.L, inv.L, inv_sec.L, kap.L, lab.L, out.L, adj.L;
display inv_sec_path;

*==============================================================================
*-----------compute Euler errors at the pre-tipping path
*==============================================================================

parameters 
  integrand(r, i, tt, p)
  errs(r, i, tt)
;

* integrand(r, i, tt, p) $ (ord(tt) <= T_STAR + 1)
*   = lam_path(r, i, tt, p) * (1 - delta) + mu_path(tt, p) 
*     * (A * ALPHA
*     * ((kap_path(r, i, tt, p) / lab_path(r, i, tt, p)) ** (ALPHA - 1))
*     - PHI_ADJ / 2 * sqr(inv_path(r, i, tt, p) / kap_path(r, i, tt, p) - delta)
*     + PHI_ADJ*(inv_path(r, i, tt, p) / kap_path(r, i, tt, p) - delta)
*     * inv_path(r, i, tt,  p) / kap_path(r, i, tt, p)
*     )
* ;
integrand(r, i, tt, p) $ (ord(tt) <= T_STAR + 1) 
  = lam_path(r, i, tt, p) * (1 - delta) 
    + mu_path(i, tt, p) * (
      A * ALPHA
        * (kap_path(r, i, tt, p) / lab_path(r, i, tt, p)) ** (ALPHA - 1)
      - PHI_ADJ / 2
        * sqr(kap_path(r, i, tt + 1, p) / kap_path(r, i, tt, p) - 1)
      + PHI_ADJ
        * (kap_path(r, i, tt + 1, p) / kap_path(r, i, tt, p) - 1)
        * kap_path(r, i, tt + 1, p) / kap_path(r, i, tt, p)
    )
;
   
errs(r, i, tt) $ (ord(tt) <= T_STAR)
  = abs(1
    - BETA * (
      (1 - PROB1) * integrand(r, i, tt + 1, 'a')
      + PROB1 * sum(p $ (ord(p) = ord(tt) + 1), integrand(r, i, tt + 1, p))
    ) / lam_path(r, i, tt,'a')
  )
;
    
*==============================================================================
*-----------Export solutions to file 
*==============================================================================
*File sol_SCEQ_RBC_con /sol_SCEQ_RBC_con.csv/;
*sol_SCEQ_RBC_con.pc=5;
*sol_SCEQ_RBC_con.pw=4000;
*
*Put sol_SCEQ_RBC_con;
*
*loop(p,
*  loop(tt$(ord(tt)<=T_STAR),
*    put tt.tl::4;    
*        loop(r,
*          put con_path(r, *, tt, p)::6;
*          );
*    put /;
*  );
*);
*
*File sol_SCEQ_RBC_kap /sol_SCEQ_RBC_kap.csv/;
*sol_SCEQ_RBC_kap.pc=5;
*sol_SCEQ_RBC_kap.pw=4000;
*
*Put sol_SCEQ_RBC_kap;
*
*loop(p,
*  loop(tt$(ord(tt)<=T_STAR),
*    put tt.tl::4;
*    loop(r,
*      put kap_path(r, *, tt, p)::6;
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
*  loop(tt$(ord(tt)<=T_STAR),
*    put tt.tl::4;
*    loop(r,
*      put inv_path(r, *, tt, p)::6;
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
*  loop(tt$(ord(tt)<=T_STAR),
*    put tt.tl::4;
*    loop(r,
*      put lab_path(r, *, tt, p)::6;
*    );    
*    put /;
*  );
*);
*
*File sol_SCEQ_RBC_err /sol_SCEQ_RBC_err.csv/;
*sol_SCEQ_RBC_err.pc=5;
*sol_SCEQ_RBC_err.pw=4000;

*Put sol_SCEQ_RBC_err;
*
*loop(tt$(ord(tt)<=T_STAR),
*    put tt.tl::4;
*    loop(r,
*      put errs(r, i, tt)::6;
*    ); 
*    put /;
*);

* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

