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
  loop(tt $ (ord(tt) <= Tstar + 1),
* starting period
    s = ord(tt);    

* optimization step

* fix the state variable at s
    kap.fx(r, i, tt) = kap_path(r, i, tt, p);
    
* if tipping event has not happened by the beginning of the current period
    probs(t) $ (s <= ord(t)) = (1 - PROB1) ** (ord(t) - s);
    E_shk(r, i, t) $ (s <= ord(t)) = ZETA2 + probs(t) * (ZETA1 - ZETA2);
    loop(niter,
        solve busc using nlp maximizing obj;
        if((busc.MODELSTAT <= 2 and busc.SOLVESTAT = 1),
            break;
        );
    );
    abort$(busc.MODELSTAT > 2 or busc.SOLVESTAT <> 1) "FAILED in solving!";

    con_path(r, i, tt, p) = con.L(r, i, tt);
    inv_sec_path(r, j, tt, p) = inv_sec.L(r, j, tt);
    lab_path(r, i, tt, p) = lab.L(r, i, tt);
    lam_path(r, i, tt, p) = dynamics_eq.m(r, i, tt);
    mu_path(i, tt, p) = market_clearing_eq.m(i, tt);

* simulation step
    kap_path(r, j, tt + 1, p) = (1 - delta) * kap_path(r, j, tt, p)
      + inv_sec_path(r, j, tt, p);
  );
  
* relax the fixed constraints on the state variables
  kap.lo(r, i, t) = 0.001;
  kap.up(r, i, t) = 1000;  
);

*==============================================================================
* solve the tipped paths

loop(p $ (ord(p) > 1),
* starting period is also the period that the tipped event happens
    s = ord(p);

*-----------fix the state variable at s: the tipping event happens at s, but
*-----------the capital at s has not been impacted 
    kap.fx(r, i, tt) $ (ord(tt) = s) = kap_path(r, i, tt,'1');
    
    loop(niter,
        solve busc_tipped using nlp maximizing obj;
        if((busc_tipped.MODELSTAT <= 2 and busc_tipped.SOLVESTAT = 1),
            break;
        );
    );
    abort$(busc_tipped.MODELSTAT > 2 or busc_tipped.SOLVESTAT <> 1)
      "FAILED in solving!"
    ;
    con_path(r, i, tt, p) = con.L(r, i, tt);
    inv_sec_path(r, j, tt, p) = inv_sec.L(r, j, tt);
    lab_path(r, i, tt, p) = lab.L(r, i, tt);
    kap_path(r, i, tt, p) = kap.L(r, i, tt);
    lam_path(r, i, tt, p) = dynamics_eq.m(r, i, tt);
    mu_path(i, tt, p) = market_clearing_eq.m(i, tt);
    
* relax the fixed constraints on the state variables
    kap.lo(r, i, t) = 0.001;
    kap.up(r, i, t) = 1000;  
);
*==============================================================================
display con.L, inv.L, inv_sec.L, kap.L, lab.L, out.L, adj.L;

*==============================================================================
* compute Euler errors at the pre-tipping path

*parameter integrand(r, i, tt, p)
*    errs(r, i, tt);
*
*integrand(r, i, tt,'1')$(ord(tt)<=Tstar+1) = lam_path(r, i, tt,'1')*(1-delta) + 
*    mu_path(tt,'1') * ( A*alpha*((kap_path(r, i, tt,'1')/lab_path(r, i, tt,'1'))**(alpha-1)) -
*    phi/2*sqr(inv_path(r, i, tt,'1')/kap_path(r, i, tt,'1')-delta) +
*    phi*(inv_path(r, i, tt,'1')/kap_path(r, i, tt,'1')-delta)*inv_path(r, i, tt,'1')/kap_path(r, i, tt,'1') );
*integrand(r, i, tt, p)$(ord(p)>1 and ord(tt)<=Tstar+1) = lam_path(r, i, tt, p)*(1-delta) + 
*    mu_path(tt, p) * ( A*alpha*((kap_path(r, i, tt, p)/lab_path(r, i, tt, p))**(alpha-1)) -
*    phi/2*sqr(inv_path(r, i, j, tt, p)/kap_path(r, i, tt, p)-delta) +
*    phi*(inv_path(r, i, j, tt, p)/kap_path(r, i, tt, p)-delta)*inv_path(r, i, j, tt, p)/kap_path(r, i, tt, p) );
*   
*errs(r, i, tt)$(ord(tt)<=Tstar) = abs(1 - beta*( (1-PROB1)*integrand(r, i, tt+1,'1') +
*    PROB1*sum(p$(ord(p)=ord(tt)+1),integrand(r, i, tt+1, p)) ) / lam_path(r, i, tt,'1'));
    
************************
* Output solutions 

*File sol_SCEQ_RBC_con /sol_SCEQ_RBC_con.csv/;
*sol_SCEQ_RBC_con.pc=5;
*sol_SCEQ_RBC_con.pw=4000;
*
*Put sol_SCEQ_RBC_con;
*
*loop(p,
*  loop(tt$(ord(tt)<=Tstar),
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
*  loop(tt$(ord(tt)<=Tstar),
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
*  loop(tt$(ord(tt)<=Tstar),
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
*  loop(tt$(ord(tt)<=Tstar),
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
*loop(tt$(ord(tt)<=Tstar),
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

