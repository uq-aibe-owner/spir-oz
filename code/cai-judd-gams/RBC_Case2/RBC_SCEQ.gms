** This is a code of SCEQ to solve a multi-country real business cycle problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include RBC_SCEQ_common.gms

*********************************************
* run SCEQ

parameter 
    Cpath(r,tt,npath) simulated consumption paths
    Ipath(r,tt,npath) simulated investment paths
    Lpath(r,tt,npath) simulated labor supply paths
    Kpath(r,tt,npath) simulated capital paths
    lampath(r,tt,npath) shadow prices for capital transition
    mupath(tt,npath) shadow prices for budget constraint
;

Cpath(r,tt,npath) = 1;
Ipath(r,tt,npath) = 1;
Lpath(r,tt,npath) = 1;
Kpath(r,tt,npath) = 1;
lampath(r,tt,npath) = 1;
mupath(tt,npath) = 1;
    
set niter / 1*10 /;

Kpath(r,'1',npath) = K0(r);

************************
* solve the pre-tipping path

loop(npath$(ord(npath)=1),
* iterate over periods of interest (the last extra period is for error checking only)
  loop(tt$(ord(tt)<=Tstar+1),
* starting period
    s = ord(tt);    

* optimization step

* fix the state variable at s
    K.fx(r,tt) = Kpath(r,tt,npath);
    
* if tipping event has not happened by the beginning of the current period
    Probs(t)$(ord(t)>=s) = (1-prob1)**(ord(t)-s);
        
    loop(niter,
        solve busc using nlp maximizing obj;
        if((busc.MODELSTAT<=2 and busc.SOLVESTAT=1),
            break;
        );
    );
    abort$(busc.MODELSTAT>2 or busc.SOLVESTAT<>1) "FAILED in solving!";

    Cpath(r,tt,npath) = C.l(r,tt);
    Ipath(r,tt,npath) = Inv.l(r,tt);
    Lpath(r,tt,npath) = L.l(r,tt);
    lampath(r,tt,npath) = TransitionCapital.m(r,tt);
    mupath(tt,npath) = BudgetConstraint.m(tt);

* simulation step    
    Kpath(r,tt+1,npath) = (1-delta)*Kpath(r,tt,npath) + Ipath(r,tt,npath);
  );
  
* relax the fixed constraints on the state variables
  K.lo(r,t) = 0.001;
  K.up(r,t) = 1000;  
);


************************
* solve the tipped paths

loop(npath$(ord(npath)>1),
* starting period is also the period that the tipped event happens
    s = ord(npath);

* fix the state variable at s: the tipping event happens at s but the capital at s has not been impacted 
    K.fx(r,tt)$(ord(tt)=s) = Kpath(r,tt,'1');
    
    loop(niter,
        solve busc_tipped using nlp maximizing obj;
        if((busc_tipped.MODELSTAT<=2 and busc_tipped.SOLVESTAT=1),
            break;
        );
    );
    abort$(busc_tipped.MODELSTAT>2 or busc_tipped.SOLVESTAT<>1) "FAILED in solving!";

    Cpath(r,tt,npath) = C.l(r,tt);
    Ipath(r,tt,npath) = Inv.l(r,tt);
    Lpath(r,tt,npath) = L.l(r,tt);
    Kpath(r,tt,npath) = K.l(r,tt);
    lampath(r,tt,npath) = TransitionCapital.m(r,tt);
    mupath(tt,npath) = BudgetConstraint.m(tt);
    
* relax the fixed constraints on the state variables
    K.lo(r,t) = 0.001;
    K.up(r,t) = 1000;  
);

************************
* compute Euler errors at the pre-tipping path

parameter integrand(r,tt,npath)
    errs(r,tt);

integrand(r,tt,'1')$(ord(tt)<=Tstar+1) = lampath(r,tt,'1')*(1-delta) + 
    mupath(tt,'1') * ( A*alpha*((Kpath(r,tt,'1')/Lpath(r,tt,'1'))**(alpha-1)) -
    phi/2*sqr(Ipath(r,tt,'1')/Kpath(r,tt,'1')-delta) +
    phi*(Ipath(r,tt,'1')/Kpath(r,tt,'1')-delta)*Ipath(r,tt,'1')/Kpath(r,tt,'1') );
integrand(r,tt,npath)$(ord(npath)>1 and ord(tt)<=Tstar+1) = lampath(r,tt,npath)*(1-delta) + 
    mupath(tt,npath) * ( A*alpha*((Kpath(r,tt,npath)/Lpath(r,tt,npath))**(alpha-1)) -
    phi/2*sqr(Ipath(r,tt,npath)/Kpath(r,tt,npath)-delta) +
    phi*(Ipath(r,tt,npath)/Kpath(r,tt,npath)-delta)*Ipath(r,tt,npath)/Kpath(r,tt,npath) );
   
errs(r,tt)$(ord(tt)<=Tstar) = abs(1 - beta*( (1-prob1)*integrand(r,tt+1,'1') +
    prob1*sum(npath$(ord(npath)=ord(tt)+1),integrand(r,tt+1,npath)) ) / lampath(r,tt,'1'));
    
************************
* Output solutions 

File sol_SCEQ_RBC_C /sol_SCEQ_RBC_C.csv/;
sol_SCEQ_RBC_C.pc=5;
sol_SCEQ_RBC_C.pw=4000;

Put sol_SCEQ_RBC_C;

loop(npath,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;    
    loop(r,
      put Cpath(r,tt,npath)::6;
    );
    put /;
  );
);

File sol_SCEQ_RBC_K /sol_SCEQ_RBC_K.csv/;
sol_SCEQ_RBC_K.pc=5;
sol_SCEQ_RBC_K.pw=4000;

Put sol_SCEQ_RBC_K;

loop(npath,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;
    loop(r,
      put Kpath(r,tt,npath)::6;
    );    
    put /;
  );
);

File sol_SCEQ_RBC_I /sol_SCEQ_RBC_I.csv/;
sol_SCEQ_RBC_I.pc=5;
sol_SCEQ_RBC_I.pw=4000;

Put sol_SCEQ_RBC_I;

loop(npath,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;
    loop(r,
      put Ipath(r,tt,npath)::6;
    );    
    put /;
  );
);

File sol_SCEQ_RBC_L /sol_SCEQ_RBC_L.csv/;
sol_SCEQ_RBC_L.pc=5;
sol_SCEQ_RBC_L.pw=4000;

Put sol_SCEQ_RBC_L;

loop(npath,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;
    loop(r,
      put Lpath(r,tt,npath)::6;
    );    
    put /;
  );
);

File sol_SCEQ_RBC_err /sol_SCEQ_RBC_err.csv/;
sol_SCEQ_RBC_err.pc=5;
sol_SCEQ_RBC_err.pw=4000;

Put sol_SCEQ_RBC_err;

loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;
    loop(r,
      put errs(r,tt)::6;
    ); 
    put /;
);

* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

