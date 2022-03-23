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
    Cpath(reg,tt,npath) simulated consumption paths
    Ipath(reg,tt,npath) simulated investment paths
    Lpath(reg,tt,npath) simulated labor supply paths
    Kpath(reg,tt,npath) simulated capital paths
    lampath(reg,tt,npath) shadow prices for capital transition
    mupath(tt,npath) shadow prices for budget constraint
;

Cpath(reg,tt,npath) = 1;
Ipath(reg,tt,npath) = 1;
Lpath(reg,tt,npath) = 1;
Kpath(reg,tt,npath) = 1;
lampath(reg,tt,npath) = 1;
mupath(tt,npath) = 1;
    
set niter / 1*10 /;

Kpath(reg,'1',npath) = K0(reg);

************************
* solve the pre-tipping path

loop(npath$(ord(npath)=1),
* iterate over periods of interest (the last extra period is for error checking only)
  loop(tt$(ord(tt)<=Tstar+1),
* starting period
    s = ord(tt);    

* optimization step

* fix the state variable at s
    K.fx(reg,tt) = Kpath(reg,tt,npath);
    
* if tipping event has not happened by the beginning of the current period
    Probs(t)$(ord(t)>=s) = (1-prob1)**(ord(t)-s);
        
    loop(niter,
        solve busc using nlp maximizing obj;
        if((busc.MODELSTAT<=2 and busc.SOLVESTAT=1),
            break;
        );
    );
    abort$(busc.MODELSTAT>2 or busc.SOLVESTAT<>1) "FAILED in solving!";

    Cpath(reg,tt,npath) = C.l(reg,tt);
    Ipath(reg,tt,npath) = Inv.l(reg,tt);
    Lpath(reg,tt,npath) = L.l(reg,tt);
    lampath(reg,tt,npath) = TransitionCapital.m(reg,tt);
    mupath(tt,npath) = BudgetConstraint.m(tt);

* simulation step    
    Kpath(reg,tt+1,npath) = (1-delta)*Kpath(reg,tt,npath) + Ipath(reg,tt,npath);
  );
  
* relax the fixed constraints on the state variables
  K.lo(reg,t) = 0.001;
  K.up(reg,t) = 1000;  
);


************************
* solve the tipped paths

loop(npath$(ord(npath)>1),
* starting period is also the period that the tipped event happens
    s = ord(npath);

* fix the state variable at s: the tipping event happens at s but the capital at s has not been impacted 
    K.fx(reg,tt)$(ord(tt)=s) = Kpath(reg,tt,'1');
    
    loop(niter,
        solve busc_tipped using nlp maximizing obj;
        if((busc_tipped.MODELSTAT<=2 and busc_tipped.SOLVESTAT=1),
            break;
        );
    );
    abort$(busc_tipped.MODELSTAT>2 or busc_tipped.SOLVESTAT<>1) "FAILED in solving!";

    Cpath(reg,tt,npath) = C.l(reg,tt);
    Ipath(reg,tt,npath) = Inv.l(reg,tt);
    Lpath(reg,tt,npath) = L.l(reg,tt);
    Kpath(reg,tt,npath) = K.l(reg,tt);
    lampath(reg,tt,npath) = TransitionCapital.m(reg,tt);
    mupath(tt,npath) = BudgetConstraint.m(tt);
    
* relax the fixed constraints on the state variables
    K.lo(reg,t) = 0.001;
    K.up(reg,t) = 1000;  
);

************************
* compute Euler errors at the pre-tipping path

parameter integrand(reg,tt,npath)
    errs(reg,tt);

integrand(reg,tt,'1')$(ord(tt)<=Tstar+1) = lampath(reg,tt,'1')*(1-delta) + 
    mupath(tt,'1') * ( A*alpha*((Kpath(reg,tt,'1')/Lpath(reg,tt,'1'))**(alpha-1)) -
    phi/2*sqr(Ipath(reg,tt,'1')/Kpath(reg,tt,'1')-delta) +
    phi*(Ipath(reg,tt,'1')/Kpath(reg,tt,'1')-delta)*Ipath(reg,tt,'1')/Kpath(reg,tt,'1') );
integrand(reg,tt,npath)$(ord(npath)>1 and ord(tt)<=Tstar+1) = lampath(reg,tt,npath)*(1-delta) + 
    mupath(tt,npath) * ( A*alpha*((Kpath(reg,tt,npath)/Lpath(reg,tt,npath))**(alpha-1)) -
    phi/2*sqr(Ipath(reg,tt,npath)/Kpath(reg,tt,npath)-delta) +
    phi*(Ipath(reg,tt,npath)/Kpath(reg,tt,npath)-delta)*Ipath(reg,tt,npath)/Kpath(reg,tt,npath) );
   
errs(reg,tt)$(ord(tt)<=Tstar) = abs(1 - beta*( (1-prob1)*integrand(reg,tt+1,'1') +
    prob1*sum(npath$(ord(npath)=ord(tt)+1),integrand(reg,tt+1,npath)) ) / lampath(reg,tt,'1'));
    
************************
* Output solutions 

File sol_SCEQ_RBC_C /sol_SCEQ_RBC_C.csv/;
sol_SCEQ_RBC_C.pc=5;
sol_SCEQ_RBC_C.pw=4000;

Put sol_SCEQ_RBC_C;

loop(npath,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;    
    loop(reg,
      put Cpath(reg,tt,npath)::6;
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
    loop(reg,
      put Kpath(reg,tt,npath)::6;
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
    loop(reg,
      put Ipath(reg,tt,npath)::6;
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
    loop(reg,
      put Lpath(reg,tt,npath)::6;
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
    loop(reg,
      put errs(reg,tt)::6;
    ); 
    put /;
);

* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

