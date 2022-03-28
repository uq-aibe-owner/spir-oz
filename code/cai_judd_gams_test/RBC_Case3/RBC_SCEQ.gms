** This is a code of SCEQ to solve a multi-country real business cycle problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include RBC_SCEQ_common.gms

parameter seedv;
option seed = %seedv%;

*********************
* run SCEQ

parameter 
    Cpath(j,tt,nsim) simulated consumption paths
    Ipath(j,tt,nsim) simulated investment paths
    Lpath(j,tt,nsim) simulated labor supply paths
    Kpath(j,tt,nsim) simulated capital paths
    thetapath(j,tt,nsim) simulated paths of shocks
    lampath(j,tt,nsim) shadow prices for capital transition
    mupath(tt,nsim) shadow prices for budget constraint
    systemShock     realized system shock     
;

set niter / 1*10 /;

* Initialization step for simulating shocks
thetapath(j,'1',nsim) = 1;
loop((tt,nsim)$(ord(tt)<card(tt)),
  systemShock = sigma2*normal(0,1);
  thetapath(j,tt+1,nsim) = thetapath(j,tt,nsim)**rho * exp(sigma1*normal(0,1)+systemShock);
);

Kpath(j,'1',nsim) = K0(j);

* solve the first period only: note that its solution is independent of simulation
* starting period s
s = 1;    

* optimization step 

* fix the state variable at s 
K.fx(j,'1') = K0(j);
        
theta(j,t)$(ord(t)>=s) = thetapath(j,'1','1')**(rho**(ord(t)-s));

loop(niter,
    solve busc using nlp maximizing obj;
    if((busc.MODELSTAT<=2 and busc.SOLVESTAT=1),
        break;
    );
);
abort$(busc.MODELSTAT>2 or busc.SOLVESTAT<>1) "FAILED in solving!";

Cpath(j,'1',nsim) = C.l(j,'1');
Ipath(j,'1',nsim) = Inv.l(j,'1');
Lpath(j,'1',nsim) = L.l(j,'1');
lampath(j,'1',nsim) = TransitionCapital.m(j,'1');
mupath('1',nsim) = BudgetConstraint.m('1');
  
* simulation step
Kpath(j,'2',nsim) = (1-delta)*Kpath(j,'1',nsim) + Ipath(j,'1',nsim);
  
* iterate over periods of interest
loop(tt$(ord(tt)>1),
* starting period s
  s = ord(tt);    

* optimization step 
  loop(nsim,  
* fix the state variable at s 
    K.fx(j,tt) = Kpath(j,tt,nsim);
        
    theta(j,t)$(ord(t)>=s) = thetapath(j,tt,nsim)**(rho**(ord(t)-s));

    loop(niter,
        solve busc using nlp maximizing obj;
        if((busc.MODELSTAT<=2 and busc.SOLVESTAT=1),
            break;
        );
    );
    abort$(busc.MODELSTAT>2 or busc.SOLVESTAT<>1) "FAILED in solving!";

    Cpath(j,tt,nsim) = C.l(j,tt);
    Ipath(j,tt,nsim) = Inv.l(j,tt);
    Lpath(j,tt,nsim) = L.l(j,tt);
    lampath(j,tt,nsim) = TransitionCapital.m(j,tt);
    mupath(tt,nsim) = BudgetConstraint.m(tt);
  );
  
* simulation step
  Kpath(j,tt+1,nsim) = (1-delta)*Kpath(j,tt,nsim) + Ipath(j,tt,nsim);
);

***************
* error checking for the first period 
parameter errs(j);

errs(j) = abs(1 - beta/card(nsim)*sum(nsim, ( lampath(j,'2',nsim)*(1-delta) + 
        mupath('2',nsim) * ( thetapath(j,'2',nsim) * A*alpha*((Kpath(j,'2',nsim)/Lpath(j,'2',nsim))**(alpha-1)) -
        phi/2*sqr(Ipath(j,'2',nsim)/Kpath(j,'2',nsim)-delta) +
        phi*(Ipath(j,'2',nsim)/Kpath(j,'2',nsim)-delta)*Ipath(j,'2',nsim)/Kpath(j,'2',nsim) ) ) / lampath(j,'1',nsim)));

display errs;

***************
* Output solutions 

File sol_SCEQ_RBC_theta /sol_SCEQ_RBC_theta.csv/;
sol_SCEQ_RBC_theta.pc=5;
sol_SCEQ_RBC_theta.pw=4000;

Put sol_SCEQ_RBC_theta;

loop(nsim,
  loop(tt,
    put tt.tl::4;    
    loop(j,
      put thetapath(j,tt,nsim)::6;
    );
    put /;
  );
);

File sol_SCEQ_RBC_C /sol_SCEQ_RBC_C.csv/;
sol_SCEQ_RBC_C.pc=5;
sol_SCEQ_RBC_C.pw=4000;

Put sol_SCEQ_RBC_C;

loop(nsim,
  loop(tt,
    put tt.tl::4;    
    loop(j,
      put Cpath(j,tt,nsim)::6;
    );
    put /;
  );
);


File sol_SCEQ_RBC_K /sol_SCEQ_RBC_K.csv/;
sol_SCEQ_RBC_K.pc=5;
sol_SCEQ_RBC_K.pw=4000;

Put sol_SCEQ_RBC_K;

loop(nsim,
  loop(tt,
    put tt.tl::4;
    loop(j,
      put Kpath(j,tt,nsim)::6;
    );    
    put /;
  );
);

File sol_SCEQ_RBC_I /sol_SCEQ_RBC_I.csv/;
sol_SCEQ_RBC_I.pc=5;
sol_SCEQ_RBC_I.pw=4000;

Put sol_SCEQ_RBC_I;

loop(nsim,
  loop(tt,
    put tt.tl::4;
    loop(j,
      put Ipath(j,tt,nsim)::6;
    );    
    put /;
  );
);

File sol_SCEQ_RBC_L /sol_SCEQ_RBC_L.csv/;
sol_SCEQ_RBC_L.pc=5;
sol_SCEQ_RBC_L.pw=4000;

Put sol_SCEQ_RBC_L;

loop(nsim,
  loop(tt,
    put tt.tl::4;
    loop(j,
      put Lpath(j,tt,nsim)::6;
    );    
    put /;
  );
);

File sol_SCEQ_RBC_lam /sol_SCEQ_RBC_lam.csv/;
sol_SCEQ_RBC_lam.pc=5;
sol_SCEQ_RBC_lam.pw=4000;

Put sol_SCEQ_RBC_lam;

loop(nsim,
  loop(tt,
    put tt.tl::4;
    loop(j,
      put lampath(j,tt,nsim)::6;
    );    
    put /;
  );
);

File sol_SCEQ_RBC_mu /sol_SCEQ_RBC_mu.csv/;
sol_SCEQ_RBC_mu.pc=5;
sol_SCEQ_RBC_mu.pw=4000;

Put sol_SCEQ_RBC_mu;

loop(nsim,
  loop(tt,
    put tt.tl::4;
    put mupath(tt,nsim)::6;
    put /;
  );
);

* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

