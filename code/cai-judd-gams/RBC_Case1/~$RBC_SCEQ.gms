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
    Lpath(j,tt,nsim) simulated labor supply paths
    mupath(tt,nsim) shadow prices for budget constraint
    epsilon(tt,nsim)     realized shock
;

set niter / 1*10 /;

* initiaization step for simulating shocks
epsilon(tt,nsim) = Uniform(0,1);
thetapath('1',nsim) = TFPshock('2');
loop((tt,nsim)$(ord(tt)<Tstar),    
    loop(ii$(TFPshock(ii)=thetapath(tt,nsim)),
        if(epsilon(tt,nsim)<=tranProbs('1',ii),
            thetapath(tt+1,nsim) = TFPshock('1');            
        else 
            if(epsilon(tt,nsim)<=tranProbs('1',ii)+tranProbs('2',ii),
                thetapath(tt+1,nsim) = TFPshock('2');
            else
                thetapath(tt+1,nsim) = TFPshock('3');
            );
        );
    );
);

* initial state
Kpath(j,'1',nsim) = k0(j);

************************
* solve the first period only: note that its solution is independent of simulation

* starting period  
s = 1;    

* use the certainty equivalent approximation:
  
* compute the probability distribution of TFPshock(t) from the 
* distribution at s conditional on a realized TFPshock(s)  
probs(t,ii)$(ord(t)=s) = 0;
probs(t,ii)$(ord(t)=s and TFPshock(ii)=thetapath('1','1')) = 1;
loop(t$(ord(t)>=s),
    probs(t+1,ii) = sum(ii2, tranProbs(ii,ii2) * probs(t,ii2));    
);

* compute expectation of TFPshock(t) conditional on a realized TFPshock(s)
theta(t)$(ord(t)>=s) = sum(ii, probs(t,ii)*TFPshock(ii));

* fix the state variable at s to be its solution at the last iteration
K.fx(j,'1') = K0(j);

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

  
************************
* iterate over periods of interest

loop(tt$(ord(tt)>1 and ord(tt)<=Tstar),
* starting period  
  s = ord(tt);    

* optimization step
  loop(nsim,  
* use the certainty equivalent approximation:
  
* compute the probability distribution of TFPshock(t) from the 
* distribution at s conditional on a realized TFPshock(s)  
    probs(t,ii)$(ord(t)=s) = 0;
    probs(t,ii)$(ord(t)=s and TFPshock(ii)=thetapath(tt,nsim)) = 1;
    loop(t$(ord(t)>=s),
      probs(t+1,ii) = sum(ii2, tranProbs(ii,ii2) * probs(t,ii2));    
    );

* compute expectation of TFPshock(t) conditional on a realized TFPshock(s)
    theta(t)$(ord(t)>=s) = sum(ii, probs(t,ii)*TFPshock(ii));

* fix the state variable at s to be its solution at the last iteration
    K.fx(j,tt) = Kpath(j,tt,nsim);

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

************************
* error checking for the first period 
parameter err(j);

err(j) = abs( 1 - beta/card(nsim)*sum( nsim, ( lampath(j,'2',nsim)*(1-delta) + 
        mupath('2',nsim) * ( thetapath('2',nsim) * A*alpha*((Kpath(j,'2',nsim)/Lpath(j,'2',nsim))**(alpha-1)) -
        phi/2*sqr(Ipath(j,'2',nsim)/Kpath(j,'2',nsim)-delta) +
        phi*(Ipath(j,'2',nsim)/Kpath(j,'2',nsim)-delta)*Ipath(j,'2',nsim)/Kpath(j,'2',nsim) ) ) / lampath(j,'1',nsim) ) );

display err;

************************
* Output solutions 

* save simulated paths for error checking
execute_unload 'sim_paths_SCEQ_RBC', Kpath, thetapath, lampath, Ipath;

File sol_SCEQ_RBC_theta /sol_SCEQ_RBC_theta.csv/;
sol_SCEQ_RBC_theta.pc=5;
sol_SCEQ_RBC_theta.pw=4000;

Put sol_SCEQ_RBC_theta;

loop(nsim,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;    
    put thetapath(tt,nsim)::6;
    put /;
  );
);


File sol_SCEQ_RBC_C /sol_SCEQ_RBC_C.csv/;
sol_SCEQ_RBC_C.pc=5;
sol_SCEQ_RBC_C.pw=4000;

Put sol_SCEQ_RBC_C;

loop(nsim,
  loop(tt$(ord(tt)<=Tstar),
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
  loop(tt$(ord(tt)<=Tstar),
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
  loop(tt$(ord(tt)<=Tstar),
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
  loop(tt$(ord(tt)<=Tstar),
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
  loop(tt$(ord(tt)<=Tstar),
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
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;
    put mupath(tt,nsim)::6;
    put /;
  );
);

* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

