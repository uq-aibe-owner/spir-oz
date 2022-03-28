** This is a code of SCEQ to solve a multi-country real business cycle problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include RBC_SCEQ_common.gms

*********************
* run SCEQ error checking

set st(tt) periods for error checking /1,5,10,15,20/;
*alias(st, tt);

parameter 
    L2(j,ii)      labor
    Inv2(j,ii)    investment
    lam2(j,ii)    shadow prices for capital transition
    mu2(ii)       shadow prices for budget constraint
    errs(j,st,nsim) Euler errors
;


$gdxin sim_paths_SCEQ_RBC
$load Kpath thetapath lampath Ipath

set niter / 1*10 /;

L2(j,ii3) = 1;

loop(st,
* starting period s
  s = st.val + 1;

* optimization step
  loop(nsim,
* fix the state variable at s
    K.fx(j,t)$(ord(t)=s) = Kpath(j,t,nsim);
        
    prob1(ii) = sum(ii2$(TFPshock(ii2)=thetapath(st,nsim)),tranProbs(ii,ii2));    
    loop(ii3$(prob1(ii3)>0),
* use the certainty equivalent approximation
    
* compute the probability distribution of TFPshock(t) from the 
* distribution at s conditional on TFPshock(s)  
        probs(t,ii)$(ord(t)=s) = 0;
        probs(t,ii3)$(ord(t)=s) = 1;
        loop(t$(ord(t)>=s),
          probs(t+1,ii) = sum(ii2, tranProbs(ii,ii2) * probs(t,ii2));    
        );
    
* compute expectation of TFPshock(t) conditional on a TFPshock(s)
        theta(t)$(ord(t)>=s) = sum(ii, probs(t,ii)*TFPshock(ii));
            
        loop(niter,
            solve busc using nlp maximizing obj;
            if((busc.MODELSTAT<=2 and busc.SOLVESTAT=1),
                break;
            );
        );
        abort$(busc.MODELSTAT>2 or busc.SOLVESTAT<>1) "FAILED in solving!";
        
* save decisions and multipliers for computing Euler errors
        L2(j,ii3) = sum(t$(ord(t)=s), L.l(j,t));
        Inv2(j,ii3) = sum(t$(ord(t)=s), Inv.l(j,t));
        lam2(j,ii3) = sum(t$(ord(t)=s), TransitionCapital.m(j,t));
        mu2(ii3) = sum(t$(ord(t)=s), BudgetConstraint.m(t));
    );
        
    errs(j,st,nsim) = sum( t$(ord(t)=s), abs(1 - beta*sum(ii, prob1(ii)*( lam2(j,ii)*(1-delta) + 
        mu2(ii) * ( TFPshock(ii) * A*alpha*((Kpath(j,t,nsim)/L2(j,ii))**(alpha-1)) -
        phi/2*sqr(Inv2(j,ii)/Kpath(j,t,nsim)-delta) +
        phi*(Inv2(j,ii)/Kpath(j,t,nsim)-delta)*Inv2(j,ii)/Kpath(j,t,nsim) ) ) )/ lampath(j,st,nsim)) );
  );  
);

***************
* Output solutions 

File sol_sto_err /errs_SCEQ_RBC.csv/;
sol_sto_err.pc=5;
sol_sto_err.pw=4000;

Put sol_sto_err;

loop(nsim,
  loop(st,
    put st.tl::4;
    loop(j, put errs(j,st,nsim)::6; );
    put /;
  );  
);


* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

