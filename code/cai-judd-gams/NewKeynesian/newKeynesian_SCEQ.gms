*----------------------------------------------------------------------
* This program solves the New Keynesian DSGE model with zero lower bound using SCEQ
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include newKeynesian_SCEQ_common.gms

parameter seedv;
option seed = %seedv%;

*******************************************************
* run SCEQ

* initialization step for simulating shocks
betapath('1',nsim) = beta0;
loop(tt$(ord(tt)<=Tstar),
  betapath(tt+1,nsim) = exp((1-rho)*log(betass) + rho*log(betapath(tt,nsim)) + sigma*normal(0,1));
);  

* initial state
vpath('1',nsim) = v0;

*****************************
* solve the first period only

* starting period s
s = 1;    

* optimization step

* set state at the starting period
v.fx('1') = v0;
betas('1') = beta0;

* use certainty equivalent approximation    
loop(t$(ord(t)<card(t) and ord(t)>=s),
    betas(t+1) = exp((1-rho)*log(betass) + rho*log(betas(t)));
);

solve NewKeynesian using dnlp minimizing obj;
            
chi1path('1',nsim) = chi1.l('1');
chi2path('1',nsim) = chi2.l('1');
ypath('1',nsim) = y.l('1');
pipath('1',nsim) = pi_t.l('1');
qpath('1',nsim) = q.l('1');
zpath('1',nsim) = z.l('1');
  
* simulation step    
vpath('2',nsim) = (1-theta)*power(qpath('1',nsim),-alpha) + theta*power(pipath('1',nsim),alpha)*vpath('1',nsim);
  
*****************************
* iterate over periods of interest

loop(tt$(ord(tt)>1 and ord(tt)<=Tstar),
* starting period s
  s = ord(tt);    

* optimization step
  loop(nsim,  
* set state at the starting period
    v.fx(tt) = vpath(tt,nsim);
    betas(tt) = betapath(tt,nsim);

* use certainty equivalent approximation    
    loop(t$(ord(t)<card(t) and ord(t)>=s),
        betas(t+1) = exp((1-rho)*log(betass) + rho*log(betas(t)));
    );

    solve NewKeynesian using dnlp minimizing obj;
            
    chi1path(tt,nsim) = chi1.l(tt);
    chi2path(tt,nsim) = chi2.l(tt);
    ypath(tt,nsim) = y.l(tt);
    pipath(tt,nsim) = pi_t.l(tt);
    qpath(tt,nsim) = q.l(tt);
    zpath(tt,nsim) = z.l(tt);
  );
  
* simulation step    
  vpath(tt+1,nsim) = (1-theta)*power(qpath(tt,nsim),-alpha) + theta*power(pipath(tt,nsim),alpha)*vpath(tt,nsim);        
);

*****************************
* error checking for the first period 
parameter err1
    err2
    err3;

err1 = abs(1 - 1/card(nsim)*sum(nsim, betapath('2',nsim)*(1+max(0,zpath('1',nsim)))/pipath('2',nsim)*ypath('1',nsim)/ypath('2',nsim)));
err2 = abs(1 - 1/card(nsim)*sum(nsim, (ypath('1',nsim)**(1+eta)*vpath('2',nsim)**eta + theta*
        (betapath('2',nsim)*pipath('2',nsim)**alpha*chi1path('2',nsim)))/chi1path('1',nsim)));
err3 = abs(1 - 1/card(nsim)*sum(nsim, (1/(1-sg) + theta*
        (betapath('2',nsim)*pipath('2',nsim)**(alpha-1)*chi2path('2',nsim)))/chi2path('1',nsim)));

display err1, err2, err3;

*****************************
* Output solutions 

* save simulated paths for error checking
execute_unload 'sim_paths_SCEQ_NK', vpath, betapath, chi1path, chi2path, ypath, pipath, qpath, zpath;

File sol_SCEQ_NK /sol_SCEQ_NK.csv/;
sol_SCEQ_NK.pc=5;
sol_SCEQ_NK.pw=4000;

Put sol_SCEQ_NK;

loop(nsim,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;    
    put betapath(tt,nsim)::6;
    put vpath(tt,nsim)::6;
    put chi1path(tt,nsim)::6;
    put chi2path(tt,nsim)::6;
    put ypath(tt,nsim)::6;
    put pipath(tt,nsim)::6;
    put qpath(tt,nsim)::6;
    put zpath(tt,nsim)::6;
    put /;
  );
);


* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

