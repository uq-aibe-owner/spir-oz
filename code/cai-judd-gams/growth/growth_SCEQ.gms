** This is an illustrative code of SCEQ to solve a simple 
** optimal growth problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include growth_SCEQ_common.gms

parameter seedv;
option seed = %seedv%;

************************************************
* run SCEQ

* initiaization step for simulating shocks
thetapath('1',nsim) = theta0;
loop(tt$(ord(tt)<Tstar),
  thetapath(tt+1,nsim) = thetapath(tt,nsim)**rho * exp(sigma*normal(0,1));
);

Kpath('1',nsim) = k0;

************************
* solve the first period only

* starting period  
s = 1;

* optimization step

* fix the state variable at s  
K.fx('1') = k0;
        
* use the certainty equivalent approximation for theta        
theta(t)$(ord(t)>=s) = theta0**(rho**(ord(t)-s));

solve growth using nlp maximizing obj;

Cpath('1',nsim) = C.l('1');        
lampath('1',nsim) = TransitionCapital.m('1');    

* Simulation step
Kpath('2',nsim) = (1-delta)*Kpath('1',nsim) + thetapath('1',nsim)*A*(Kpath('1',nsim)**alpha) - Cpath('1',nsim);     


************************
* iterate forward over periods of interest

loop(tt$(ord(tt)>1 and ord(tt)<=Tstar),
* starting period  
  s = ord(tt);

* optimization step    
  loop(nsim,    
* fix the state variable at s  
    K.fx(tt) = Kpath(tt,nsim);
        
* use the certainty equivalent approximation for theta        
    theta(t)$(ord(t)>=s) = thetapath(tt,nsim)**(rho**(ord(t)-s));

    solve growth using nlp maximizing obj;

    Cpath(tt,nsim) = C.l(tt);        
    lampath(tt,nsim) = TransitionCapital.m(tt);    
  );

* Simulation step
  Kpath(tt+1,nsim) = (1-delta)*Kpath(tt,nsim) + thetapath(tt,nsim)*A*(Kpath(tt,nsim)**alpha) - Cpath(tt,nsim);     
);

************************
* error checking for the first period 
parameter err;

err = abs(1 - beta/card(nsim)*sum(nsim, (lampath('2',nsim)*
    (1-delta+thetapath('2',nsim)*A*alpha*(Kpath('2',nsim)**(alpha-1))))/lampath('1',nsim)));

display err;

************************
* Output solutions 

* save simulated paths for error checking
execute_unload 'sim_paths_SCEQ_growth', Kpath, thetapath, Cpath, lampath;

File sol_SCEQ_growth /sol_SCEQ_growth.csv/;
sol_SCEQ_growth.pc=5;
sol_SCEQ_growth.pw=4000;

Put sol_SCEQ_growth;

Put "t";
Put "K";
Put "theta";
Put "C";
Put "lam";

put /;	
loop(nsim,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;
    put Kpath(tt,nsim)::6;
    put thetapath(tt,nsim)::6;    
    put Cpath(tt,nsim)::6;
    put lampath(tt,nsim)::6;
    put /;
  );
);

* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

