** This is an illustrative code of SCEQ to solve a simple 
** optimal growth problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include growth_SCEQ_common.gms

*----------------------------------------------------------------------
* run error checking

set ii indices of Gaussian Hermite quadrature nodes /1*7/;
set nw nodes and weights /1*2/;

Table GHData(ii,nw)
$ondelim
$include GaussHermiteData.txt
$offdelim
;

parameter   GHNodes(ii)     Gaussian Hermite quadrature nodes
            GHWeights(ii)   Gaussian Hermite quadrature weights;
GHNodes(ii) =  GHData(ii,'1')*sqrt(2.0);
GHWeights(ii) =  GHData(ii,'2')/sqrt(pi);


parameter
    theta2(tt,nsim,ii)
    lam2(tt,nsim,ii)    shadow prices
    errs(tt,nsim)       Euler errors    
;

$gdxin sim_paths_SCEQ_growth
$load Kpath thetapath Cpath lampath

* quadrature nodes for theta 
theta2(tt,nsim,ii) = thetapath(tt-1,nsim)**rho * exp(sigma*GHNodes(ii));
  
loop(tt$(ord(tt)>1 and ord(tt)<=Tstar+1),
* starting period
  s = ord(tt);

* optimization step
  loop(nsim,          
* fix the state variable at s   
    K.fx(tt) = Kpath(tt,nsim);    
    
    loop(ii,
* use the certainty equivalent approximation for theta        
        theta(t)$(ord(t)>=s) = theta2(tt,nsim,ii)**(rho**(ord(t)-s));
        
        solve growth using nlp maximizing obj;
        
        lam2(tt,nsim,ii) = TransitionCapital.m(tt);
    );    
  );  
);


errs(tt-1,nsim) = abs(1 - beta*sum(ii, GHWeights(ii)*(lam2(tt,nsim,ii)*
    (1-delta+theta2(tt,nsim,ii)*A*alpha*(Kpath(tt,nsim)**(alpha-1)))))/lampath(tt-1,nsim));

***************
* Output solutions 

File err_SCEQ_growth /err_SCEQ_growth.csv/;
err_SCEQ_growth.pc=5;
err_SCEQ_growth.pw=4000;

Put err_SCEQ_growth;

Put "t";
Put "Err";

put /;	
loop(nsim,
  loop(tt$(ord(tt)<=Tstar),
    put tt.tl::4;
    put errs(tt,nsim)::6;
    put /;
  );
);


* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;
