*----------------------------------------------------------------------
* This program solves the New Keynesian DSGE model with zero lower bound using SCEQ
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

$include newKeynesian_SCEQ_common.gms

*******************************************************

$gdxin sim_paths_SCEQ_NK
$load vpath betapath chi1path chi2path ypath pipath qpath zpath

set ii indices of Gaussian Hermite quadrature nodes /1*15/;
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

set st(tt) periods for error checking /1,5,10,15,20/;


parameter
    beta2(st,nsim,ii) next period beta   
    chi1_2(st,nsim,ii) next period chi1
    chi2_2(st,nsim,ii) next period chi2
    y2(st,nsim,ii) next period y
    pi2(st,nsim,ii) next period pi_t
    z2(st,nsim,ii) next period z
    
    errs1(st,nsim)
    errs2(st,nsim)
    errs3(st,nsim)
;

* quadature nodes 
beta2(st,nsim,ii) = exp((1-rho)*log(betass) + rho*log(betapath(st,nsim)) + sigma*GHNodes(ii));

* iterate over periods
loop(st,
* starting period
  s = st.val + 1;

* optimization step    
  loop(nsim,
* fix the state variable at s  
    v.fx(t)$(ord(t)=s) = vpath(t,nsim);   

    loop(ii,
* use the certainty equivalent approximation for theta
        betas(t)$(ord(t)=s) = beta2(st,nsim,ii);        
        loop(t$(ord(t)<card(t) and ord(t)>=s),
            betas(t+1) = exp((1-rho)*log(betass) + rho*log(betas(t)));
        );

        solve NewKeynesian using dnlp minimizing obj;

        chi1_2(st,nsim,ii) = sum(t$(ord(t)=s), chi1.l(t));
        chi2_2(st,nsim,ii) = sum(t$(ord(t)=s), chi2.l(t));
        y2(st,nsim,ii) = sum(t$(ord(t)=s), y.l(t));
        pi2(st,nsim,ii) = sum(t$(ord(t)=s), pi_t.l(t));
        z2(st,nsim,ii) = sum(t$(ord(t)=s), z.l(t));
    );    
  );
);

errs1(st,nsim) = abs(1 - sum(ii, GHWeights(ii)*
    (beta2(st,nsim,ii)*(1+max(0,z2(st,nsim,ii)))/pi2(st,nsim,ii)*ypath(st,nsim)/y2(st,nsim,ii))));
errs2(st,nsim) = abs(1 - (ypath(st,nsim)**(1+eta)*sum(t$(ord(t)=st.val+1),vpath(t,nsim)**eta) + theta*sum(ii, GHWeights(ii)*
    (beta2(st,nsim,ii)*pi2(st,nsim,ii)**alpha*chi1_2(st,nsim,ii))))/chi1path(st,nsim));
errs3(st,nsim) = abs(1 - (1/(1-sg) + theta*sum(ii, GHWeights(ii)*
    (beta2(st,nsim,ii)*pi2(st,nsim,ii)**(alpha-1)*chi2_2(st,nsim,ii))))/chi2path(st,nsim));  


***************
* Output solutions 

File err_SCEQ_NK /err_SCEQ_NK.csv/;
err_SCEQ_NK.pc=5;
err_SCEQ_NK.pw=4000;

Put err_SCEQ_NK;

loop(nsim,
  loop(st,
    put st.tl::4;    
    put errs1(st,nsim)::6;
    put errs2(st,nsim)::6;
    put errs3(st,nsim)::6;
    put /;
  );
);


* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

