$ontext
This GAMS code applies SCEQ to solve DSICE with economic risk

This code is based on
(1) Working paper about DICE-CJL model:
    Cai, Yongyang, Kenneth L. Judd and Thomas S. Lontzek (2012).
    Continuous-Time Methods for Integrated Assessment Models. NBER working paper No. w18365.
(2) DICE 2007 with 10 year time periods given in the webpage:
    http://nordhaus.econ.yale.edu/DICE2007_short.gms

* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
$offtext

$include SCEQ_DSICE_TFP_common.gms

set st(t1) periods for error checking /1,5,10,20,50,100/;
    
parameter    
    prob1(j)
    Ys(j)
    Eind(j)
    dF(t)
    dMitiCost(j)
    Dam(t)
    dDam(t)
    lamK2(j)
    lamMAT2(j)
    lamMU2(j)
    lamML2(j)
    lamTA2(j)
    lamTL2(j)
            
    errsK(st,nsim)
    errsMAT(st,nsim)
    errsMU(st,nsim)
    errsML(st,nsim)
    errsTA(st,nsim)
    errsTL(st,nsim)
;

$gdxin sim_paths_SCEQ_DSICE_TFP
$load thetapath Kpath MATpath MUpath MLpath TApath TLpath Cpath miupath lamK lamMAT lamMU lamML lamTA lamTL   

loop(st,
* starting period
    s = st.val + 1;

    loop(nsim,    
* fix the state variable at s 
        K.fx(t)$(ord(t)=s) = Kpath(t,nsim); 
        MAT.fx(t)$(ord(t)=s) = MATpath(t,nsim);
        MU.fx(t)$(ord(t)=s) = MUpath(t,nsim);
        ML.fx(t)$(ord(t)=s) = MLpath(t,nsim);
        TATM.fx(t)$(ord(t)=s) = TApath(t,nsim);
        TOCEAN.fx(t)$(ord(t)=s) = TLpath(t,nsim);
        
        prob1(j) = sum(j2$(TFPshock(j2)=thetapath(st,nsim)),tranProbs(j,j2));    
        loop(j3$(prob1(j3)>0),
* compute the probability distribution of TFP(t) from the 
* distribution at s conditional on TFP(s)  
            probs(t,j)$(ord(t)=s) = 0;
            probs(t,j3)$(ord(t)=s) = 1;
            loop(t$(ord(t)>=s),
              probs(t+1,j) = sum(j2, tranProbs(j,j2) * probs(t,j2));    
            );
        
* compute expectation of TFP(t) conditional on a TFP(s)
            TFP(t)$(ord(t)>=s) = sum(j, probs(t,j)*TFPshock(j)) * AL(t);
    
            solve DSICE_TFP maximizing Welfare using nlp ;
            
            lamK2(j3) = sum(t$(ord(t)=s), KK.m(t));
            lamMAT2(j3) = sum(t$(ord(t)=s), MMAT.m(t));
            lamMU2(j3) = sum(t$(ord(t)=s), MMU.m(t));
            lamML2(j3) = sum(t$(ord(t)=s), MML.m(t));
            lamTA2(j3) = sum(t$(ord(t)=s), TATMEQ.m(t));
            lamTL2(j3) = sum(t$(ord(t)=s), TOCEANEQ.m(t));

            Ys(j3) = sum(t$(ord(t)=s), Y.l(t));
            Eind(j3) = sum(t$(ord(t)=s), SIGMA(t)*(1-MIU.l(t))* GrossY.l(t));
            dMitiCost(j3) = sum(t$(ord(t)=s), cost1(t)*(EXPcost2*(miu.l(t)**(EXPcost2-1))));
        );

        Dam(t)$(ord(t)=s) = a1*TATM.l(t) + a2*sqr(TATM.l(t));        
        dF(t)$(ord(t)=s) = FCO22X/(MAT.l(t)*log(2));
        dDam(t)$(ord(t)=s) = a1 + 2*a2*TATM.l(t);

        errsK(st,nsim) = abs(sum( t$(ord(t)=s), beta * sum(j, prob1(j)*
            (lamK2(j)*((1-DK) + Ys(j)*GAMA / K.l(t)) + lamMAT2(j)*Eind(j)*GAMA / K.l(t))) ) / lamK(st,nsim) - 1);
        errsMAT(st,nsim) = abs(sum( t$(ord(t)=s), beta * sum(j, prob1(j)*
            (lamMAT2(j)*b11 + lamMU2(j)*b12 + lamTA2(j)*C1*dF(t))) ) / lamMAT(st,nsim) - 1);  
        errsMU(st,nsim) = abs(beta*sum(j, prob1(j)*(lamMAT2(j)*b21 + lamMU2(j)*b22 + lamML2(j)*b23)) / lamMU(st,nsim) - 1);
        errsML(st,nsim) = abs(beta * sum(j, prob1(j)*(lamMU2(j)*b32 + lamML2(j)*b33)) / lamML(st,nsim) - 1);
        errsTA(st,nsim) = abs(sum( t$(ord(t)=s), beta * sum(j, prob1(j)*
            (lamTA2(j)*(1-C1*LAM-C1*C3) + lamTL2(j)*C4 + lamK2(j)*dDam(t)*((-Ys(j))/(1+Dam(t))))) ) /
            lamTA(st,nsim) - 1);
        errsTL(st,nsim) = abs(beta * sum(j, prob1(j)*(lamTA2(j)*C1*C3 + lamTL2(j)*(1-C4))) / lamTL(st,nsim) - 1);
    );
);    
    
***************
* Output solutions 

file SCEQ_DSICE_TFP_errs;
put SCEQ_DSICE_TFP_errs;
SCEQ_DSICE_TFP_errs.nw = 12;
SCEQ_DSICE_TFP_errs.nr = 2;
SCEQ_DSICE_TFP_errs.nz = 1e-15;
loop(nsim,
  loop(st,
    put st.tl:4:0;
    put errsK(st,nsim):13:5;
    put errsMAT(st,nsim):13:5;
    put errsMU(st,nsim):13:5;
    put errsML(st,nsim):13:5;
    put errsTA(st,nsim):13:5;
    put errsTL(st,nsim):13:5;
    put /;
  );  
);


* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

