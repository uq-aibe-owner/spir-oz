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

parameter seedv;
option seed = %seedv%;

*************************
* run SCEQ

parameter
    opt_SCC(t,nsim)
    opt_tax(t,nsim)
    epsilon(t,nsim)     realized shock
;

* initiaization step for simulating shocks
epsilon(t,nsim) = Uniform(0,1);
thetapath('1',nsim) = TFPshock('2');
loop((t,nsim)$(ord(t)<Tstar),    
    loop(j$(TFPshock(j)=thetapath(t,nsim)),
        if(epsilon(t,nsim)<=tranProbs('1',j),
            thetapath(t+1,nsim) = TFPshock('1');            
        else 
            if(epsilon(t,nsim)<=tranProbs('1',j)+tranProbs('2',j),
                thetapath(t+1,nsim) = TFPshock('2');
            else
                thetapath(t+1,nsim) = TFPshock('3');
            );
        );
    );
);

Kpath('1',nsim) = K0;
MATpath('1',nsim) = MAT2000;
MUpath('1',nsim) = MU2000;
MLpath('1',nsim) = ML2000;    
TApath('1',nsim) = TATM0;
TLpath('1',nsim) = TOCEAN0;

*************************
* solve the first period only

* starting period
s = ord(t1);

* optimization step

* fix the state variable at s
K.fx('1') = Kpath('1','1');
MAT.fx('1') = MATpath('1','1');
MU.fx('1') = MUpath('1','1');
ML.fx('1') = MLpath('1','1');
TATM.fx('1') = TApath('1','1');
TOCEAN.fx('1') = TLpath('1','1');

* use the certainty equivalent approximation:
        
* compute the probability distribution of TFP(t) from the 
* distribution at s conditional on a realized TFP(s)  
probs(t,j)$(ord(t)=s) = 0;
probs(t,j)$(ord(t)=s and TFPshock(j)=thetapath('1','1')) = 1;
loop(t$(ord(t)>=s),
    probs(t+1,j) = sum(j2, tranProbs(j,j2) * probs(t,j2));    
);
    
* compute expectation of TFP(t) conditional on a realized TFP(s)
TFP(t)$(ord(t)>=s) = sum(j, probs(t,j)*TFPshock(j))*AL(t);
                
solve DSICE_TFP maximizing Welfare using nlp ;
            
miupath('1',nsim) = miu.l('1');
Cpath('1',nsim) = C.l('1');
opt_SCC('1',nsim) = -1*MMAT.m('1')*1000/(KK.m('1')+.00000000001);
opt_tax('1',nsim) = expcost2*cost1('1')*miu.l('1')**(expcost2-1)/sigma('1')*1000;
lamK('1',nsim) = KK.m('1');
lamMAT('1',nsim) = MMAT.m('1');
lamMU('1',nsim) = MMU.m('1');
lamML('1',nsim) = MML.m('1');
lamTA('1',nsim) = TATMEQ.m('1');
lamTL('1',nsim) = TOCEANEQ.m('1');
GrossYpath('1',nsim) = GrossY.l('1');
Ypath('1',nsim) = Y.l('1');


* simulation step
Kpath('2',nsim) = K.l('2');
MATpath('2',nsim) = MAT.l('2');
MUpath('2',nsim) = MU.l('2');
MLpath('2',nsim) = ML.l('2');   
TApath('2',nsim) = TATM.l('2');
TLpath('2',nsim) = TOCEAN.l('2');

    
*************************
* iterate over periods of interest

loop(t1$(ord(t1)>1 and ord(t1)<=Tstar),
* starting period
    s = ord(t1);

* optimization step
    loop(nsim,
* fix the state variable at s
        K.fx(t1) = Kpath(t1,nsim);
        MAT.fx(t1) = MATpath(t1,nsim);
        MU.fx(t1) = MUpath(t1,nsim);
        ML.fx(t1) = MLpath(t1,nsim);
        TATM.fx(t1) = TApath(t1,nsim);
        TOCEAN.fx(t1) = TLpath(t1,nsim);

* use the certainty equivalent approximation:
        
* compute the probability distribution of TFP(t) from the 
* distribution at s conditional on a realized TFP(s)  
        probs(t,j)$(ord(t)=s) = 0;
        probs(t,j)$(ord(t)=s and TFPshock(j)=thetapath(t1,nsim)) = 1;
        loop(t$(ord(t)>=s),
          probs(t+1,j) = sum(j2, tranProbs(j,j2) * probs(t,j2));    
        );
    
* compute expectation of TFP(t) conditional on a realized TFP(s)
        TFP(t)$(ord(t)>=s) = sum(j, probs(t,j)*TFPshock(j))*AL(t);
                
        solve DSICE_TFP maximizing Welfare using nlp ;
            
        miupath(t1,nsim) = miu.l(t1);
        Cpath(t1,nsim) = C.l(t1);            
        opt_SCC(t1,nsim) = -1*MMAT.m(t1)*1000/(KK.m(t1)+.00000000001);
        opt_tax(t1,nsim) = expcost2*cost1(t1)*miu.l(t1)**(expcost2-1)/sigma(t1)*1000;
        lamK(t1,nsim) = KK.m(t1);
        lamMAT(t1,nsim) = MMAT.m(t1);
        lamMU(t1,nsim) = MMU.m(t1);
        lamML(t1,nsim) = MML.m(t1);
        lamTA(t1,nsim) = TATMEQ.m(t1);
        lamTL(t1,nsim) = TOCEANEQ.m(t1);
        GrossYpath(t1,nsim) = GrossY.l(t1);
        Ypath(t1,nsim) = Y.l(t1);


* simulation step
        Kpath(t1+1,nsim) = K.l(t1+1);
        MATpath(t1+1,nsim) = MAT.l(t1+1);
        MUpath(t1+1,nsim) = MU.l(t1+1);
        MLpath(t1+1,nsim) = ML.l(t1+1);   
        TApath(t1+1,nsim) = TATM.l(t1+1);
        TLpath(t1+1,nsim) = TOCEAN.l(t1+1);
    );
);    

***************
* error checking for the first period
parameter              
    errK
    errMAT
    errMU
    errML
    errTA
    errTL
;

errK = abs(beta/card(nsim) * sum(nsim, (lamK('2',nsim)*((1-DK) + Ypath('2',nsim)*GAMA / Kpath('2',nsim)) +
    lamMAT('2',nsim)*(SIGMA('2')*(1-MIUpath('2',nsim))* GrossYpath('2',nsim))*GAMA / Kpath('2',nsim)) / lamK('1',nsim)) - 1);
errMAT = abs(beta/card(nsim) * sum(nsim, (lamMAT('2',nsim)*b11 + lamMU('2',nsim)*b12 +
    lamTA('2',nsim)*C1*(FCO22X/(MATpath('2',nsim)*log(2)))) / lamMAT('1',nsim)) - 1);  
errMU = abs(beta/card(nsim) * sum(nsim, (lamMAT('2',nsim)*b21 + lamMU('2',nsim)*b22 + lamML('2',nsim)*b23) / lamMU('1',nsim)) - 1);
errML = abs(beta/card(nsim) * sum(nsim, (lamMU('2',nsim)*b32 + lamML('2',nsim)*b33) / lamML('1',nsim)) - 1);
errTA = abs(beta/card(nsim) * sum(nsim, (lamTA('2',nsim)*(1-C1*LAM-C1*C3) +
    lamTL('2',nsim)*C4 + lamK('2',nsim)*(a1 + 2*a2*TApath('2',nsim))*
    ((-Ypath('2',nsim))/(1+a1*TApath('2',nsim) + a2*sqr(TApath('2',nsim))))) / lamTA('1',nsim)) - 1);
errTL = abs(beta/card(nsim) * sum(nsim, (lamTA('2',nsim)*C1*C3 + lamTL('2',nsim)*(1-C4)) / lamTL('1',nsim)) - 1);

display errK, errMAT, errMU, errML, errTA, errTL;

***************
* Output solutions 

* save simulated paths for error checking
execute_unload 'sim_paths_SCEQ_DSICE_TFP', thetapath, Kpath, Kpath, MATpath, MUpath, MLpath, TApath, TLpath,
    Cpath, miupath, lamK, lamMAT, lamMU, lamML, lamTA, lamTL;
    
* save the different paths for simulation by other software like Matlab
file SCEQ_DSICE_TFP_sol;
put SCEQ_DSICE_TFP_sol;
SCEQ_DSICE_TFP_sol.nw = 12;
SCEQ_DSICE_TFP_sol.nr = 2;
SCEQ_DSICE_TFP_sol.nz = 1e-15;
loop(nsim,
  loop(t1$(ord(t1)<=Tstar),
    put t1.tl:4:0;
    put thetapath(t1,nsim):13:5;    
    put Kpath(t1,nsim):13:5;
    put MATpath(t1,nsim):13:5;
    put MUpath(t1,nsim):13:5;
    put MLpath(t1,nsim):13:5;
    put TApath(t1,nsim):13:5;
    put TLpath(t1,nsim):13:5;
    put miupath(t1,nsim):13:5;
    put Cpath(t1,nsim):13:5;
    put opt_SCC(t1,nsim):13:5;
    put opt_tax(t1,nsim):13:5;        
    put /;
  );
);


* display the running time in minutes
scalar elapsed;
elapsed = (jnow - starttime)*24*60;
display elapsed;

