** This is an illustrative code of SCEQ to solve a simple 
** optimal growth problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

scalar starttime;
starttime = jnow;

* model parameters

set t indices of periods /1*51/;
alias(t,tt);
set nsim /1*167/;

parameters   
    beta        discount factor / 0.96 /
    gamma       elasticity for consumption / 2 /
    alpha       production function parameter / 0.3 /
    delta       capital stock depreciation  / 0.1 /
    rho         persistence parameter for TFP growth /0.95/
    sigma       standard deviation of shock / 0.02 /
    theta0      initial value of shock / 1 /
    k0          initial capital / 1 /
    theta(t)    shock to total factor of productivity (TFP)
    A           deterministic productivity 
    s           starting period
    DT          number of periods for truncation / 30 /
    Tstar       number of periods of interest / 20 /
;

A = (1 - (1-delta)*beta) / (alpha * beta);
display A;

**------------------------------------------------------------

options limrow = 0, limcol = 0;
option reslim = 10000;
option iterlim = 10000;
option solprint = off;
option nlp = conopt;

variables 
    obj     objective value
    K(t)    capital
    C(t)    consumption    
;

equations
objfun                  object function
TransitionCapital(t)    next period capital
;

objfun..
obj =e= sum(t$(ord(t)>=s and ord(t)<s+DT), beta**(ord(t)-s) * (C(t)**(1-gamma))/(1-gamma)) +
    sum(t$(ord(t)=s+DT), beta**(ord(t)-s) * ((A*(K(t)**alpha)-delta*K(t))**(1-gamma))/(1-gamma))/(1-beta);

TransitionCapital(t)$(ord(t)>=s and ord(t)<s+DT)..
K(t+1) =e= (1-delta)*K(t) + theta(t)*A*(K(t)**alpha) - C(t);


model growth / all /;

* initial guess
K.l(t) = 1;
C.l(t) = A-delta;

* bounds
C.lo(t) = 0.01;
K.lo(t) = 0.01;

parameter 
    Cpath(tt,nsim) simulated consumption paths
    Kpath(tt,nsim) simulated capital paths
    thetapath(tt,nsim) simulated TFP paths
    lampath(tt,nsim)    shadow prices
;
