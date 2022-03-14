** This is a code of SCEQ to solve a multi-country real business cycle problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

scalar starttime;
starttime = jnow;
    
set j countries /1*3/;
set t time /1*71/;
alias(tt,t);

* number of different paths + 1 (the extra one is for error checking at the last period of interest)
set npath /1*10/;

parameters
beta        discount rate /0.99/
alpha       capital cost share /0.33/
delta       capital stock depreciation /0.025/
phi         adjustment cost parameter /0.5/
gamma       intertemporal elasticity of substitution /0.5/
gammahat    utility parameter
eta         Frisch elasticity of labor supply /0.5/
etahat      utility parameter
A           technology parameter
B           relative weight of consumption and leisure
s           starting period
DT          number of periods for optimization in SCEQ / 10 /
Tstar       number of periods of interest 
tau(j)      weight
Imin        lower bound of investment 
k0(j)       initial capital
kmin        smallest capital    / 0.1 /
kmax        largest capital     / 10 /
zeta1       TFP before shock   / 1 /
zeta2       TFP after shock   / 0.95 /
prob1       one period probability of jump of TFP / 0.01 /
probs(t)    probability of jump of TFP
TCS         tail consumption share (of output) / .45 /
;

Tstar = card(npath)-1;

A = (1 - (1-delta)*beta) / (alpha * beta);
gammahat = 1-(1/gamma);
B = (1 - alpha)*A*(A-delta)**(-1/gamma);
etahat = 1+(1/eta);

Imin = 0.9*delta;

tau(j) = 1;
*k0(j) = kmin + (kmax-kmin)*(ord(j)-1)/(card(j)-1);
*k0(j) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(j)-1)/(card(j)-1));
k0(j) = 1

display A, B, etahat, k0;

*************
* define model

Variables
obj objective criterion
Inv(j,t) investment
;

Positive variables
k(j,t) capital stock
c(j,t) consumption
l(j,t) labor supply
;

Equations
objfun Objective function
TransitionCapital(j,t) Law of Motion for Capital Stock
BudgetConstraint(t) budget constraint before jump
TippedBudgetConstraint(t) budget constraint after jump
;

objfun .. 
obj =e= sum(j, tau(j) *
                    sum(t$(ord(t)>=s and ord(t)<s+DT),
                        beta**(ord(t)-s)*((c(j,t)**gammahat)/gammahat - B * (l(j,t)**etahat)/etahat)
                    )
        )
        + sum(j, tau(j) *
                    sum(t$(ord(t)=s+DT),
                        beta**(ord(t)-s)*((( (TCS*A*(k(j,t)**alpha))**gammahat )/gammahat-B)/(1-beta))
                    )
        ) ;

TransitionCapital(j,t)$(ord(t)>=s and ord(t)<s+DT) .. 
k(j,t+1) =e= (1-delta)*k(j,t) + Inv(j,t);

BudgetConstraint(t)$(ord(t)>=s and ord(t)<s+DT) .. 
sum(j, c(j,t) + Inv(j,t) + (phi/2)*k(j,t)*sqr(Inv(j,t)/k(j,t)-delta)) =e= sum(j, (zeta2 + Probs(t)*(zeta1-zeta2))*A*(k(j,t)**alpha) * (l(j,t)**(1-alpha)));

TippedBudgetConstraint(t)$(ord(t)>=s and ord(t)<s+DT) .. 
sum(j, c(j,t) + Inv(j,t) + (phi/2)*k(j,t)*sqr(Inv(j,t)/k(j,t)-delta)) =e= sum(j, zeta2*A*(k(j,t)**alpha) * (l(j,t)**(1-alpha)));

* Bound Constraints
k.lo(j,t) = 0.001;
k.up(j,t) = 1000;
c.lo(j,t) = 0.001;
c.up(j,t) = 1000;
l.lo(j,t) = 0.001;
l.up(j,t) = 1000;
Inv.lo(j,t) = Imin;

* Initial Guess
s = 1;
Inv.l(j,t) = delta;
k.l(j,t) = 1;
l.l(j,t) = 1;
c.l(j,t) = A-delta;
obj.l = sum(j, tau(j) * sum(t$(ord(t)>=s and ord(t)<s+DT), beta**(ord(t)-s)*((c.l(j,t)**gammahat)/gammahat - B * (l.l(j,t)**etahat)/etahat))) +
  sum(j, tau(j) * sum(t$(ord(t)=s+DT), beta**(ord(t)-s)*((((0.75*A*(k.l(j,t)**alpha))**gammahat)/gammahat-B)/(1-beta))));


options limrow = 0, limcol = 0;
option reslim = 10000;
option iterlim = 10000;
option solprint = off;
option nlp = conopt;

model busc /objfun, TransitionCapital, BudgetConstraint/;
model busc_tipped /objfun, TransitionCapital, TippedBudgetConstraint/;

