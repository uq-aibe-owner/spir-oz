** This is a code of SCEQ to solve a multi-country real business cycle problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

scalar starttime;
starttime = jnow;
    
set reg regions /1*2/;
alias(r, reg);
set t time /1*71/;
alias(tt,t);

set sec sectors /1*2/;
alias(j, sec);
* number of different paths + 1 (the extra one is for error checking at the last period of interest)
set npath /1*10/;

parameters
beta        discount rate /0.01/
alpha       capital cost share /0.33/
delta       capital stock depreciation /0.5/
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
tau(reg)      weight
Imin        lower bound of investment 
k0(reg)       initial capital
kmin        smallest capital    / 0.1 /
kmax        largest capital     / 10 /
zeta1       TFP before shock   / 1 /
zeta2       TFP after shock   / 0.95 /
prob1       one period probability of regump of TFP / 0.01 /
probs(t)    probability of jump of TFP
TCS         tail consumption share (of output) / .45 /
;

Tstar = card(npath)-1;

A = (1 - (1-delta)*beta) / (alpha * beta);
gammahat = 1-(1/gamma);
B = (1 - alpha)*A*(A-delta)**(-1/gamma);
etahat = 1+(1/eta);

Imin = 0.9*delta;

*the following is the vector of population weights: it will enter the objective and determine demand
tau(reg) = 1;

*the following is initial kapital: it will vary across regions
*k0(reg) = kmin + (kmax-kmin)*(ord(reg)-1)/(card(reg)-1);
*k0(reg) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(reg)-1)/(card(reg)-1));
k0(reg) = 1

display A, B, etahat, k0;

*************
* define model

Variables
obj objective criterion
Inv(reg,t) investment
;

Positive variables
k(reg,t) capital stock
c(reg,t) consumption
l(reg,t) labor supply
;

Equations
objfun Obregective function
TransitionCapital(reg,t) Law of Motion for Capital Stock
BudgetConstraint(t) budget constraint before regump
TippedBudgetConstraint(t) budget constraint after regump
;

objfun .. 
obj =e= sum(reg, tau(reg) *
                    sum(t$(ord(t)>=s and ord(t)<s+DT),
                        beta**(ord(t)-s)*((c(reg,t)**gammahat)/gammahat - B * (l(reg,t)**etahat)/etahat)
                    )
        )
        + sum(reg, tau(reg) *
                    sum(t$(ord(t)=s+DT),
                        beta**(ord(t)-s)*((( (TCS*A*(k(reg,t)**alpha))**gammahat )/gammahat-B)/(1-beta))
                    )
        ) ;

TransitionCapital(reg,t)$(ord(t)>=s and ord(t)<s+DT) .. 
k(reg,t+1) =e= (1-delta)*k(reg,t) + Inv(reg,t);

BudgetConstraint(t)$(ord(t)>=s and ord(t)<s+DT) .. 
sum(reg, c(reg,t) + Inv(reg,t) + (phi/2)*k(reg,t)*sqr(Inv(reg,t)/k(reg,t)-delta)) =e= sum(reg, (zeta2 + Probs(t)*(zeta1-zeta2))*A*(k(reg,t)**alpha) * (l(reg,t)**(1-alpha)));

TippedBudgetConstraint(t)$(ord(t)>=s and ord(t)<s+DT) .. 
sum(reg, c(reg,t) + Inv(reg,t) + (phi/2)*k(reg,t)*sqr(Inv(reg,t)/k(reg,t)-delta)) =e= sum(reg, zeta2*A*(k(reg,t)**alpha) * (l(reg,t)**(1-alpha)));

* Bound Constraints
k.lo(reg,t) = 0.001;
k.up(reg,t) = 1000;
c.lo(reg,t) = 0.001;
c.up(reg,t) = 1000;
l.lo(reg,t) = 0.001;
l.up(reg,t) = 1000;
Inv.lo(reg,t) = Imin;

* Initial Guess
s = 1;
Inv.l(reg,t) = delta;
k.l(reg,t) = 1;
l.l(reg,t) = 1;
c.l(reg,t) = A-delta;
obj.l = sum(reg, tau(reg) * sum(t$(ord(t)>=s and ord(t)<s+DT), beta**(ord(t)-s)*((c.l(reg,t)**gammahat)/gammahat - B * (l.l(reg,t)**etahat)/etahat))) +
  sum(reg, tau(reg) * sum(t$(ord(t)=s+DT), beta**(ord(t)-s)*((((0.75*A*(k.l(reg,t)**alpha))**gammahat)/gammahat-B)/(1-beta))));


options limrow = 0, limcol = 0;
option reslim = 10000;
option iterlim = 10000;
option solprint = off;
option nlp = conopt;

model busc /objfun, TransitionCapital, BudgetConstraint/;
model busc_tipped /objfun, TransitionCapital, TippedBudgetConstraint/;

