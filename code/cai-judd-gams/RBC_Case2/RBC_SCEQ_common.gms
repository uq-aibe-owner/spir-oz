** This is a code of SCEQ to solve a multi-country real business cycle problem
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

scalar starttime;
starttime = jnow;
    
set r regions /1*2/;
alias(rr, r);
set t time /1*71/;
alias(tt,t);

set j sectors /1*2/;
alias(i, j);
* number of different paths + 1 (the extra one is for error checking at the last period of interest)
set npath /1*10/;

parameters
beta        discount rate /0.98/
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
tau(r)      weight
Imin        lower bound of investment 
k0(r)       initial capital
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

*the following is the vector of population weights: it will enter the objective and determine demand
tau(r) = 1;

*the following is initial kapital: it will vary across rions
*k0(r) = kmin + (kmax-kmin)*(ord(r)-1)/(card(r)-1);
*k0(r) = exp(log(kmin) + (log(kmax)-log(kmin))*(ord(r)-1)/(card(r)-1));
k0(r, sec) = 1

display A, B, etahat, k0;

*************
* define model

Variables
obj objective criterion
Inv(r, sec,t) investment
;

Positive variables

c(r, sec, t) consumption
c_sectors(r, t) consumption aggregate (across sectors)
k(r, sec, t) kapital stock
k_sectors(r, t) kapital aggregate (across sectors)
l(r, sec, t) labor supply
;

Equations
objfun Objective function
TransitionCapital(r, sec, t) Law of Motion for Capital Stock
BudgetConstraint(sec, t) budget constraint before jump
TippedBudgetConstraint(sec, t) budget constraint after jump
;

objfun .. 
obj =e= sum(r, tau(r) *
                    sum(t$(ord(t)>=s and ord(t)<s+DT),
                        beta**(ord(t)-s)*((c(r, sec, t) )**gammahat)/gammahat - B * (l(r, sec, t)**etahat)/etahat)
                    )
        )
        + sum(r, tau(r) *
                    sum(t$(ord(t)=s+DT),
                        beta**(ord(t)-s)*((( (TCS*A*(k(r,t)**alpha))**gammahat )/gammahat-B)/(1-beta))
                    )
        ) ;

TransitionCapital(r,t)$(ord(t)>=s and ord(t)<s+DT) .. 
k(r,t+1) =e= (1-delta)*k(r,t) + Inv(r,t);

BudgetConstraint(t)$(ord(t)>=s and ord(t)<s+DT) .. 
sum(r, c(r,t) + Inv(r,t) + (phi/2)*k(r,t)*sqr(Inv(r,t)/k(r,t)-delta)) =e= sum(r, (zeta2 + Probs(t)*(zeta1-zeta2))*A*(k(r,t)**alpha) * (l(r,t)**(1-alpha)));

TippedBudgetConstraint(t)$(ord(t)>=s and ord(t)<s+DT) .. 
sum(r, c(r,t) + Inv(r,t) + (phi/2)*k(r,t)*sqr(Inv(r,t)/k(r,t)-delta)) =e= sum(r, zeta2*A*(k(r,t)**alpha) * (l(r,t)**(1-alpha)));

* Bound Constraints
k.lo(r,t) = 0.001;
k.up(r,t) = 1000;
c.lo(r,t) = 0.001;
c.up(r,t) = 1000;
l.lo(r,t) = 0.001;
l.up(r,t) = 1000;
Inv.lo(r,t) = Imin;

* Initial Guess
s = 1;
Inv.l(r,t) = delta;
k.l(r,t) = 1;
l.l(r,t) = 1;
c.l(r,t) = A-delta;
obj.l = sum(r, tau(r) * sum(t$(ord(t)>=s and ord(t)<s+DT), beta**(ord(t)-s)*((c.l(r,t)**gammahat)/gammahat - B * (l.l(r,t)**etahat)/etahat))) +
  sum(r, tau(r) * sum(t$(ord(t)=s+DT), beta**(ord(t)-s)*((((0.75*A*(k.l(r,t)**alpha))**gammahat)/gammahat-B)/(1-beta))));


options limrow = 0, limcol = 0;
option reslim = 10000;
option iterlim = 10000;
option solprint = off;
option nlp = conopt;

model busc /objfun, TransitionCapital, BudgetConstraint/;
model busc_tipped /objfun, TransitionCapital, TippedBudgetConstraint/;

