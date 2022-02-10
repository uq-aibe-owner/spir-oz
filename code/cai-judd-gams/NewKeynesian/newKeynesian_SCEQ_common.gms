*----------------------------------------------------------------------
* This program solves the New Keynesian DSGE model with zero lower bound using SCEQ
*
* If using material from this code, the user should cite the following paper:
* Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
*   Approximation Method for Dynamic Stochastic Problems. Working Paper.
*----------------------------------------------------------------------

scalar starttime;
starttime = jnow;

set t time /1*221/;
alias(tt,t);
set nsim /1*167/;

scalars
eta	    labor supply elasticity / 1 /
rho 	persistence of discount rate shock / 0.8 /
sigma   standard deviation of discount rate shock / 0.005 /
alpha 	elasticity of substitution across intermediate goods / 6 /
theta   Calvo parameter / 0.9 /
sg	    ratio of govenment spending to output / 0.2 /
phi_pi  response to inflation (monetary policy rule) / 2.5 /
phi_y   response to output (monetary policy rule) / 0.25 /
pi_ss   steady state inflation / 1.005 /
DT      number of periods for optimization in SCEQ / 50 /
Tstar   number of periods of interest / 20 /
v0      initial value of v
beta0   initial value of beta / 0.994 /
betass  steady state beta / 0.994 /
;

parameters
s   starting period for SCEQ
rss	steady state interest rate
chi2ss
chi1ss
qss
zss
lss
css
yss	steady state output
vss	steady state of v
;

rss = pi_ss/betass-1;
chi2ss = 1/((1-sg)*(1-theta*betass*pi_ss**(alpha-1)));
qss = ((1-theta*pi_ss**(alpha-1))/(1-theta))**(1/(1-alpha));
chi1ss = chi2ss*qss*(alpha-1)/alpha;
vss = (1-theta)*qss**(-alpha) / (1-theta*pi_ss**alpha);
yss = (chi1ss*(1-theta*betass*pi_ss**alpha)/(vss**eta))**(1/(1+eta));
zss = rss;
lss = vss*yss;
css = (1-sg)*yss;

v0 = vss;

display rss, vss, yss, lss, css, chi1ss, chi2ss, qss;

*******************************************************
* define model

parameter betas(t) discount;

Variables
z(t)
obj
;

Positive variables
v(t)
y(t)
pi_t(t)
q(t)
chi1(t)
chi2(t)
;

Equations
Objective
StateFun(t)
NotionalInterest(t)
QFun(t)
Chi1Fun(t)
Chi2Fun(t)
Chi1Chi2(t)
Euler(t)
TermChi1(t)
TermChi2(t)
TermY(t)
Termv(t)
;

Objective..
obj =e= 1;

StateFun(t)$(ord(t)>=s and ord(t)<s+DT)..
v(t+1) =e= (1-theta)*power(q(t),-alpha) + theta*power(pi_t(t),alpha)*v(t);

NotionalInterest(t)$(ord(t)>=s and ord(t)<s+DT)..
z(t) =e= (1+rss)*(pi_t(t)/pi_ss)**phi_pi*(y(t)/yss)**phi_y - 1;

QFun(t)$(ord(t)>=s and ord(t)<s+DT)..
1 =e= power(q(t),alpha-1) * (1-theta*power(pi_t(t),alpha-1))/(1-theta);

Chi1Fun(t)$(ord(t)>=s and ord(t)<s+DT)..
chi1(t) =e= power(y(t),1+eta)*power(v(t+1),eta) + theta*betas(t+1)*power(pi_t(t+1),alpha)*chi1(t+1);

Chi2Fun(t)$(ord(t)>=s and ord(t)<s+DT)..
chi2(t) =e= 1/(1-sg) + theta*betas(t+1)*power(pi_t(t+1),alpha-1)*chi2(t+1);

Chi1Chi2(t)$(ord(t)>=s and ord(t)<s+DT)..
chi1(t) =e= q(t)*chi2(t)*(alpha-1)/alpha;

Euler(t)$(ord(t)>=s and ord(t)<s+DT)..
pi_t(t+1)*y(t+1) =e= betas(t+1)*(1+max(0,z(t)))*y(t);

TermChi1(t)$(ord(t)=s+DT)..
chi1(t) =e= chi1ss;

TermChi2(t)$(ord(t)=s+DT)..
chi2(t) =e= chi2ss;

TermY(t)$(ord(t)=s+DT)..
y(t) =e= yss;

Termv(t)$(ord(t)=s+DT)..
v(t) =e= vss;

* bounds
y.lo(t) = 0.001;
v.lo(t) = 0.001;
pi_t.lo(t) = 0.001;
q.lo(t) = 0.001;

* initial guess
v.l(t) = vss;
pi_t.l(t) = pi_ss;
y.l(t) = yss;
z.l(t) = (1+rss)*(pi_t.l(t)/pi_ss)**phi_pi*(y.l(t)/yss)**phi_y - 1;
q.l(t) = ((1-theta*pi_t.l(t)**(alpha-1))/(1-theta))**(1/(1-alpha));
chi2.l(t) = chi2ss;
chi1.l(t) = q.l(t)*chi2.l(t)*(alpha-1)/alpha;
obj.l = 0;

Option limrow=0,limcol=0,solprint=off;
OPTION ITERLIM = 500000;
OPTION RESLIM = 500000;
OPTION DOMLIM = 1000;
OPTION DNLP= conopt;
option decimals = 7;

model NewKeynesian /all/;

parameter
    betapath(tt,nsim) simulated paths of beta
    vpath(tt,nsim) simulated paths of v    
    chi1path(tt,nsim) simulated paths of chi1
    chi2path(tt,nsim) simulated paths of chi2
    ypath(tt,nsim) simulated paths of y
    pipath(tt,nsim) simulated paths of pi_t
    qpath(tt,nsim) simulated paths of q
    zpath(tt,nsim) simulated paths of z
;


