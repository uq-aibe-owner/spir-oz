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

scalar starttime;
starttime = jnow;


SET t          Time periods                     / 1*601 / 
    nsim       number of simulation paths       / 1*167 /;
alias(t1, t);

parameter s    starting period
    Tstar       number of periods of interest / 100 /
;


SCALARS

 gamma elasticity of marginal Welfare of consumption / 1.45 /

** Preferences
 beta    discount factor   / .985    /

** Population and technology
 POP0     2005 world population millions                  /6514     /
 GPOP0    Growth rate of population per year              /.035      /
 POPASYM  Asymptotic population                           / 8600    /
 A0       Initial level of total factor productivity      /.02722   /
 GA0      Initial growth rate for technology per year     /.0092      /
 DELA     Decline rate of technol change per year         /.001     /
 DK       Depreciation rate on capital per year           /.100     /
 GAMA     Capital elasticity in production function       /.300     /
 K0       2005 value capital trill 2005 US dollars        /137.     /

** Emissions
 SIG0     CO2-equivalent emissions-GNP ratio 2005         /.13418    /
 GSIGMA   Initial growth of sigma per year                /-.00730    /
 DSIG     Decline rate of decarbonization per year        /.003   /
 ELAND0   Carbon emissions from land 2005(GtC per year)   / 1.1000  /

** Carbon cycle
 MAT2000  Concentration in atmosphere 2005 (GtC)          /808.9   /
 MU2000   Concentration in upper strata 2005 (GtC)        /1255     /
 ML2000   Concentration in lower strata 2005 (GtC)        /18365    /
 b12      Carbon cycle transition matrix per year         /0.01908349 /
 b23      Carbon cycle transition matrix per year         /0.005405052     /

** Climate model
 T2XCO2   Equilibrium temp impact of CO2 doubling oC      / 3 /
 FEX0     Estimate of 2000 forcings of non-CO2 GHG        / -.06   /
 FEX1     Estimate of 2100 forcings of non-CO2 GHG        / 0.30   /
 TOCEAN0  2000 lower strat. temp change (C) from 1900     /.0068   /
 TATM0    2000 atmospheric temp change (C)from 1900       /.7307   /
 C1       Climate-equation coefficient for upper level per year    /.03713025    /
 C3       Transfer coeffic upper to lower stratum per year        /.2767396    /
 C4       Transfer coeffic for lower level per year                /.004801917    /
 FCO22X   Estimated forcings of equilibrium co2 doubling  /3.8     /

** Climate damage parameters calibrated for quadratic at 2.5 C for 2105
 A1       Damage intercept                                / 0.00000    /
 A2       Damage quadratic term                           /  0.0028388 /
* A3       Damage exponent                                 / 2.00       /

** Abatement cost
 EXPCOST2   Exponent of control cost function               /2.8   /
 PBACK      Cost of backstop 2005 000$ per tC 2005          /1.17  /
 BACKRAT    Ratio initial to final backstop cost            / 2    /
 GBACK      Initial cost decline backstop pc per year       /.005   /
 LIMMIU     Upper limit on control rate                     / 1    /

** Availability of fossil fuels
 FOSSLIM  Maximum cumulative extraction fossil fuels         / 6000  /
;

SET TLAST(T);
TLAST(T)  = YES$(ORD(T) EQ CARD(T));


PARAMETERS
  LAM           Climate model parameter
  L(T)          Level of population and labor
  AL(T)         Level of total factor productivity
  TFP(t)
  SIGMA(T)      CO2-equivalent-emissions output ratio
  FORCOTH(T)    Exogenous forcing for other greenhouse gases
  ETREE(T)      Emissions from deforestation
  cost1(t)      Adjusted cost for backstop
 ;

* Important parameters for the model
LAM     = FCO22X/ T2XCO2;
L(T) = popasym  - (popasym-POP0)*exp(-GPOP0*(ORD(T)-1));
AL(T) = A0*exp(GA0*(1-exp(-DELA*(ORD(T)-1)))/DELA);
SIGMA(T) = SIG0*exp(GSIGMA*(1-exp(-DSIG*(ORD(T)-1)))/DSIG);
FORCOTH(T)= FEX0+ 0.01*(FEX1-FEX0)*(ORD(T)-1)$((ORD(T)-1)<=100)+ 0.36$((ORD(T)-1)>100);
ETREE(T) = ELAND0*exp(-0.01*(ORD(T)-1));
cost1(T) = (PBACK*SIGMA(T)*(1+exp(-GBACK*(ORD(T)-1)))/EXPCOST2) * ((BACKRAT-1)/BACKRAT);

parameters
  b11          Carbon cycle transition matrix
  b21          Carbon cycle transition matrix
  b22          Carbon cycle transition matrix
  b32          Carbon cycle transition matrix
  b33          Carbon cycle transition matrix
;

b11 = 1 - b12;
b21 = 587.473*b12/1143.894;
b22 = 1 - b21 - b23;
b32 = 1143.894*b23/18340;
b33 = 1 - b32 ;



*****************************************************************

VARIABLES
 K(T)            Capital stock trillions US dollars
 MAT(T)          Carbon concentration in atmosphere GtC
 MU(T)           Carbon concentration in shallow oceans Gtc
 ML(T)           Carbon concentration in lower oceans GtC
 TATM(T)         Temperature of atmosphere in degrees C
 TOCEAN(T)       Temperature of lower oceans degrees C
 E(T)            CO2-equivalent emissions GtC
 C(T)            Consumption trillions US dollars
 MIU(T)          Emission control rate GHGs
 GrossY(T)
 Y(T)
 Welfare;

POSITIVE VARIABLES K, MAT, MU, ML, TATM, C, MIU, E;


*****************************************************************

EQUATIONS

 KK(T)            Capital balance equation
 MMAT(T)          Atmospheric concentration equation
 MMU(T)           Shallow ocean concentration
 MML(T)           Lower ocean concentration
 TATMEQ(T)        Temperature-climate equation for atmosphere
 TOCEANEQ(T)      Temperature-climate equation for lower oceans
 
 GrossOutputEq(t)
 NetOutputEq(t)
 
 ObjFun             Objective function
;

** Equations of the model

KK(T)$(ord(t)>=s and ord(t)<card(t))..          K(T+1)      =L= (1-DK)*K(T) + Y(T)  - C(T);
MMAT(T)$(ord(t)>=s and ord(t)<card(t))..        MAT(T+1)    =E= MAT(T)*b11+MU(T)*b21 + SIGMA(T)*(1-MIU(T))*GrossY(t) + ETREE(T);
MML(T)$(ord(t)>=s and ord(t)<card(t))..         ML(T+1)     =E= ML(T)*b33+b23*MU(T);
MMU(T)$(ord(t)>=s and ord(t)<card(t))..         MU(T+1)     =E= MAT(T)*b12+MU(T)*b22+ML(T)*b32;
TATMEQ(T)$(ord(t)>=s and ord(t)<card(t))..      TATM(T+1)   =E= TATM(t)+C1*(FCO22X*(log(MAT(T)/596.4)/log(2))+FORCOTH(T) - LAM*TATM(t)-C3*(TATM(t)-TOCEAN(t)));
TOCEANEQ(T)$(ord(t)>=s and ord(t)<card(t))..    TOCEAN(T+1) =E= TOCEAN(T)+C4*(TATM(T)-TOCEAN(T));

GrossOutputEq(t)$(ord(t)>=s and ord(t)<card(t))..   GrossY(t) =e= TFP(t)*L(T)**(1-GAMA)*K(T)**GAMA;
NetOutputEq(t)$(ord(t)>=s and ord(t)<card(t))..     Y(t) =e= ( 1 - cost1(T)*(MIU(T)**EXPcost2) )*GrossY(t) / (1+A1*TATM(T)+ A2*TATM(T)*TATM(T));

ObjFun..             Welfare =E= sum( T$(ord(T)<card(T) and ord(t)>=s), (((C(T)/L(T))**(1-gamma))/(1-gamma)) * (beta**(ord(t)-s)*L(T)/POPASYM) );

*******************************
**  Upper and Lower Bounds

K.lo(T)         = 100;
MAT.lo(T)       = 10;
MU.lo(t)        = 100;
ML.lo(t)        = 1000;
TOCEAN.up(T)    = 20;
TOCEAN.lo(T)    = -1;
TATM.lo(t)      = 0;
TATM.up(t)      = 20;

C.lo(T)         = 20;
miu.lo(t)       = 0.001;
miu.up(t)       = LIMMIU;


**********************************

** Solution options
option iterlim = 99900;
option reslim = 99999;
option solprint = off;
option limrow = 0;
option limcol = 0;
option nlp = conopt;


model DSICE_TFP / all /;


set j index of TFP /1*3/;
alias(j, j2, j3);

parameter TFPshock(j)
    tranProbs(j,j2)
    probs(t,j)
;


TFPshock('1') = 0.9;
TFPshock('2') = 1;
TFPshock('3') = 1.1;

tranProbs(j,j2) = 0;
tranProbs('1','1') = 0.8;
tranProbs('2','1') = 0.2;
tranProbs('1','2') = 0.2;
tranProbs('2','2') = 0.6;
tranProbs('3','2') = 0.2;
tranProbs('2','3') = 0.2;
tranProbs('3','3') = 0.8;

    
parameter
    thetapath(t,nsim)
    Kpath(t,nsim)
    MATpath(t,nsim)
    MUpath(t,nsim)
    MLpath(t,nsim)    
    TApath(t,nsim)
    TLpath(t,nsim)
    miupath(t,nsim)
    Cpath(t,nsim)
    lamK(t,nsim)
    lamMAT(t,nsim)
    lamMU(t,nsim)
    lamML(t,nsim)
    lamTA(t,nsim)
    lamTL(t,nsim)
    GrossYpath(t1,nsim)
    Ypath(t1,nsim)        
;


