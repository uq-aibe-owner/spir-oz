function [err,runtime] = SCEQ_growth()

% This is an illustrative code of SCEQ to solve a simple 
% optimal growth problem
%
% If using material from this code, the user should cite the following paper:
% Cai, Y., and K.L. Judd (2021). A Simple but Powerful Simulated Certainty Equivalent
%   Approximation Method for Dynamic Stochastic Problems. Working Paper.
%----------------------------------------------------------------------

% number of periods for one optimization problem (i.e., Delta_s)
T = 30;

% time of interest
Tstar = 20;

% parameter values
beta=0.96;
delta=0.1;
gamma=2;
alpha=0.3;
rho = 0.95;
sigma = 0.02;
A = (1 - (1-delta)*beta) / (alpha * beta);
betas = beta.^(0:T)';
betas(T+1) = betas(T+1)/(1-beta);

% initial state
k0=1;
theta0=1;

% bounds of variables
lb = 0.001*ones(2*T+1,1);

options = optimoptions('fmincon','MaxFunctionEvaluations',100000,...
    'Display','off','Algorithm','sqp');

% number of simulation paths
nsim = 1000;

ksols = zeros(Tstar+1,nsim);
csols = zeros(Tstar,nsim);
thetas = zeros(Tstar+1,nsim);

t0 = datetime('now');

parfor i = 1:nsim
    ksol1 = zeros(Tstar+1,1);
    csol1 = zeros(Tstar,1);
    theta1 = zeros(Tstar+1,1);

% starting at the initial state    
    ksol1(1) = k0;
    theta1(1) = theta0;
    
% variables: c(1),c(2),...,c(T), k(1),k(2),...,k(T+1)    
    xsol = ones(2*T+1,1);
    xsol(1:T) = A-delta;
  
% Initialization step for simulating shocks    
% set the seed for the MATLAB random number generator as i (the results
% can be replicated with the same seed)
    rng(i);    
    shocks = randn(Tstar,1);
    
    for s = 1:Tstar
        % optimization step
        theta = theta1(s).^(rho.^(0:T-1)');    
        x0 = xsol;
        xsol = fmincon(@(x) growth_Objfun(x,betas,T,gamma,alpha,A,delta),...
            x0,[],[],[],[],lb,[],...
            @(x) growth_ConFun(x,T,delta,alpha,A,ksol1(s),theta),options);
        csol1(s) = xsol(1);
        
        % simulation step
        ksol1(s+1) = xsol(T+2);
        theta1(s+1) = theta1(s)^rho * exp(sigma*shocks(s));
    end
    
    csols(:,i) = csol1;
    ksols(:,i) = ksol1;
    thetas(:,i) = theta1;    
end

t1 = datetime('now');
runtime = minutes(t1-t0);

% Euler error at the first period
err = abs(1 - beta*mean(csols(2,:).^(-gamma)./csols(1,:).^(-gamma) .* ...
    (1-delta+alpha*A*thetas(2,:).*ksols(2,:).^(alpha-1))));

end


function y=growth_Objfun(x,betas,T,gamma,alpha,A,delta)
% variables: c(1),c(2),...,c(T), k(1),k(2),...,k(T+1)

Cterm = A*x(end)^alpha-delta*x(end);
y = -(sum(betas(1:T) .* x(1:T).^(1-gamma)) + betas(T+1)*Cterm^(1-gamma))/(1-gamma);

end


function [cineq,ceq]=growth_ConFun(x,T,delta,alpha,A,k0,theta)
% variables: c(1),c(2),...,c(T), k(1),k(2),...,k(T+1)]

cineq = [];

% transition law for k
ceq(1:T) = x(T+2:end) - ((1-delta)*x(T+1:2*T) + A*theta.*x(T+1:2*T).^alpha - x(1:T));

% initial capital
ceq(T+1) = x(T+1) - k0;

end


