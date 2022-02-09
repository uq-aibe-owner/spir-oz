# SPIR-Oz
A Simple, Powerful, Intersectoral and Regional model of Australia (SPIR-Oz: pronounced as in the name Spyros) inspired by the Australian Treasury's Intersectoral Model (TIM). Spyros is an abbreviation of Spyridon (Σπυρίδων), which, in ancient Greek, means "basket used to carry seeds". 

## Purpose
With this model we aim to help policy makers sow the seeds of future growth and prosperity in Australia. We aim to answer questions such as: what is the present value of an Australia-wide net-zero emissions target for 2050? What carbon price/tax will achieve it? And, on a regional and sectoral basis, what is the additional value of a Queensland-specific net-zero target for 2050?

## Model in brief
SPIR-Oz is a modern intersectoral and inter-regional model with a supply-chain network. It is an adaptation of the simple but powerful solution method of Cai and Judd (2021), the python code of Scheidegger and Bilionis (2019) and the  a intersectoral framework similar to Atalay (2017). We estimate and calibrate economic parameters for Australia, Queensland and regions within Queensland in particular. 

## Model in more detail
The result is a fully intertemporal model of the Australian economy that is smaller, yet deeper, than TIM. In contrast with TIM, our model accommodates uncertainty, is regional and has a global solution method. That is, instead of log-linearising around the non-stochastic steady state, we generate a large number sample paths using the Stochastic Certainty-Equivalent approximation of Cai and Judd to obtain statistical moments of the full solution. In contrast with Cai--Judd, our model involves multiple industries and our code is fully open-source (in python) and builds on Scheidegger and Bilionis (2019). Since we are interested in regional transition in Queensland across industries, we build in the functional forms for inter-industry capital allocations and production of Atalay (2017). Finally, a novel feature of our model is the groupoid or Jacobi property for inter-sectoral/regional savings. Without any loss of generality, this ensures the model dimension is linear rather than quadratic in the number of sectors and regions.
