# SPIR-Oz
A Simple, Powerful, Intersectoral and Regional model of Australia (SPIR-Oz pronounced Spyros). Spyros is an abbreviation of Spyridon (Σπυρίδων), which, in ancient Greek, means "basket used to carry seeds": in this case the seeds of future growth and prosperity in Australia.

SPIR-Oz is an adaptation of the Stochastic Certainty-Equivalent solution method of Cai and Judd (2021), the python code of Scheidegger and Bilionis (2019) and the  a intersectoral framework similar to Atalay (2017). We estimate and calibrate economic parameters for Australia, Queensland and regions within Queensland in particular. 

We aim to answer questions relating to 2050 net-zero targets. For instance, what is the present value of a Qld net-zero target beyond that of Australia?

The result is a fully intertemporal model of the Australian economy that is smaller, yet deeper, than the Australian Treasury's Intersectoral Model (TIM). In contrast with TIM, our solution method is global; involves looking-ahead multiple-periods; moreover it accommodates uncertainty. That is, instead of log-linearising around the non-stochastic steady state, we generate a large number sample paths using the Certainty-Equivalent method of Cai and Judd to obtain the full solution. In contrast with Cai--Judd, our code is fully open-source (in python) and builds on Scheidegger and Bilionis (2019). Since we are interested in regional transition in Queensland across industries, we build in the functional forms for inter-industry capital allocations and production of Atalay (2017).
