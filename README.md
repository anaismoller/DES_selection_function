DES spectroscopic selection function
modeling the ratio of spectroscopically classified SNe Ia versus expected SNe Ia (simulations)

A. Moller 2018
______________________________________________________
Updates:
- 2018/08/07: errors in ratio data/sim are not binomial nor gaussian so we shouldnt use gaussian approx. in emcee model fitting.
	implementing poisson distribution for ndata.
	Beware, option plateau is not implemented in this version, however, it is not needed to get a reasonable fit

______________________________________________________	

How to get the Spectroscopic Selection function 
using method: data vs simulations?
______________________________________________________

0. run simulations and fits 
	S. Hinton code
		RUN_SIM_SPECEFF_DES.sh

1. call:
	python run_SpecSelFunction_Moller.py

	you can choose the following options:

	--data
   	 Data file (ligthcurve fits: FITRES file)
    	

	--sim	
    	Sim file (ligthcurve fits: FITRES file)
    	

	--nameout
   	 Out name for spectroscopic selection function (useful for C10,G11 distinction)
    	default=./SPEC_EFF_Moller.DAT

	--plots 
   	 When called it plots data, simulation parameter distributions

	--path_plots 
   	 Path to save plots
    	default= ./plots

	--onlybias
	will only plot, no selection function computed

	--plateau
	Activating plateau for sigmoid function at mag <20.7

How to get the bias? computing for AMG10, AMc11, CdAndrea v 2.3
0. run simulations and fits
	modified S. Hinton code
		RUN_SIM_SPECEFF_BIAS_DES.sh
1. call for example
	python run_SpecSelFunction_Moller.py --sim FITOPT000.FITRES.gz --onlybias --path_plots bias_AMG10 	
