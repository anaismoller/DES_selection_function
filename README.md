How to get the Spectroscopic Selection function 
using method: data vs simulations?

A. Moller 2017/12/27


0. run simulations and fits 
	S. Hinton code
		RUN_SIM_SPECEFF_DES.sh

1. call:
	python run_SpecSelFunction_Moller.py

	you can choose the following options:

	--data
   	 Data file (ligthcurve fits: FITRES file)
    	default= $SCRATCH_FITDIR/DES3YR_v1_freeze/DES3YR_DES_COMBINED_FITS/DES3YR_DES_COMBINED_FITS/FITOPT000.FITRES.gz

	--sim	
    	Sim file (ligthcurve fits: FITRES file)
    	default= $SCRATCH_FITDIR/DES3YR_v2/DES3YR_DES_SPECEFF/DES3YR_DES_SPECEFF_AMC11/FITOPT000.FITRES.gz

	--nameout
   	 Out name for spectroscopic selection function (useful for C10,G11 distinction)
    	default=./SPEC_EFF_Moller.DAT

	--plots 
   	 When called it plots data, simualtion parameter distributions

	--path_plots 
   	 Path to save plots
    	default= ./plots

	—onlybias
	will only plot, no selection function computed

How to get the bias? computing for AMG10, AMc11, CdAndrea v 2.3
0. run simulations and fits
	modified S. Hinton code
		RUN_SIM_SPECEFF_BIAS_DES.sh
		
