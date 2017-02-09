# DES_spectro_selection_function
Determining the spectroscopic selection function through comparison of data and simulation (SNe Ia) and evaluating the mB bias

Method: comparing DES Y3 spectroscopically confirmed SNe Ia  and simulated SNe Ia (simulation includes detection in pipeline, SNR)

#Part 1: determining the selection function

1. do_selection_function.py
	input: a) fitted light-curves data and simulation (SNANA format)
	!!! Beware: if you are producing the FITRES you need to format the FITRES to eliminate blank spaces and the line 			before VARNAMES must be eliminated.
		b) Chris and Mat's selection function (data driven) for comparison	
	output: plots, csv with division of data/sim by magnitude

2. do_emceee_fit_sigmoid.py
	input: division___.csv from previous code
	output: plots (both normal fits and emcee fitting), emcee fit 

3. do_sel_eff_file.py create file in SNANA format so it can be introduced in the simulations.
	
#Part 2: determining the bias
After the previously determined selection function is applied to a new simulation, it is time to study the color, stretch, redshift distributions and the mB bias we need to correct.
1. do_bias_and_distributions.py: create plots.
	input: FITRES (see formatting above)
