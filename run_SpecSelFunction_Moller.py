import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import utils_emcee_gaussian as mc
import utils_emcee_poisson as mc
import utils_plots as mplot
import os

'''
2018/08/07 A. Moller

Obtain spectroscopic selection efficiency
using ratio data/sim (spec confirmed SNe Ia/simulated Ias)
2018/08/13 
normalizes the simulation to have perfect efficiency at the bright tail
normalization comes from sigmoid fit, beware this makes the +1sigma above 1 eff!

'''


def copy_uncompress(fname):
    import gzip
    import shutil
    thisfname = fname.split('/')[-1].split('.gz')[0]
    with gzip.open(fname, 'rb') as f_in, open(thisfname, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return thisfname


def load_fitres(filepath,validation=False,nsn=-1):
    '''Load light curve fit file and apply cuts    
    Arguments:
        filepath (str) -- Lightcurve fit file with path   
    Returns:
        pandas dataframe 
    '''
    skiprows = 11
    filetoload = pd.read_csv(filepath, index_col=False,comment='#', delimiter=' ',
                             skiprows=skiprows, skipinitialspace=True)
    tmp = filetoload[(filetoload['c'] > -0.3) & (filetoload['c'] < 0.3) & (filetoload['x1'] > -3) & (filetoload['x1']
                                                                                                     < 3) & (filetoload['zHD'] > 0.05) & (filetoload['zHD'] < 0.9) & (filetoload['FITPROB'] > 1E-05)]
    df = tmp.copy()
    if nsn != -1:
        dfout = df[:int(nsn)]
    else:
        dfout = df
    return dfout


def data_sim_division(filt, min_mag, norm_bin, nbins,plots, path_plots):
    '''
    dividing data and sim to obtain the selection function
    '''
    var = 'm0obs_' + filt
    bin_centers, content_division, errors_division, n_data, n_sim = mplot.mag_histos(
        filt, data, sim, norm_bin, min_mag, nbins, plots, path_plots)

    result = {}
    result['x'] = bin_centers
    result['div'] = content_division
    result['err'] = errors_division
    result['n_data'] = n_data
    result['n_sim'] = n_sim
    df = pd.DataFrame(result, columns=['x', 'div', 'err','n_data','n_sim'])

    '''
    Use if you want a table with data/sim i_mag
    '''
    df2 = pd.DataFrame()
    df2['i'] = df['x']
    df2['datasim_ratio'] = df['div']
    df2['error'] = df['err']
    df2['n_sim'] = df['n_sim']
    df2['n_data'] = df['n_data']
    df2.to_csv('datasim_ratio_%s.csv' % filt,index=False,sep=' ')

    return df


if __name__ == "__main__":

    scratch_path = os.environ.get("SCRATCH_FITDIR")

    '''Parse arguments
    '''
    parser = argparse.ArgumentParser(
        description='Selection function data vs simulations')
    parser.add_argument('--data', type=str,
                        default='%s/DES3YR_v1_freeze/DES3YR_DES_COMBINED_FITS/DES3YR_DES_COMBINED_FITS/FITOPT000.FITRES.gz' % (
                            scratch_path),
                        help="Data file (ligthcurve fits: FITRES file)")
    parser.add_argument('--sim', type=str,
                        default='%s/DES3YR_v7/DES3YR_DES_SPECEFF/DES3YR_DES_SPECEFF_AMG10/FITOPT000.FITRES.gz' % (
                            scratch_path),
                        help="Simulation file (ligthcurve fits: FITRES file)")
    parser.add_argument('--nameout', type=str, default='./SEARCHEFF_SPEC_DES_Moller_G10_v7.DAT',
                        help="Out name for spectroscopic selection function (useful for C10,G11 distinction)")
    parser.add_argument('--plots', action="store_true", default=False,
                        help="Data / Simulation plots")
    parser.add_argument(
        '--path_plots', default='./plots/', help='Path to save plots')
    parser.add_argument('--onlybias', action="store_true", default=False,
                        help="Computing bias, no selection function computation")
    parser.add_argument('--verbose',action='store_true', default=False)
    parser.add_argument('--validation',action='store_true', default=False,help='save fit parameters in text file')
    parser.add_argument('--nsn',default=-1,help='limit number of SNe used in fit')

    args = parser.parse_args()

    fdata = args.data
    fsim = args.sim
    nameout = args.nameout
    path_plots = args.path_plots
    nsn = args.nsn
    # no normalization
    # normalization is now taken into account in the sigmoid fit
    norm_bin = -1

    '''Read data/sim files
        if required copy,uncompress,read,clean
    '''
    print('___________________')
    print('   data: %s' % fdata)
    print('   sim: %s' % fsim)
    if '.gz' in fdata:
        if args.verbose:
            print('       copy & unzip %s' % (fdata))
        newfdata = copy_uncompress(fdata)
        fdata = newfdata
        data = load_fitres(fdata,nsn=nsn)
        os.remove(newfdata)
    else:
        data = load_fitres(fdata,nsn=nsn)

    if '.gz' in fsim:
        if args.verbose:
            print('       copy & unzip %s' % (fsim))
        newfsim = copy_uncompress(fsim)
        fsim = newfsim
        sim = load_fitres(fsim,validation=args.validation)
        os.remove(newfsim)
    else:
        sim = load_fitres(fsim,validation=args.validation)

    if not args.onlybias:
        '''Selection Function
        compute selection function by dividing data/sim by magnitude bin
        '''
        print('   Computing selection function')
        print('   Method: Data vs. Simulations (A. Moller)')
        # Init
        nbins = 20
        filt = 'i'
        min_mag = 20  # where are we complete? here you need a human choice, or do we?
        # Data vs Sim
        datsim = data_sim_division(
            filt, min_mag, norm_bin, nbins, args.plots, path_plots)
        # Emcee fit of dat/sim
        mc.emcee_fitting(datsim, args.plots, path_plots, nameout,args.verbose,args.validation)

    '''Plots
        (optional)
        variable distribution plots for data control
    '''
    if args.plots or args.onlybias:
        print('>> Plotting data, simulation distributions %s' % path_plots)
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)
        # c,x1,z distributions
        norm = mplot.distribution_plots(norm_bin, data, sim, path_plots)
        # c,x1 as a function of z
        mplot.plots_vs_z(data, sim, path_plots,args.onlybias)
