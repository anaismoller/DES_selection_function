import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import utils_emcee_gaussian as mc
import utils_emcee_poisson as mc
import utils_plots as mplot
import os


def copy_uncompress(fname):
    import gzip
    import shutil
    thisfname = fname.split('/')[-1].split('.gz')[0]
    with gzip.open(fname, 'rb') as f_in, open(thisfname, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return thisfname


def load_fitres(filepath):
    '''Load light curve fit file and apply cuts    
    Arguments:
        filepath (str) -- Lightcurve fit file with path   
    Returns:
        pandas dataframe 
    '''
    filetoload = pd.read_csv(filepath, index_col=False,
                             comment='#', delimiter=' ', skiprows=11, skipinitialspace=True)
    tmp = filetoload[(filetoload['c'] > -0.3) & (filetoload['c'] < 0.3) & (filetoload['x1'] > -3) & (filetoload['x1']
                                                                                                     < 3) & (filetoload['zHD'] > 0.05) & (filetoload['zHD'] < 0.9) & (filetoload['FITPROB'] > 1E-05)]
    df = tmp.copy()
    return df


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

    scratch_path = os.environ["SCRATCH_FITDIR"]

    '''Parse arguments
    '''
    parser = argparse.ArgumentParser(
        description='Selection function data vs simulations')
    parser.add_argument('--data', type=str,
                        default='%s/DES3YR_v1_freeze/DES3YR_DES_COMBINED_FITS/DES3YR_DES_COMBINED_FITS/FITOPT000.FITRES.gz' % (
                            scratch_path),
                        help="Data file (ligthcurve fits: FITRES file)")
    parser.add_argument('--sim', type=str,
                        default='%s/DES3YR_v5/DES3YR_DES_SPECEFF/DES3YR_DES_SPECEFF_AMG10/FITOPT000.FITRES.gz' % (
                            scratch_path),
                        help="Simulation file (ligthcurve fits: FITRES file)")
    parser.add_argument('--nameout', type=str, default='./SEARCHEFF_SPEC_DES_Moller_G10_v5.DAT',
                        help="Out name for spectroscopic selection function (useful for C10,G11 distinction)")
    parser.add_argument('--plots', action="store_true", default=False,
                        help="Data / Simulation plots")
    parser.add_argument(
        '--path_plots', default='./plots/', help='Path to save plots')
    parser.add_argument('--onlybias', action="store_true", default=False,
                        help="Computing bias, no selection function computation")
    parser.add_argument('--plateau', action="store_true", default=False,
                        help="Activating plateau for sigmoid function at mag <20.7")

    args = parser.parse_args()

    fdata = args.data
    fsim = args.sim
    nameout = args.nameout
    path_plots = args.path_plots
    plateau = args.plateau 
    norm_bin = 0  # this is searching sel function

    '''Read data/sim files
        if required copy,uncompress,read,clean
    '''
    print('>> Reading data/sim and copying/unzipping if needed')
    print('   data: %s' % fdata)
    if '.gz' in fdata:
        print('       copy & unzip %s' % (fdata))
        newfdata = copy_uncompress(fdata)
        fdata = newfdata
        data = load_fitres(fdata)
        os.remove(newfdata)
    else:
        data = load_fitres(fdata)

    print('   sim: %s' % fsim)
    if '.gz' in fsim:
        print('       copy & unzip %s' % (fsim))
        newfsim = copy_uncompress(fsim)
        fsim = newfsim
        sim = load_fitres(fsim)
        os.remove(newfsim)
    else:
        sim = load_fitres(fsim)

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

    if not args.onlybias:
        '''Selection Function
        compute selection function by dividing data/sim by magnitude bin
        '''
        print('>> Computing selection function')
        print('   Method: Data vs. Simulations (A. Moller)')
        # Init
        nbins = 20
        filt = 'i'
        min_mag = 20  # where are we complete? here you need a human choice, or do we?
        # Data vs Sim
        datsim = data_sim_division(
            filt, min_mag, norm_bin, nbins, args.plots, path_plots)
        # Emcee fit of dat/sim
        mc.emcee_fitting(datsim, args.plots, path_plots, nameout, plateau)
