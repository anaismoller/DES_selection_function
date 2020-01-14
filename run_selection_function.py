import os
import argparse
import numpy as np
import pandas as pd
import utils_logging as lu
# import utils_plots as mplot
import matplotlib.pyplot as plt
import utils_emcee_poisson as mc

'''
2020 A. Moller

Obtain sample selection efficiency
using ratio data/sim (selected SNe Ia/simulated Ias)

'''


def calculate_sigmoid(row, min_theta_mcmc, theta_mcmc):
    return mc.sigmoid_func(row, min_theta_mcmc[0], min_theta_mcmc[1],
                           min_theta_mcmc[2]) / theta_mcmc[0]


def copy_uncompress(fname):
    import gzip
    import shutil
    thisfname = fname.split('/')[-1].split('.gz')[0]
    with gzip.open(fname, 'rb') as f_in, open(thisfname, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return thisfname


def load_fitres(fpath):
    '''Load light curve fit file and apply cuts    
    Arguments:
        fpath (str) -- Lightcurve fit file with path   
    Returns:
        pandas dataframe 
    '''
    if '.gz' in fpath:
        fpath = copy_uncompress(fpath)

    filetoload = pd.read_csv(fpath, index_col=False,
                             comment='#', delimiter=' ', skipinitialspace=True)
    # apply cosmology cuts
    tmp = filetoload[(filetoload['c'] > -0.3) & (filetoload['c'] < 0.3) & (filetoload['x1'] > -3) & (filetoload['x1']
                                                                                                     < 3) & (filetoload['zHD'] > 0.05) & (filetoload['zHD'] < 0.9) & (filetoload['FITPROB'] > 1E-05)]
    df = tmp.copy()
    dfout = df

    return dfout


def data_sim_ratio(data,sim,var='m0obs_i',path_plots='./'):
    """
    Ratio between data and simulation in a given variable
    """
    # Init
    # TODO: no hardcut for lower limit
    data_var = data[data[var]>20][var] 
    sim_var = sim[sim[var]>20][var] 

    minv = min([x.quantile(0.01) for x in [data_var, sim_var]])
    maxv = max([x.quantile(0.99) for x in [data_var, sim_var]])
    bins = np.linspace(minv, maxv, 15)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    hist_data, _ = np.histogram(data_var, bins=bins)
    hist_sim, _ = np.histogram(sim_var, bins=bins)
    err_data = np.sqrt(hist_data)
    err_sim = np.sqrt(hist_sim)

    ratio = hist_data / hist_sim
    ratio = ratio / ratio.max()
    err_ratio = np.sqrt((err_data / hist_data) ** 2 + (err_sim / hist_sim) ** 2) * ratio
    err_ratio = np.nan_to_num(err_ratio)

    # save as dataframe
    df = pd.DataFrame()
    df['x'] = bin_centers
    df['ratio'] = ratio
    df['err_ratio'] = err_ratio
    df['ndata'] = hist_data
    df['nsim'] = hist_sim
    df.meta = {}
    df.meta['ratio_variable'] = var

    return df


if __name__ == "__main__":

    # scratch_path = os.environ.get("SCRATCH_FITDIR")

    '''Parse arguments
    '''
    parser = argparse.ArgumentParser(
        description='Selection function data vs simulations')

    parser.add_argument('--data', type=str,
                        default='tests/data/FITOPT000.FITRES',
                        help="Data file (ligthcurve fits: FITRES file)")

    parser.add_argument('--sim', type=str,
                        default='tests/sim/FITOPT000.FITRES',
                        help="Simulation file (ligthcurve fits: FITRES file)")

    parser.add_argument('--outpath', default='./dump/', help='Path to save output')

    # Init
    args = parser.parse_args()
    fdata = args.data
    fsim = args.sim
    path_plots = f"{args.outpath}/plots"
    os.makedirs(path_plots, exist_ok=True)

    # Load fits files
    print(f"data: {fdata}")
    print(f"sim: {fsim}")
    data = load_fitres(fdata)
    sim = load_fitres(fsim)
    lu.print_green('Finished loading data and sim fits')

    # Data vs Sim ratio
    df = data_sim_ratio(data,sim, path_plots=path_plots)

    # # Emcee fit of dat/sim
    theta_mcmc, min_theta_mcmc, max_theta_mcmc = mc.emcee_fitting(
        df,path_plots)

    # # add model errors to csv
    # datsim_tosave['datasim_ratio_normalized'] = datsim_tosave['datasim_ratio'] / theta_mcmc[0]
    # datsim_tosave['model'] = datsim_tosave.apply(lambda datsim_tosave: calculate_sigmoid(
    #     datsim_tosave['i'], theta_mcmc, theta_mcmc), axis=1)
    # datsim_tosave['model_min'] = datsim_tosave.apply(lambda datsim_tosave: calculate_sigmoid(
    #     datsim_tosave['i'], min_theta_mcmc, theta_mcmc), axis=1)
    # datsim_tosave['model_max'] = datsim_tosave.apply(lambda datsim_tosave: calculate_sigmoid(
    #     datsim_tosave['i'], max_theta_mcmc, theta_mcmc), axis=1)
    # # datsim_tosave.round(2)
    # datsim_tosave.to_csv('datasim_ratio_%s_%s.csv' % (
    #     model, version), index=False, sep=' ', float_format='%.2f')

    # '''Plots
    #     (optional)
    #     variable distribution plots for data control
    # '''
    # print('>> Plotting data, simulation distributions %s' % path_plots)
    # # c,x1,z distributions
    # norm = mplot.distribution_plots(norm_bin, data, sim, path_plots)
    # # c,x1 as a function of z
    # mplot.plots_vs_z(data, sim, path_plots, args.onlybias)
