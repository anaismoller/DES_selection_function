import os
import argparse
import numpy as np
import pandas as pd
import utils_logging as lu
import utils_plots as mplot
import matplotlib.pyplot as plt
import utils_emcee_poisson as mc

'''
2020 A. Moller

Obtain sample selection efficiency
using ratio data/sim (selected SNe Ia/simulated Ias)

'''

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


def data_sim_ratio(data, sim, var='HOST_MAG_i', min_var=15, path_plots='./'):
    """
    Ratio between data and simulation in a given variable
    """
    # Init
    # TODO: no hardcut for lower limit
    data_var = data[data[var] > min_var][var]
    sim_var = sim[sim[var] > min_var][var]

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
    err_ratio = np.sqrt((err_data / hist_data) ** 2 +
                        (err_sim / hist_sim) ** 2) * ratio
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

    return df, minv, maxv


def write_seleff(A_mcmc, alpha_mcmc, beta_mcmc, nameout='SELEFF.DAT', min_var=15, max_var=26):
    '''Writing selection function in SNANA friendly format

    SNANA doesn't like uneven binning
    take emcee output for sigmoid fit
    create array with those values with uniform binning

    Arguments:
        emcee_fit A_mcmc, alpha_mcmc, beta_mcmc --  sigmoid fit parameters
        nameout -- output name of selection function
    '''
    # min magnitude of the seleff
    min_mag = min_var
    mag = np.arange(min_mag, max_var, 0.1)
    # create fit with emcee found parameters
    sigmoid_arr = np.around(mc.sigmoid_func(
        mag, 1., alpha_mcmc, beta_mcmc), decimals=2)

    # normalizing to 1
    sigmoid_arr = sigmoid_arr
    # in case my sigmoid goes outside bounds
    sigmoid_arr[sigmoid_arr > 1] = 1
    sigmoid_arr[sigmoid_arr < 0] = 0

    df = pd.DataFrame()
    df["eff"] = sigmoid_arr
    df["eff"] = df["eff"].astype(float).round(2)
    df["mag_i"] = mag
    df["i"] = df["mag_i"].astype(float).round(2)

    # filling from high eff to lower the magnitudes
    new_i = df.fillna(method='bfill')
    df['SPECEFF'] = df['eff']

    # column speceff
    charar = np.chararray(len(new_i), itemsize=8)
    charar[:] = 'SPECEFF:'
    df["VARNAMES:"] = charar
    df2 = pd.DataFrame(df, columns=['VARNAMES:', 'i', 'SPECEFF'])

    # save
    fout = open(nameout, "w")
    fout.write("NVAR: 2 \n")
    df2.to_csv(fout, sep=" ", index=False)
    fout.close()


if __name__ == "__main__":
    '''Parse arguments
    '''
    parser = argparse.ArgumentParser(
        description='Selection function data vs simulations')

    parser.add_argument('--data', type=str,
                        default='2_LCFIT/D_DATADES5YR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES',
                        help="Data file (ligthcurve fits: FITRES file)")

    parser.add_argument('--sim', type=str,
                        default='2_LCFIT/D_DESSIMCC/output/PIP_COMPUTE_SPEC_EFF_DESSIMCC/FITOPT000.FITRES',
                        help="Simulation file (ligthcurve fits: FITRES file)")

    parser.add_argument('--outpath', default='./dump/',
                        help='Path to save output')

    parser.add_argument('--extra_plots', action="store_true",
                        help='If c,x1 distributions are plotted')

    # Init
    args = parser.parse_args()
    fdata = args.data
    fsim = args.sim
    path_selection_function = args.outpath
    path_plots = f"{args.outpath}/plots"
    os.makedirs(path_plots, exist_ok=True)

    # Load fits files
    print(f"data: {fdata}")
    print(f"sim: {fsim}")
    data = load_fitres(fdata)
    sim = load_fitres(fsim)
    lu.print_green('Finished loading data and sim fits')

    # Selection config
    var = 'HOST_MAG_i'
    min_var = data[var].min()
    max_var = data[var].max()

    # Data vs Sim ratio
    df, minv, maxv = data_sim_ratio(data, sim, var=var,
                                    min_var=min_var, path_plots=path_plots)

    # Emcee fit of dat/sim
    A_mcmc, alpha_mcmc, beta_mcmc = mc.emcee_fitting(
        df, path_plots, min_var=min_var)

    # Write selection function in SNANA format
    write_seleff(A_mcmc, alpha_mcmc, beta_mcmc, nameout=f"{path_selection_function}/SELEFF.DAT", min_var=min_var, max_var=max_var)

    # Optional plots
    if args.extra_plots:
        # c,x1,z distributions
        norm = mplot.distribution_plots(0, data, sim, path_plots)
        # c,x1 as a function of z
        mplot.plots_vs_z(data, sim, path_plots)
        lu.print_blue('Finished extra plots')
