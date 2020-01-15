import math
import numpy as np
import matplotlib.pyplot as plt

'''
Module for plotting data/sim distributions
2017/12/27 A. Moller

still a mess, to be improved!
'''

color_dic = {'data': 'red', 'sim': 'blue'}


def distribution_plots(norm_bin, data, sim, path_plots):
    '''
    some preliminary plots, c, x1, z distributions
    '''
    var_list = ['zHD', 'x1', 'c']
    for var in var_list:
        fig = plt.figure()
        n_dat, bins_dat, patches_dat = plt.hist(
            data[var], bins=15, histtype='step', color='red', label='data')
        index_of_bin_belonging_to_dat = np.digitize(data[var], bins_dat)
        n_sim, bins_sim, patches_sim = plt.hist(
            sim[var], bins=bins_dat, histtype='step', color='blue', label='sim', linestyle='--')
        index_of_bin_belonging_to_sim = np.digitize(sim[var], bins_sim)
        # error
        nbins = len(bins_dat)
        err_dat = []
        err_sim = []
        for ibin in range(nbins - 1):
            # data
            bin_elements_dat = np.take(data[var].values, np.where(
                index_of_bin_belonging_to_dat == ibin)[0])
            error_dat = np.sqrt(len(bin_elements_dat))
            err_dat.append(error_dat)
            # sim
            bin_elements_sim = np.take(sim[var].values, np.where(
                index_of_bin_belonging_to_sim == ibin)[0])
            error_sim = np.sqrt(len(bin_elements_sim))
            err_sim.append(error_sim)
            del bin_elements_sim, bin_elements_dat
        n_dat, bins_dat, patches_dat = plt.hist(
            data[var], bins=15, histtype='step', color='red', label='data')
        bin_centers = bins_dat[:-1] + (bins_dat[1] - bins_dat[0]) / 2.
        n_sim, bins_sim, patches_sim = plt.hist(
            sim[var], bins=bins_dat, histtype='step', label='sim', color='blue', linestyle='--')
        # sim normalization
        if norm_bin == -1:
            norm = 1  # normalization
        else:
            norm = n_dat[norm_bin] / n_sim[norm_bin]
        n_dat = n_dat
        n_sim = n_sim * norm
        # plot
        del fig
        fig = plt.figure()
        err_sim = np.array(err_sim) * norm  # is this true?
        plt.errorbar(bin_centers, n_dat, yerr=err_dat,
                     fmt='o', color='red', label='data')
        plt.errorbar(bin_centers, n_sim, yerr=err_sim,
                     fmt='o', color='blue', label='sim')
        plt.xlabel(var)
        plt.legend()
        plt.savefig('%s/hist_%s.png' % (path_plots, var))
        del fig
    return norm


def plot_2d(mean_dic, err_dic, var1, var2, zbin_dic, path_plots):
    fig = plt.figure()
    for db in ['data', 'sim']:
            fig = plt.errorbar(zbin_dic['z_bins_plot'],mean_dic[db][var1],
                               yerr=err_dic[db][var1],fmt='o',color=color_dic[db],label=db)
    plt.xlim(0, zbin_dic['max_z'] + zbin_dic['half_z_bin_step'])
    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.legend()
    plt.savefig('%s/evol_%s_%s.png' % (path_plots,var1,var2))
    del fig


def plots_vs_z(data, sim, path_plots):
    # Binning data by z, c and x1 distributions

    # zbin information
    zbin_dic = {}
    zbin_dic['step'] = 0.05
    zbin_dic['min_z'] = data['zHD'].min()
    zbin_dic['max_z'] = data['zHD'].max()
    zbin_dic['z_bins'] = np.arange(zbin_dic['min_z'], zbin_dic['max_z'], zbin_dic['step'])
    zbin_dic['half_z_bin_step'] = zbin_dic['step'] / 2.
    zbin_dic['z_bins_plot'] = np.arange(zbin_dic['min_z'] + zbin_dic['half_z_bin_step'],
                            zbin_dic['max_z'] - zbin_dic['half_z_bin_step'], zbin_dic['step'])

    # Bin data
    mean_dic = {}
    err_dic = {}
    for db in ['data', 'sim']:
        mean_dic[db] = {}
        err_dic[db] = {}
        for v in ['x1','c']:
            mean_dic[db][v] = []
            err_dic[db][v] = []

        for i, z_bin in enumerate(zbin_dic['z_bins'][:-1]):
            if db == 'sim':
                binned = sim[(sim['zHD'] >= z_bin) & (
                    sim['zHD'] < zbin_dic['z_bins'][i + 1])]
            if db == 'data':
                binned = data[(data['zHD'] >= z_bin) &
                              (data['zHD'] < zbin_dic['z_bins'][i + 1])]
            for v in ['x1','c']:
                mean_dic[db][v].append(np.mean(binned[v]))
                err_dic[db][v].append(np.std(binned[v]) / np.sqrt(len(binned)))

    # Plot
    for v in ['x1','c']:
        plot_2d(mean_dic, err_dic, v,'zHD', zbin_dic, path_plots)

