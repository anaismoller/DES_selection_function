import numpy as np
import matplotlib.pyplot as plt
import math

'''
Module for plotting data/sim distributions
2017/12/27 A. Moller

still a mess, can be improved!
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
    if var1 =='mu':
        fig = plt.errorbar(zbin_dic['z_bins_plot'],mean_dic['sim'][var1],
                           yerr=err_dic['sim'][var1],fmt='o',color=color_dic['sim'],label='sim')
    else:
        for db in ['data', 'sim']:
            fig = plt.errorbar(zbin_dic['z_bins_plot'],mean_dic[db][var1],
                               yerr=err_dic[db][var1],fmt='o',color=color_dic[db],label=db)
    plt.xlim(0, zbin_dic['max_z'] + zbin_dic['half_z_bin_step'])
    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.legend()
    plt.savefig('%s/evol_%s_%s.png' % (path_plots,var1,var2))
    del fig


def plots_vs_z(data, sim, path_plots,onlybias):
    # Binning data by z, c and x1 distributions

    # zbin information
    z_bin_step = 0.05
    min_z = data['zHD'].min()
    max_z = data['zHD'].max()
    z_bins = np.arange(min_z, max_z, z_bin_step)
    half_z_bin_step = z_bin_step / 2.
    z_bins_plot = np.arange(min_z + half_z_bin_step,
                            max_z - half_z_bin_step, z_bin_step)

    zbin_dic = {}
    zbin_dic['step'] = 0.05
    zbin_dic['min_z'] = data['zHD'].min()
    zbin_dic['max_z'] = data['zHD'].max()
    zbin_dic['z_bins'] = z_bins
    zbin_dic['z_bins_plot'] = z_bins_plot
    zbin_dic['half_z_bin_step'] = half_z_bin_step

    # Physical values to use
    Mb = 19.365
    alpha = 0.144  # from sim
    beta = 3.1

    # Need to define mu before binning
    sim['mu'] = np.array(sim['mB']) + Mb + np.array(alpha * sim['x1']) - \
        np.array(beta * sim['c']) - sim['SIM_DLMAG']
    data['mu'] = np.zeros(len(data['mB']))

    # Bin data
    mean_dic = {}
    err_dic = {}
    for db in ['data', 'sim']:
        mean_dic[db] = {}
        err_dic[db] = {}
        mean_dic[db]['x1'] = []
        mean_dic[db]['c'] = []
        mean_dic[db]['mB'] = []
        mean_dic[db]['alphax1'] = []
        mean_dic[db]['betac'] = []
        mean_dic[db]['mu'] = []
        err_dic[db]['x1'] = []
        err_dic[db]['c'] = []
        err_dic[db]['mB'] = []
        err_dic[db]['alphax1'] = []
        err_dic[db]['betac'] = []
        err_dic[db]['mu'] = []

        for i, z_bin in enumerate(z_bins[:-1]):
            if db == 'sim':
                binned = sim[(sim['zHD'] >= z_bin) & (
                    sim['zHD'] < z_bins[i + 1])]
            if db == 'data':
                binned = data[(data['zHD'] >= z_bin) &
                              (data['zHD'] < z_bins[i + 1])]

            mean_x1 = np.mean(binned['x1'])
            mean_c = np.mean(binned['c'])
            mean_mb = np.mean(binned['mB'])
            mean_mu = np.mean(binned['mu'])
            # gaussian err=sigma/sqrt(n) : sigma=std
            err_x1 = np.std(binned['x1']) / np.sqrt(len(binned))
            err_c = np.std(binned['c']) / np.sqrt(len(binned))
            err_mb = np.std(binned['mBERR']) / np.sqrt(len(binned))
            err_mu = np.sqrt(np.power(err_mb,2) + np.power(alpha,2) * np.power(
                err_x1,2) + np.power(beta,2) * np.power(err_c,2))

            mean_dic[db]['x1'].append(mean_x1)
            mean_dic[db]['c'].append(mean_c)
            mean_dic[db]['mB'].append(mean_mb)
            err_dic[db]['x1'].append(err_x1)
            err_dic[db]['c'].append(err_c)
            err_dic[db]['mB'].append(err_mb)
            mean_dic[db]['alphax1'].append(alpha * mean_x1)
            mean_dic[db]['betac'].append(beta * mean_c)
            err_dic[db]['alphax1'].append(alpha * err_x1)
            err_dic[db]['betac'].append(beta * err_c)
            mean_dic[db]['mu'].append(mean_mu)
            err_dic[db]['mu'].append(err_mu)

    # Plot
    plot_2d(mean_dic, err_dic, 'alphax1', 'zHD', zbin_dic, path_plots)
    plot_2d(mean_dic, err_dic, 'betac', 'zHD', zbin_dic, path_plots)
    plot_2d(mean_dic, err_dic, 'c','zHD', zbin_dic, path_plots)
    plot_2d(mean_dic, err_dic, 'x1','zHD', zbin_dic, path_plots)
    plot_2d(mean_dic, err_dic, 'mu','zHD', zbin_dic, path_plots)

    if onlybias:
        #save bias correction in txt file
        import pandas as pd
        bias_dic = {}
        bias_dic['z'] = zbin_dic['z_bins_plot']
        bias_dic['mu'] = mean_dic['sim']['mu']
        bias_dic['err'] = err_dic['sim']['mu']
        bias_df = pd.DataFrame(bias_dic,columns=['z','mu','err'])
        name = '%s/bias.csv'%path_plots
        bias_df.to_csv(name,index=False,float_format='%2.4f')



def mag_histos(filt, data,sim, norm_bin, min_mag, nbins,plots,path_plots):
    '''
    Histograms of magnitudes for data and sim
    '''

    var = 'm0obs_' + filt
    # get sim and data histograms
    fig = plt.figure()
    n_data, bins_data, patches_data = plt.hist(
        data[data[var] > min_mag][var], nbins, fill=True, alpha=0.4)
    n_sim_norm, bins_sim, patches_sim = plt.hist(
        sim[sim[var] > min_mag][var], bins_data, fill=True, alpha=0.4)
    if norm_bin == -1:
        norm = 1
    else:
        norm = n_data[norm_bin] / n_sim_norm[norm_bin]
    # normalize simulation
    n_sim = np.round(np.multiply(n_sim_norm, norm), decimals=2)

    del fig
    fig = plt.figure()
    bin_centers = bins_sim[:-1] + ((bins_sim[1] - bins_sim[0]) / 2.)
    plt.scatter(bin_centers, n_sim, color='green', label='sim')

    # Data
    # get bins
    n_data, bins_data, patches_data = plt.hist(
        data[var], bins_sim, fill=True, alpha=0.4, color='red', label='data')
    plt.legend()
    plt.xlabel(var)
    if plots:
        plt.savefig(path_plots + 'histo_' + var + '.png')
    # division
    del fig
    content_division = np.divide(n_data, n_sim, dtype=float)
    # for efficiencies bigger than 1, this is due to our normalisation
    content_division[content_division > 1] = 1
    # errors binomial with p=n/N and sigma^2 = Np(1-p)
    errors_division = []
    for i in range(len(n_data)):
        n = n_data[i]
        N = n_sim[i]
        if N > 0 and n <= N:
            error_bin = math.sqrt(n / N * (N - n)) / N
        else:
            # setting a minimal error (if we have 0 measurements this doesn't
            # mean we have eff=0+-0)
            error_bin = 0.05
        if error_bin < 0.05:
            error_bin = 0.05
        errors_division.append(error_bin)
    # plot division
    plt.clf()
    fig = plt.figure()
    plt.errorbar(bin_centers, content_division, yerr=errors_division, fmt='o')
    plt.xlabel(var)
    plt.ylim(-0.01, 1.01)
    plt.title('division with binomial errors var2=Npq')
    if plots:
        plt.savefig(path_plots + 'division_' + var + '.png')

    return bin_centers, content_division, errors_division
