import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import argparse
import sys
from scipy.optimize import curve_fit
import os
from mu_cosmo import dist_mu

#______SETTINGS
path_to_save = './preliminary_plots/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
debugging=False

#______LOAD DATA AND SIM, AND APPLY CUTS
data_ori = pd.read_csv('../data_and_sim/DESALL_fitted_myself/FITOPT000.FITRES',
                       index_col=False, comment='#',delimiter=' ')
tmp = data_ori[(data_ori['c'] > -0.3) & (data_ori['c'] < 0.3) & (data_ori['x1'] > -3) & (data_ori['x1']
                                                                                         < 3) & (data_ori['z'] > 0.05) & (data_ori['z'] < 0.9) & (data_ori['FITPROB'] > 1E-05)]
tmp2 = tmp[tmp.columns.values[:-1]]
#Selecting type Ias!
data=tmp2[tmp2['TYPE']==1]

sim = pd.read_csv('../data_and_sim/NEFF_YPIPELINE_SEO1_colour_changed/FITOPT000.FITRES',
                  index_col=False, comment='#', delimiter=' ')
tmp2 = sim[(sim['c'] > -0.3) & (sim['c'] < 0.3) & (sim['x1'] > -3) & (sim['x1'] < 3)
           & (sim['z'] > 0.05) & (sim['z'] < 0.9) & (sim['FITPROB'] > 1E-05)]
sim = tmp2

#______LOAD OLD Mat's & Chris's selection function from data (classified and non-class)
sel_function_data = {}
sel_function_data["i"] = pd.read_csv('../MATS_SPEC_EFF/i_eff.csv', delimiter=' ')
sel_function_data["r"] = pd.read_csv('../MATS_SPEC_EFF/r_eff.csv', delimiter=' ')


#______LOAD NEW Mat's & Chris's selection function from data (classified and non-class)
sel_function_data_new = {}
sel_function_data_new["i"] = pd.read_csv('../2017_MAT/SEARCHEFF_SPEC_DES_i.DAT', delimiter=' ')
sel_function_data_new["r"] = pd.read_csv('../2017_MAT/SEARCHEFF_SPEC_DES_r.DAT', delimiter=' ')

def initial_plots(norm_bin):
    '''
    some preliminary plots, c, x1, z distributions
    '''
    var_list = ['z','x1','c']
    for var in var_list:
        fig = plt.figure()
        n_dat, bins_dat, patches_dat = plt.hist(
            data[var],bins=15,histtype='step',color='red',label='data')
        index_of_bin_belonging_to_dat = np.digitize(data[var],bins_dat)
        n_sim, bins_sim, patches_sim = plt.hist(
            sim[var],bins=bins_dat,histtype='step',color='blue',label='sim',linestyle='--')
        index_of_bin_belonging_to_sim = np.digitize(sim[var],bins_sim)
        # error
        nbins = len(bins_dat)
        err_dat = []
        err_sim = []
        for ibin in range(nbins - 1):
            # data
            bin_elements_dat = np.take(data[var],np.where(
                index_of_bin_belonging_to_dat == ibin)[0])
            error_dat = np.sqrt(len(bin_elements_dat))
            err_dat.append(error_dat)
            # sim
            bin_elements_sim = np.take(sim[var],np.where(index_of_bin_belonging_to_sim == ibin)[0])
            error_sim = np.sqrt(len(bin_elements_sim))
            err_sim.append(error_sim)
            del bin_elements_sim, bin_elements_dat
        n_dat, bins_dat, patches_dat = plt.hist(
            data[var],bins=15,histtype='step',color='red',label='data')
        bin_centers = bins_dat[:-1] + (bins_dat[1] - bins_dat[0]) / 2.
        n_sim, bins_sim, patches_sim = plt.hist(
            sim[var],bins=bins_dat,histtype='step',label='sim',color='blue',linestyle='--')
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
        plt.errorbar(bin_centers,n_dat,yerr=err_dat,fmt='o',color='red',label='data')
        plt.errorbar(bin_centers,n_sim,yerr=err_sim,fmt='o',color='blue',label='sim')
        plt.xlabel(var)
        plt.legend()
        plt.savefig('%s/hist_%s.png' % (path_to_save,var))
        del fig
    return norm

def plots_vs_z():
    # now binned by z, c and x1 distributions
    z_bin_step = 0.05
    min_z = data['z'].min()
    max_z = data['z'].max()
    z_bins = np.arange(min_z,max_z,z_bin_step)
    mean_dic = {}
    err_dic = {}

    for db in ['data', 'sim']:
        mean_dic[db] = {}
        err_dic[db] = {}
        mean_dic[db]['x1'] = []
        mean_dic[db]['c'] = []
        mean_dic[db]['mB'] = []
        mean_dic[db]['distmu'] = []
        err_dic[db]['x1'] = []
        err_dic[db]['c'] = []
        err_dic[db]['mB'] = []
        if db=='sim':
            mean_dic[db]['delmu']=[]
            err_dic[db]['delmu']=[]

        for i, z_bin in enumerate(z_bins[:-1]):
            if db == 'sim':
                binned = sim[(sim['z'] >= z_bin) & (sim['z'] < z_bins[i + 1])]
            if db == 'data':
                binned = data[(data['z'] >= z_bin) & (data['z'] < z_bins[i + 1])]

            mean_x1 = np.mean(binned['x1'])
            mean_c = np.mean(binned['c'])
            mean_mb = np.mean(binned['mB'])
            # gaussian err=sigma/sqrt(n) : sigma=std
            err_x1 = np.std(binned['x1']) / np.sqrt(len(binned))
            err_c = np.std(binned['c']) / np.sqrt(len(binned))
            err_mb = np.std(binned['mBERR']) / np.sqrt(len(binned))

            if db=='sim':
                mean_delmu=np.mean(binned['delmu'])
                mean_dic[db]['delmu'].append(mean_delmu)
                err_delmu=np.std(binned['delmu'])/np.sqrt(len(binned))
                err_dic[db]['delmu'].append(err_delmu)

            mean_dic[db]['x1'].append(mean_x1)
            mean_dic[db]['c'].append(mean_c)
            mean_dic[db]['mB'].append(mean_mb)
            err_dic[db]['x1'].append(err_x1)
            err_dic[db]['c'].append(err_c)
            err_dic[db]['mB'].append(err_mb)
            av_z=z_bin+(z_bins[i+1]-z_bin)/2.
            mean_dic[db]['distmu'].append(dist_mu(av_z))

        #plots def
    half_z_bin_step=z_bin_step/2.
    z_bins_plot=np.arange(min_z+half_z_bin_step,max_z-half_z_bin_step,z_bin_step)
    color_dic={'data':'red','sim':'blue'}

    Mb_arr=np.ones(len(z_bins)-1)*(19.4)
    alpha=0.144 #from sim
    beta=3.1

    #Bias correction
    mean_mu_arr=[]
    mean_z_arr=[]
    err_mu_arr=[]
    sim['new_mu']=np.array(sim['mB'])+19.38+np.array(alpha*sim['x1'])-np.array(beta*sim['c'])-np.array(dist_mu(sim['z']))
    for i, z_bin in enumerate(z_bins[:-1]):
            binned_indices=sim[(sim['z'] >= z_bin) & (sim['z'] < z_bins[i + 1])].index.tolist()
            binned_mu=sim['new_mu'][binned_indices]
            mean_mu=np.mean(binned_mu)
            mean_z=z_bin+(z_bins[i+1]-z_bin)/2.
            err_mu_arr=np.sqrt(np.power(err_dic['sim']['mB'],2)+alpha*np.power(err_dic['sim']['x1'],2)+beta*np.power(err_dic['sim']['c'],2))
            mean_mu_arr.append(mean_mu)
            mean_z_arr.append(mean_z)
    fig=plt.figure()
    plt.errorbar(mean_z_arr,mean_mu_arr,yerr=np.array(err_mu_arr),fmt='o')
    plt.xlabel('z')
    plt.ylabel('bias correction')
    plt.title('mB+19.38+alpha*x1-beta*c-dist_mu(z)')
    plt.savefig('%s/bias.png'%path_to_save)
    del fig

    #alpha x1
    alpha_x1={}
    fig = plt.figure()
    alpha_x1['sim']= alpha * np.array(mean_dic['sim']['x1'])
    fig=plt.errorbar(z_bins_plot,alpha_x1['sim'],yerr=alpha*np.array(err_dic['sim']['x1']),fmt='o',color=color_dic['sim'],label='sim')
    alpha_x1['data']= alpha * np.array(mean_dic['data']['x1'])
    fig=plt.errorbar(z_bins_plot,alpha_x1['data'],yerr=alpha*np.array(err_dic['data']['x1']),fmt='o',color=color_dic['data'],label='data')
    chi= np.sum(np.divide(np.power(np.array(alpha_x1['sim'])-np.array(alpha_x1['data']),2),alpha_x1['data']))
    plt.title('chi square %f'%float(chi))
    plt.ylabel('%s x1'%alpha)
    plt.xlim(0,max_z+half_z_bin_step)
    plt.xlabel('z')
    plt.legend()
    plt.savefig('%s/evol_alpha_x1.png'%path_to_save)
    del fig

    #beta c
    beta_c={}
    fig = plt.figure()
    beta_c['data']= beta * np.array(mean_dic['data']['c'])
    fig=plt.errorbar(z_bins_plot,beta_c['data'],yerr=beta*np.array(err_dic['data']['c']),fmt='o',color=color_dic['data'],label='data')
    beta_c['sim']= beta * np.array(mean_dic['sim']['c'])
    fig=plt.errorbar(z_bins_plot,beta_c['sim'],yerr=beta*np.array(err_dic['sim']['c']),fmt='o',color=color_dic['sim'],label='sim')
    chi= np.sum(np.divide(np.power(np.array(beta_c['sim'])-np.array(beta_c['data']),2),beta_c['data']))
    plt.title('chi square %f'%float(chi))
    plt.ylabel(' %s c'%beta)
    plt.xlim(0,max_z+half_z_bin_step)
    plt.xlabel('z')  
    plt.legend()  
    plt.savefig('%s/evol_beta_c.png'%path_to_save)
    del fig

    #delta mu from fitres, hopefully similar to JLA
    fig = plt.figure()
    fig=plt.errorbar(z_bins_plot,np.array(mean_dic['sim']['delmu']),yerr=err_delmu,color='green',fmt='o')
    plt.ylabel('delta mu')
    plt.xlim(0,max_z+half_z_bin_step)
    plt.xlabel('z')
    plt.title('Delta Mu from SNANA (not accurate but good as ref)')
    plt.savefig('%s/SNANA_delta_mu_z.png'%path_to_save)
    del fig

    #x1 vs z
    fig = plt.figure()
    fig=plt.errorbar(z_bins_plot,mean_dic['data']['x1'],yerr=err_dic['data']['x1'],fmt='o',color='red',label='data')
    fig=plt.errorbar(z_bins_plot,mean_dic['sim']['x1'],yerr=err_dic['sim']['x1'],fmt='o',color='blue',label='sim')
    chi= np.sum(np.divide(np.power(np.array(mean_dic['sim']['x1'])-np.array(mean_dic['data']['x1']),2),mean_dic['data']['x1']))
    plt.title('chi square %f'%float(chi))
    plt.xlim(0,max_z+half_z_bin_step)
    plt.ylabel('x1')
    plt.xlabel('z')
    plt.legend()
    plt.savefig('%s/evol_x1_z.png'%path_to_save)
    del fig

    #c vs z
    fig = plt.figure()
    fig=plt.errorbar(z_bins_plot,mean_dic['data']['c'],yerr=err_dic['data']['c'],fmt='o',color='red',label='data')
    fig=plt.errorbar(z_bins_plot,mean_dic['sim']['c'],yerr=err_dic['sim']['c'],fmt='o',color='blue',label='sim')
    chi= np.sum(np.divide(np.power(np.array(mean_dic['sim']['c'])-np.array(mean_dic['data']['c']),2),mean_dic['data']['c']))
    plt.title('chi square %f'%float(chi))
    plt.xlim(0,max_z+half_z_bin_step)
    plt.ylabel('c')
    plt.xlabel('z')
    plt.legend()
    plt.savefig('%s/evol_c_z.png'%path_to_save)
    del fig


def mag_histos(filt,norm_bin,min_mag,nbins):
    '''
    Histograms of magnitudes for data and sim
    '''

    var = 'm0obs_' + filt
    # get sim and data histograms
    fig = plt.figure()
    n_data, bins_data, patches_data = plt.hist(data[data[var]>min_mag][var],nbins, fill=True,alpha=0.4)
    n_sim_norm, bins_sim, patches_sim = plt.hist(sim[sim[var]>min_mag][var],bins_data, fill=True,alpha=0.4)
    if norm_bin==-1:
        norm = 1 
    else:
        norm = n_data[norm_bin] / n_sim_norm[norm_bin]
    # normalize simulation
    n_sim = np.round(np.multiply(n_sim_norm,norm),decimals=2)

    del fig
    fig = plt.figure()
    bin_centers = bins_sim[:-1] + ((bins_sim[1] - bins_sim[0]) / 2.)
    plt.scatter(bin_centers,n_sim,color='green',label='sim')

    # Data
    # get bins
    n_data, bins_data, patches_data = plt.hist(
        data[var],bins_sim, fill=True,alpha=0.4,color='red',label='data')
    # plt.errorbar(bincenters, y_data, fmt='o',color='black', yerr=error_arr_data)
    plt.legend()
    plt.xlabel(var)
    plt.savefig(path_to_save + 'histo_' + var + '.png')
    # division
    del fig
    content_division = np.divide(n_data,n_sim,dtype=float)
    #for efficiencies bigger than 1
    content_division[content_division > 1] = 1
    # errors binomial with p=n/N and sigma^2 = Np(1-p)
    errors_division = []
    for i in range(len(n_data)):
        n = n_data[i]
        N = n_sim[i]
        if N > 0 and n <= N:
            error_bin = math.sqrt(n / N * (N - n)) / N
        else:
            #setting a minimal error (if we have 0 measurements this doesn't mean we have eff=0+-0)
            error_bin = 0.01
        if error_bin<0.01:
            error_bin = 0.01
        errors_division.append(error_bin)
    # plot division
    plt.clf()
    fig = plt.figure()
    plt.errorbar(bin_centers, content_division, yerr=errors_division,fmt='o')
    plt.xlabel(var)
    plt.ylim(-0.01,1.01)
    plt.title('division with binomial errors var2=Npq')
    plt.savefig(path_to_save + 'division_' + var + '.png')

    return bin_centers, content_division, errors_division

def exp_fit_func(x, a, b, c,d):
        return a * np.exp(-b * x + d) + c

def preliminary_selection_function(filt,min_mag,norm_bin,nbins):
    '''
    dividing data and sim to obtain the selection function
    '''
    var = 'm0obs_' + filt
    print 'processing',var
    bin_centers, content_division, errors_division = mag_histos(filt,norm_bin,min_mag,nbins)

    result = {}
    result['x'] = bin_centers
    result['div'] = content_division
    result['err'] = errors_division
    df = pd.DataFrame(result,columns=['x','div','err'])
    name = 'division_' + var + '.csv'
    df.to_csv(name,index=False)

    mag_delta=0.2
    x = np.arange(min_mag,25.9,mag_delta)
    y = np.array(content_division)
    popt, pcov = curve_fit(exp_fit_func, np.array(
        bin_centers), y,bounds=(0, [10., 2, 10,28]))

    plt.figure()
    plt.plot(np.array(bin_centers), y, 'ko', label="This analysis")
    tuple = (round(popt[0],2),round(popt[1],2),round(popt[2],2),round(popt[3],2))
    # plot data sel function Mat's & Chris's
    plt.scatter(sel_function_data[filt]['mag'],
                sel_function_data[filt]['SPECEFF'],color='green',label='v.1 Chris and Mat')
    plt.scatter(sel_function_data_new[filt]['mag'],
                sel_function_data_new[filt]['SPECEFF'],color='red',marker='*',s=50,label='v.2 Chris and Mat')
    plt.title('%s * exp(-%s*x+%s)+%s' % tuple)
    plt.legend()
    plt.xlabel(var)
    plt.savefig(path_to_save + 'selection_function_' + filt + '.png')
    
if __name__ == "__main__":

    #plot c,x1,z distributions
    norm_bin= 0 #this is searching sel function
    norm = initial_plots(norm_bin)
    #plot c,x1 as a function of z
    plots_vs_z()

    # Selection function
    # define how many bins
    nbins = 20
    #where are we complete? here you need a human choice, or do we?
    min_mag_dic={'i':20,'r':19}
    #list of peak magnitudes we may use un the selection function
    filter_list = ['r','i']   
    
    for filt in filter_list:
        min_mag=min_mag_dic[filt]
        preliminary_selection_function(filt,min_mag,norm_bin,nbins)
