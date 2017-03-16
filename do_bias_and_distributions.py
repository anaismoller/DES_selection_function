import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
from scipy.optimize import curve_fit
import os
from mu_cosmo import dist_mu
from scipy import stats  
#______SETTINGS
path_to_save = './plots/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
debugging=True

#______LOAD DATA AND SIM, APPLY CUTS
data_ori = pd.read_csv('../data_and_sim/DESALL_fitted_myself_withcuts/FITOPT000.FITRES',
                       index_col=False, comment='#',delimiter=' ')
tmp = data_ori[(data_ori['c'] > -0.3) & (data_ori['c'] < 0.3) & (data_ori['x1'] > -3) & (data_ori['x1']
                                                                                         < 3) & (data_ori['z'] > 0.05) & (data_ori['z'] < 0.9) & (data_ori['FITPROB'] > 1E-05)]
tmp2 = tmp[tmp.columns.values[:-1]]
#only Ias!!!!
data=tmp2[tmp2['TYPE']==1]

print 'SNe in the sample', len(data)

sim = pd.read_csv('../data_and_sim/20170309_YEFF_SE03/FITOPT000.FITRES',
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

#______LOAD NEW Mat's & Chris's c, x1 vs z distributions
MC_vs_z={}
MC_vs_z['c']=pd.read_csv('../2017_MATS_cx1_z/sim_c_v_z_28022017.txt', delimiter=' ')
MC_vs_z['x1']=pd.read_csv('../2017_MATS_cx1_z/sim_x1_v_z_28022017.txt', delimiter=' ')


def finding_norm_bin_histos(var):
    '''
    Using chi square to determine which norm bin to use
    '''
    min_arg={}

    n_dat, bins_dat, patches_dat = plt.hist(
        data[var],bins=15,histtype='step',color='red',label='data')
    n_sim, bins_sim, patches_sim = plt.hist(
        sim[var],bins=bins_dat,histtype='step',label='sim',color='blue',linestyle='--')
    not_to_use=np.where(n_dat == 0)[0]
    index_arr =[ i for i in list(range(0,len(n_dat))) if i not in not_to_use]

    if var=='c':#since these values are very close to zero and will explote the chi
        index_arr=index_arr[3:-3]
    chi_square=[]
    for norm_bin in index_arr:
        # sim normalization
        norm = n_dat[norm_bin] / n_sim[norm_bin]
        n_dat = n_dat
        n_sim = n_sim * norm
        #now back to the array but need to eliminate 0 bins again
        tmp= np.divide(np.power(n_sim[[index_arr]]-n_dat[[index_arr]],2),n_dat[[index_arr]])
        chi_square.append(np.sum(tmp))
    min_arg[var]=chi_square
    return np.argmin(min_arg[var]),np.min(min_arg[var])

def histograms(var,norm_bin,chi):
    '''
    some preliminary plots, c, x1, z distributions
    '''
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
    plt.title('chi square=%f'%float(chi))
    if var=='c':
        plt.title('chi square=%f, not used the close to zero bins'%float(chi))
    plt.legend()
    plt.savefig('%s/hist_%s.png' % (path_to_save,var))
    del fig

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
        if db=='data' and debugging==True:
            mean_dic['data_deep'] = {}
            mean_dic['data_shallow'] = {}
            err_dic['data_deep'] = {}
            err_dic['data_shallow'] = {}
            mean_dic['data_deep']['c'] = []
            mean_dic['data_shallow']['c'] = []
            err_dic['data_deep']['c'] = []
            err_dic['data_shallow']['c'] = []
            mean_dic['data_deep']['x1'] = []
            mean_dic['data_shallow']['x1'] = []
            err_dic['data_deep']['x1'] = []
            err_dic['data_shallow']['x1'] = []

        for i, z_bin in enumerate(z_bins[:-1]):
            if db == 'sim':
                binned = sim[(sim['z'] >= z_bin) & (sim['z'] < z_bins[i + 1])]
            if db == 'data':
                binned = data[(data['z'] >= z_bin) & (data['z'] < z_bins[i + 1])]

            mean_x1 = np.mean(binned['x1'])
            mean_c = np.mean(binned['c'])
            mean_mb = np.mean(binned['mB'])
            if db=='data' and debugging==True:
                tmp_deep = binned.loc[binned['FIELD'].isin(['X3','C3'])]
                tmp_shallow = binned.loc[binned['FIELD'].isin(['E1','E2','S1','S2','C1','C2','X1','X2'])]
                #ANOVA test
                print '>>zbin',i
                f_val, p_val = stats.f_oneway(tmp_deep['c'], tmp_shallow['c'])  
                print "c One-way ANOVA P =", p_val 
                f_val, p_val = stats.f_oneway(tmp_deep['x1'], tmp_shallow['x1'])  
                print "x1 One-way ANOVA P =", p_val 
                if len(tmp_deep)>0:
                    mean_c_deep=np.mean(tmp_deep['c'])
                    err_c_deep = np.std(tmp_deep['c']) / np.sqrt(len(tmp_deep))
                    mean_x1_deep=np.mean(tmp_deep['x1'])
                    err_x1_deep = np.std(tmp_deep['x1']) / np.sqrt(len(tmp_deep))
                else:
                    mean_c_deep=-10
                    err_c_deep=0
                    mean_x1_deep=-10
                    err_x1_deep=0
                if len(tmp_shallow)>0:
                    mean_c_shallow=np.mean(tmp_shallow['c'])
                    err_c_shallow = np.std(tmp_shallow['c']) / np.sqrt(len(tmp_shallow))
                    mean_x1_shallow=np.mean(tmp_shallow['x1'])
                    err_x1_shallow = np.std(tmp_shallow['x1']) / np.sqrt(len(tmp_shallow))
                else:
                    mean_c_shallow=-10
                    err_c_shallow=0
                    mean_x1_shallow=-10
                    err_x1_shallow=0

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
            if db=='data' and debugging==True:
                mean_dic['data_deep']['c'].append(mean_c_deep)
                mean_dic['data_shallow']['c'].append(mean_c_shallow)
                err_dic['data_deep']['c'].append(err_c_deep)
                err_dic['data_shallow']['c'].append(err_c_shallow)
                mean_dic['data_deep']['x1'].append(mean_x1_deep)
                mean_dic['data_shallow']['x1'].append(mean_x1_shallow)
                err_dic['data_deep']['x1'].append(err_x1_deep)
                err_dic['data_shallow']['x1'].append(err_x1_shallow)

    #plots def
    half_z_bin_step=z_bin_step/2.
    z_bins_plot=np.arange(min_z+half_z_bin_step,max_z-half_z_bin_step,z_bin_step)
    color_dic={'data':'red','sim':'blue'}

    #constants
    Mb_arr=np.ones(len(z_bins)-1)*(19.38)
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
    plt.savefig('./plots/bias.png')
    del fig

    #alpha x1
    alpha_x1={}
    fig = plt.figure()
    alpha_x1['sim']= alpha * np.array(mean_dic['sim']['x1'])
    fig=plt.errorbar(z_bins_plot,alpha_x1['sim'],yerr=alpha*np.array(err_dic['sim']['x1']),fmt='o',color=color_dic['sim'],label='sim')
    alpha_x1['data']= alpha * np.array(mean_dic['data']['x1'])
    fig=plt.errorbar(z_bins_plot,alpha_x1['data'],yerr=alpha*np.array(err_dic['data']['x1']),fmt='o',color=color_dic['data'],label='data')
    chi= np.abs(np.sum(np.divide(np.power(np.array(alpha_x1['sim'])-np.array(alpha_x1['data']),2),alpha_x1['data'])))
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
    chi= np.abs(np.sum(np.divide(np.power(np.array(beta_c['sim'])-np.array(beta_c['data']),2),beta_c['data'])))
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
    plt.savefig('plots/SNANA_delta_mu_z.png')
    del fig

    #x1 vs z
    fig = plt.figure()
    fig=plt.errorbar(z_bins_plot,mean_dic['data']['x1'],yerr=err_dic['data']['x1'],fmt='o',color='red',label='data')
    fig=plt.errorbar(z_bins_plot,mean_dic['sim']['x1'],yerr=err_dic['sim']['x1'],fmt='o',color='blue',label='sim')
    chi= np.abs(np.sum(np.divide(np.power(np.array(mean_dic['sim']['x1'])-np.array(mean_dic['data']['x1']),2),mean_dic['data']['x1'])))
    fig=plt.scatter(MC_vs_z['x1']['z'],MC_vs_z['x1']['x1'],label='MC sim',color='grey')
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
    chi= np.abs(np.sum(np.divide(np.power(np.array(mean_dic['sim']['c'])-np.array(mean_dic['data']['c']),2),mean_dic['data']['c'])))
    fig=plt.scatter(MC_vs_z['c']['z'],MC_vs_z['c']['c'],label='MC sim',color='grey')
    plt.title('chi square %f'%float(chi))
    plt.xlim(0,max_z+half_z_bin_step)
    plt.ylabel('c')
    plt.xlabel('z')
    plt.legend()
    plt.savefig('%s/evol_c_z.png'%path_to_save)
    del fig

    if debugging==True:
        #splitting data in shallow and deep
        fig = plt.figure()
        fig=plt.errorbar(z_bins_plot,mean_dic['data_shallow']['c'],yerr=err_dic['data_shallow']['c'],fmt='o',color='orange',label='data shallow')
        fig=plt.errorbar(z_bins_plot,mean_dic['data_deep']['c'],yerr=err_dic['data_deep']['c'],fmt='o',color='magenta',label='data deep')
        fig=plt.errorbar(z_bins_plot,mean_dic['sim']['c'],yerr=err_dic['sim']['c'],fmt='o',color='blue',label='sim')
        fig=plt.scatter(MC_vs_z['c']['z'],MC_vs_z['c']['c'],label='MC sim',color='grey')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('c')
        plt.ylim(-0.15,.15)
        plt.savefig('%s/evol_c_z_data_splitdeepshallow.png'%path_to_save)
        del fig

        fig = plt.figure()
        fig=plt.errorbar(z_bins_plot,mean_dic['data_shallow']['x1'],yerr=err_dic['data_shallow']['x1'],fmt='o',color='orange',label='data shallow')
        fig=plt.errorbar(z_bins_plot,mean_dic['data_deep']['x1'],yerr=err_dic['data_deep']['x1'],fmt='o',color='magenta',label='data deep')
        fig=plt.errorbar(z_bins_plot,mean_dic['sim']['x1'],yerr=err_dic['sim']['x1'],fmt='o',color='blue',label='sim')
        fig=plt.scatter(MC_vs_z['x1']['z'],MC_vs_z['x1']['x1'],label='MC sim',color='grey')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('x1')
        plt.ylim(-1.5,2)
        plt.savefig('%s/evol_x1_z_data_splitdeepshallow.png'%path_to_save)
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

if __name__ == "__main__":

	#plot c,x1,z distributions and c,x1 as a function of z
    var_list=['z','c','x1']
    for var in var_list:
        norm_bin,chi= finding_norm_bin_histos(var)
        if var=='c':
            norm_bin=3
        norm = histograms(var,norm_bin,chi)
    plots_vs_z()

