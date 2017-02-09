import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
from scipy.optimize import curve_fit
import os

#______SETTINGS
path_to_save = './plots_emcee/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')


def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(-20, 120, 10)
    yfit = alpha[:, None] + beta[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, trace)


def fit_MCMC(d_param):
    alpha_low,alpha_high = d_param['alpha']
    beta_low,beta_high = d_param['beta']
    A_low,A_high = d_param['A']

    def y_model(x,theta):
        A,alpha, beta = theta
        y_model = A / (1 + np.exp((alpha * x) + beta))
        return y_model

    # for debugging interactively
    # import ipdb
    # ipdb.set_trace()

    # Define our posterior using Python functions
    def log_prior(theta):  # flat prior for all parameters
        A,alpha, beta = theta
        if A_low < A < A_high and\
                alpha_low < alpha < alpha_high and \
                beta_low < beta < beta_high:
            return 0.0
        else:
            return -np.inf

    def log_likelihood(theta, x, y):
        A,alpha, beta = theta

        model = y_model(x,theta)

        log_L = -0.5 * np.sum(np.log(2 * np.pi * (err**2)) + (y - model)**2 / err**2)
        return log_L

    def log_posterior(theta, x, y):
        return log_prior(theta) + log_likelihood(theta, x, y)

    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps
    ndim = 3
    nwalkers = 12  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 30000  # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    np.random.seed(0)
    starting_guesses = np.random.random((nwalkers, ndim))

    # Here's the function call where all the work happens:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xdata, ydata])

    # Clear and run the production chain.
    pos_ini = [np.mean(d_param[key]) for key in list_key]
    pos = [pos_ini + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    for var in range(ndim):
        plt.clf()
        plt.plot(sampler.chain[:, :, var].T, color="k", alpha=0.4)
        plt.savefig(path_to_save+'/line-time' + str(var) + '_' + filt + '.png')

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    plot_MCMC_results(xdata, ydata, samples)

    import corner
    plt.clf()
    fig = corner.corner(samples, labels=list_key,quantiles=[0.5],bins=50)
    fig.savefig(path_to_save+'/triangle_' + filt + ".png")

    A_mcmc, alpha_mcmc, beta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                        zip(*np.percentile(samples, [16, 50, 84],
                                                           axis=0)))

    list_mcmc = [A_mcmc, alpha_mcmc, beta_mcmc]
    theta_mcmc = [p[0] for p in list_mcmc]
    list_mcmc = map(str,list_mcmc)
    print "value +/-"
    print "\n".join(list_mcmc)

    plt.clf()
    xx = np.linspace(lim_mag,26,200)
    plt.plot(xx,y_model(xx,theta_mcmc),color='orange',label='Emcee sigmoid fit')  # mcmc fit
    # plot data sel function Mat's & Chris's
    plt.scatter(MC_sel_function_data['mag'],
                MC_sel_function_data['SPECEFF'],color='green',label='v.1 Chris and Mat')
    plt.scatter(MC_sel_function_data_new['mag'],
                MC_sel_function_data_new['SPECEFF'],color='red',marker='*',s=50,label='v.2 Chris and Mat')
    plt.errorbar(xdata, ydata,yerr=err,fmt='o',color='blue', label="emcee selection function")
    plt.ylim(-.1,1)
    plt.xlabel('mag')
    plt.legend()
    plt.savefig(path_to_save+'/fitted_model_mcmc_' + filt + '.png')

    str_to_save=filt+' '+list_mcmc[0].split(',')[0].split('(')[1]+' '+list_mcmc[1].split(',')[0].split('(')[1]+' '+list_mcmc[2].split(',')[0].split('(')[1]

    return str_to_save

if __name__ == '__main__':

    # minimum magnitude threshold for the fit
    min_mag_dic={'i':20,'r':19}

    # initial guess
    def sigmoid_fit_func(x, a, alph, bet):
        return a / (1 + np.exp((+alph * x) + bet))
    low_bounds = [0.5,1.,-100]
    high_bounds = [2, 6, -20]

    filter_list=['r','i']
    str_to_save=[]
    for filt in filter_list:
        lim_mag = min_mag_dic[filt]
        # INPUTS
        input_name = 'division_m0obs_' + filt + '.csv'
        data = pd.read_csv(input_name)
        print 'reading',input_name
        MC_sel_function_data = pd.read_csv('../MATS_SPEC_EFF/' + filt + '_eff.csv', delimiter=' ')
        MC_sel_function_data_new = pd.read_csv('../2017_MAT/SEARCHEFF_SPEC_DES_%s.DAT'%filt, delimiter=' ')

        xdata_tmp = np.array(data[data['x'] > lim_mag]['x'])
        ydata_tmp = np.array(data[data['x'] > lim_mag]['div'])
        err_tmp = np.array(data[data['x'] > lim_mag]['err'])

        # filling max efficiency for lower magnitudes
        mag_arr = np.arange(18,lim_mag,0.2)
        eff_arr = np.ones(len(mag_arr))
        err_arr = np.divide(np.ones(len(mag_arr)),100)
        xdata = np.concatenate((xdata_tmp,mag_arr))
        ydata = np.concatenate((ydata_tmp,eff_arr))
        err = np.concatenate((err_tmp,err_arr))


        popt, pcov = curve_fit(sigmoid_fit_func,
                               xdata, ydata,bounds=(low_bounds,high_bounds))

        plt.figure()
        x = np.arange(min(xdata),26,0.2)
        plt.errorbar(xdata, ydata,yerr=err,fmt='o',color='blue', label="sigmoid fit")
        plt.plot(x, sigmoid_fit_func(x, *popt), 'r-', label="Fitted Sigmoid")
        # plot data sel function Mat's & Chris's
        plt.scatter(MC_sel_function_data['mag'],
                    MC_sel_function_data['SPECEFF'],color='green',label='v.1 Chris and Mat')
        plt.scatter(MC_sel_function_data_new['mag'],
                    MC_sel_function_data_new['SPECEFF'],color='red',marker='*',s=50,label='v.2 Chris and Mat')
        tuple = (round(popt[0],2),round(popt[1],2),round(popt[2],2))
        plt.title('%s / 1+ exp(%s*x+%s)' % tuple)
        plt.ylim(-.1,1)
        plt.legend()
        plt.xlabel('m0obs_' + filt)
        plt.savefig(path_to_save+'/fitted_sigmoid_' + filt + '.png')

        list_key = ['A','alpha','beta']
        d_param = {}
        d_param['A'] = (low_bounds[0],high_bounds[0])
        d_param['alpha'] = (low_bounds[1],high_bounds[1])
        d_param['beta'] = (low_bounds[2],high_bounds[2])
    
        str_to_save.append(fit_MCMC(d_param))

    #save into file
    text_file = open("emcee_function.txt", "w")
    text_file.write('filter A_mcmc alpha_mcmc beta_mcmc \n')
    text_file.write(str_to_save[0]+'\n')
    text_file.write(str_to_save[1])
    text_file.close()
