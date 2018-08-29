import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
from scipy.special import factorial
from chainconsumer import ChainConsumer
from scipy.optimize import curve_fit
import os

'''
2018/08/07 A. Moller

Module for emcee fitting of a sigmoid for efficiency
assumes:
    ndata distribution is poisson
    nsim is given (systematic evaluated using validation sims)
'''


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


def fit_MCMC(d_param, xdata, ydata, ndata,nsim, plots,path_plots, verbose,validation,nameout):
    alpha_low, alpha_high = d_param['alpha']
    beta_low, beta_high = d_param['beta']
    A_low, A_high = d_param['A']

    # Define our posterior using Python functions
    def log_prior(theta):  # flat prior for all parameters
        A, alpha, beta = theta
        if A_low < A < A_high and\
                alpha_low < alpha < alpha_high and \
                beta_low < beta < beta_high:
            return 0.0
        else:
            return -np.inf

    def log_likelihood(theta, x, y, ndata,nsim):
        A, alpha, beta = theta
        # sigmoid model for efficiency (y)
        model = sigmoid_func(x, A, alpha,beta) * nsim
        # Poisson likelihood of the probability of getting ndata with counts=ndata and rate=ideal number of events
        log_L = - np.sum((model) + np.log(factorial(ndata)) - (ndata * np.log(model)))
        return log_L

    def log_posterior(theta, x, y, ndata,nsim):
        return log_prior(theta) + log_likelihood(theta, x, y, ndata,nsim)

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
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=[xdata, ydata, ndata,nsim])

    # Clear and run the production chain.
    pos_ini = [np.mean(d_param[key]) for key in d_param.keys()]
    pos = [pos_ini + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    if plots:
        for var in range(ndim):
            # print('.  plotting line time')
            plt.clf()
            plt.plot(sampler.chain[:, :, var].T, color="k", alpha=0.4)
            plt.savefig('plots/line-time_' + str(var) + '.png')

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    plot_MCMC_results(xdata, ydata, samples)

    A_mcmc, alpha_mcmc, beta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                        zip(*np.percentile(samples, [16, 50, 84],
                                                           axis=0)))
    list_mcmc = [A_mcmc, alpha_mcmc, beta_mcmc]
    theta_mcmc = [p[0] for p in list_mcmc]
    min_theta_mcmc = [p[0] - p[1] for p in list_mcmc]
    max_theta_mcmc = [p[0] + p[2] for p in list_mcmc]
    list_mcmc = map(str, list_mcmc)
    if verbose:
        print(">> Emcee fitted value for efficiency sigmoid +/-")
        print("\n".join(list_mcmc))
    if validation:
        fname = "validation_fit_values.txt"
        if not os.path.exists(fname):
            outF = open(fname, "w")
            outF.write('fname a alpha beta a_min alpha_min beta_min a_max alpha_max beta_max \n')
        else:
            outF = open(fname, "a")
        output = nameout + ' ' + ' '.join(str(x) for x in theta_mcmc) + ' ' + ' '.join(str(x)
                                                                                       for x in min_theta_mcmc) + ' ' + ' '.join(str(x) for x in max_theta_mcmc) + '\n'
        outF.write(output)
        outF.close()
    else:
        # saving speceff params
        fname = "%s_fitparams.txt" % str(nameout.strip('.DAT').strip('/'))
        if not os.path.exists(fname):
            outF = open(fname, "w+")
            outF.write('fname a alpha beta a_min alpha_min beta_min a_max alpha_max beta_max \n')
        else:
            outF = open(fname, "a")
        output = nameout + ' ' + ' '.join(str(x) for x in theta_mcmc) + ' ' + ' '.join(str(x)
                                                                                       for x in min_theta_mcmc) + ' ' + ' '.join(str(x) for x in max_theta_mcmc) + '\n'
        outF.write(output)
        outF.close()

    if plots:
        # do corner plot
        plt.clf()
        c = ChainConsumer()
        c.add_chain(samples, parameters=['a','alpha','beta'])
        fig = c.plotter.plot(filename=path_plots + "/triangle.png", figsize="column")

        plt.clf()
        xx = np.linspace(xdata.min(), 25, 200)
        plt.plot(xx, sigmoid_func(xx, min_theta_mcmc[0],min_theta_mcmc[1],
                                  min_theta_mcmc[2]) / theta_mcmc[0], color='yellow',)
        plt.plot(xx, sigmoid_func(xx, max_theta_mcmc[0],max_theta_mcmc[1],max_theta_mcmc[2]) / theta_mcmc[0], color='yellow',
                 label='1 sigma')  # mcmc fit
        plt.plot(xx, sigmoid_func(xx, theta_mcmc[0],theta_mcmc[1],theta_mcmc[2]) / theta_mcmc[0],
                 color='orange', label='Emcee sigmoid fit')  # mcmc fit
        plt.scatter(xdata, ydata / theta_mcmc[0], color='blue',
                    label="emcee selection function")
        plt.ylim(-.1, 1.1)
        plt.xlabel('mag')
        plt.legend()
        plt.savefig(path_plots + '/fitted_model_mcmc_i.png')

    return theta_mcmc


def sigmoid_func(x, a, alph, bet):
    return a / (1 + np.exp((+alph * x) - bet))


def write_seleff(A_mcmc, alpha_mcmc, beta_mcmc, nameout):
    '''Writing selection function in SNANA friendly format

    SNANA doesn't like uneven binning, so we cheat
    I take emcee output for sigmoid fit
    create array with those values with uniform binning

    Arguments:
        emcee_fit A_mcmc, alpha_mcmc, beta_mcmc --  sigmoid fit parameters
        nameout -- output name of selection function
    '''
    # min magnitude of the seleff
    min_mag = 18
    mag = np.arange(min_mag, 25.9, 0.1)
    # create fit with emcee found parameters
    sigmoid_arr = np.around(sigmoid_func(
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


def emcee_fitting(datsim, plots, path_plots, nameout, verbose,validation):

    data = datsim

    lim_mag = 19.  # 20.7
    xdata = np.array(data[data['x'] > lim_mag]['x'])
    ydata = np.array(data[data['x'] > lim_mag]['div'])
    ndata = np.array(data[data['x'] > lim_mag]['n_data'])
    nsim = np.array(data[data['x'] > lim_mag]['n_sim'])

    # Initial params for fit
    low_bounds = [0.1, 1., 20]
    high_bounds = [4, 4, 70]
    # first guess
    popt, pcov = curve_fit(sigmoid_func,
                           xdata, ydata)
    if verbose:
        print('   Functional initial guess')
        print('   ',popt)
    low_bounds = [p - 3 * p / 10. for p in popt]
    high_bounds = [p + 3 * p / 10. for p in popt]
    if verbose:
        print('.  low/high now',low_bounds,high_bounds)
    if plots:
        plt.clf()
        xx = np.linspace(xdata.min(), 25, 200)
        plt.plot(xx, sigmoid_func(xx, popt[0], popt[1], popt[2]) / popt[0], color='orange',
                 label='Functional sigmoid fit')  # mcmc fit
        plt.scatter(xdata, ydata / popt[0], color='blue',
                    label="data rate")
        plt.ylim(-.1, 1.1)
        plt.xlabel('mag')
        plt.legend()
        plt.savefig(path_plots + '/fitted_model_func_i.png')

    d_param = {}
    d_param['A'] = (low_bounds[0], high_bounds[0])
    d_param['alpha'] = (low_bounds[1], high_bounds[1])
    d_param['beta'] = (low_bounds[2], high_bounds[2])

    if verbose:
        print('>> Emcee: fitting a sigmoid to data/simulation in mag i bins')
        print('          using Poisson distribution')
    A_mcmc, alpha_mcmc, beta_mcmc = fit_MCMC(
        d_param, xdata, ydata, ndata, nsim, plots, path_plots,verbose,validation,nameout)
    if verbose:
        print('   Finished emcee')
        # write the selection function
        print('>> Write selection function %s' % (nameout))
    write_seleff(A_mcmc, alpha_mcmc, beta_mcmc, nameout)
