import os
import emcee
import numpy as np
import pandas as pd
import utils_logging as lu
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import curve_fit
from chainconsumer import ChainConsumer
'''
2020 A. Moller

Module for emcee fitting of a sigmoid for efficiency
assumes that data distribution is poisson
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


def fit_MCMC(df, fit_param, path_plots):
    alpha_low, alpha_high = fit_param['alpha']
    beta_low, beta_high = fit_param['beta']
    A_low, A_high = fit_param['A']

    # Define our posterior using Python functions
    def log_prior(theta):
        # flat prior for all parameters
        A, alpha, beta = theta
        if A_low < A < A_high and\
                alpha_low < alpha < alpha_high and \
                beta_low < beta < beta_high:
            return 0.0
        else:
            return -np.inf

    def log_likelihood(theta, x, y, ndata, nsim):
        A, alpha, beta = theta
        # sigmoid model for efficiency (y)
        model = sigmoid_func(x, A, alpha, beta) * nsim
        # Poisson likelihood of the probability of getting ndata with counts=ndata and rate=ideal number of events
        log_L = - np.sum((model) + np.log(factorial(ndata)) -
                         (ndata * np.log(model)))
        return log_L

    def log_posterior(theta, x, y, ndata, nsim):
        return log_prior(theta) + log_likelihood(theta, x, y, ndata, nsim)

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
        nwalkers, ndim, log_posterior, args=[df['x'].values, df['ratio'].values, df['ndata'].values, df['nsim'].values])

    # Clear and run the production chain.
    pos_ini = [np.mean(fit_param[key]) for key in fit_param.keys()]
    pos = [pos_ini + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    for var in range(ndim):
        plt.clf()
        plt.plot(sampler.chain[:, :, var].T, color="k", alpha=0.4)
        plt.savefig(path_plots + '/line-time_' + str(var) + '.png')

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

    A_mcmc, alpha_mcmc, beta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                        zip(*np.percentile(samples, [16, 50, 84],
                                                           axis=0)))
    list_mcmc = [A_mcmc, alpha_mcmc, beta_mcmc]
    theta_mcmc = [p[0] for p in list_mcmc]
    min_theta_mcmc = [p[0] - p[1] for p in list_mcmc]
    max_theta_mcmc = [p[0] + p[2] for p in list_mcmc]
    list_mcmc = map(str, list_mcmc)

    # do corner plot
    # plt.clf()
    # cha = ChainConsumer()
    # cha.add_chain(samples, parameters=['a', 'alpha', 'beta'])
    # fig = cha.plotter.plot(filename=f"{path_plots}/triangle.png")

    plt.clf()
    variable = df.ratio_variable
    xx = np.linspace(df['x'].min(), df['x'].max(), 200)
    plt.plot(xx, sigmoid_func(xx, min_theta_mcmc[0], min_theta_mcmc[1],
                              min_theta_mcmc[2]) / theta_mcmc[0], color='yellow',)
    plt.plot(xx, sigmoid_func(xx, max_theta_mcmc[0], max_theta_mcmc[1], max_theta_mcmc[2]) / theta_mcmc[0], color='yellow',
             label='1 sigma')  # mcmc fit
    plt.plot(xx, sigmoid_func(xx, theta_mcmc[0], theta_mcmc[1], theta_mcmc[2]) / theta_mcmc[0],
             color='orange', label='Emcee sigmoid fit')  # mcmc fit
    plt.errorbar(df['x'], df['ratio'] / theta_mcmc[0], yerr=df['err_ratio'], fmt='o',
                label="emcee selection function")
    plt.ylim(-.1, 1.1)
    plt.xlabel(variable)
    plt.legend()
    plt.savefig(f"{path_plots}/fitted_model_mcmc_{variable}.png")

    return theta_mcmc, min_theta_mcmc, max_theta_mcmc


def sigmoid_func(x, a, alph, bet):
    return a / (1 + np.exp((+alph * x) - bet))


def emcee_fitting(df, path_plots,min_var=15):

    # Functional fit for initial guess
    # Initial params
    low_bounds = [0.1, 1., 20]
    high_bounds = [4, 4, 70]
    popt, pcov = curve_fit(sigmoid_func,
                           df['x'].values, df['ratio'].values)
    lu.print_green('Functional initial guess', popt)
    low_bounds = [p - 3 * p / 10. for p in popt]
    high_bounds = [p + 3 * p / 10. for p in popt]

    # Plot initial guess
    variable = df.ratio_variable
    plt.clf()
    fig = plt.figure()
    xx = np.linspace(df['x'].min(), df['x'].max(), 200)
    plt.errorbar(df['x'], df['ratio'], yerr=df['err_ratio'], fmt='o')
    plt.plot(xx, sigmoid_func(xx, popt[0], popt[1], popt[2]) /
             popt[0], color='orange', label='Functional sigmoid fit')
    plt.xlabel(variable)
    plt.ylabel('ratio')
    plt.ylim(-.1, 1.1)
    plt.savefig(f"{path_plots}/fitted_model_func_{variable}.png")
    plt.clf()
    # save initial fit params
    fit_param = {}
    fit_param['A'] = (low_bounds[0], high_bounds[0])
    fit_param['alpha'] = (low_bounds[1], high_bounds[1])
    fit_param['beta'] = (low_bounds[2], high_bounds[2])

    lu.print_blue(
        'Emcee fitting sigmoid to data/simulation ratio (Poisson errors)')
    theta_mcmc, min_theta_mcmc, max_theta_mcmc = fit_MCMC(
        df, fit_param, path_plots)
    lu.print_green('emcee:',theta_mcmc)
    lu.print_blue('Finished emcee')

    return theta_mcmc, min_theta_mcmc, max_theta_mcmc 
