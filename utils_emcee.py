import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import os

'''
Module for emcee fitting a sigmoid
2017/12/27 A. Moller
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


def fit_MCMC(d_param, xdata, ydata, err, plots,path_plots):
    alpha_low, alpha_high = d_param['alpha']
    beta_low, beta_high = d_param['beta']
    A_low, A_high = d_param['A']

    def y_model(x, theta):
        A, alpha, beta = theta
        y_model = A / (1 + np.exp((alpha * x) + beta))
        return y_model

    # Define our posterior using Python functions
    def log_prior(theta):  # flat prior for all parameters
        A, alpha, beta = theta
        if A_low < A < A_high and\
                alpha_low < alpha < alpha_high and \
                beta_low < beta < beta_high:
            return 0.0
        else:
            return -np.inf

    def log_likelihood(theta, x, y):
        A, alpha, beta = theta

        model = y_model(x, theta)

        log_L = -0.5 * np.sum(np.log(2 * np.pi * (err**2)
                                     ) + (y - model)**2 / err**2)
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
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=[xdata, ydata])

    # Clear and run the production chain.
    pos_ini = [np.mean(d_param[key]) for key in d_param.keys()]
    pos = [pos_ini + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    plot_MCMC_results(xdata, ydata, samples)

    A_mcmc, alpha_mcmc, beta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                        zip(*np.percentile(samples, [16, 50, 84],
                                                           axis=0)))

    list_mcmc = [A_mcmc, alpha_mcmc, beta_mcmc]
    theta_mcmc = [p[0] for p in list_mcmc]
    list_mcmc = map(str, list_mcmc)
    # print "value +/-"
    # print "\n".join(list_mcmc)

    if plots:
        plt.clf()
        xx = np.linspace(xdata.min(), 24, 200)
        plt.plot(xx, y_model(xx, theta_mcmc), color='orange',
                 label='Emcee sigmoid fit')  # mcmc fit
        plt.errorbar(xdata, ydata, yerr=err, fmt='o', color='blue',
                     label="emcee selection function")
        plt.ylim(-.1, 1)
        plt.xlabel('mag')
        plt.legend()
        plt.savefig(path_plots+'/fitted_model_mcmc_i.png')

    return theta_mcmc


def sigmoid_fit_func(x, a, alph, bet):
    return a / (1 + np.exp((+alph * x) + bet))


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
    sigmoid_arr = np.around(sigmoid_fit_func(
        mag, A_mcmc, alpha_mcmc, beta_mcmc), decimals=2)
    # in case my sigmoid goes outside bounds
    sigmoid_arr[sigmoid_arr > 1] = 1
    sigmoid_arr[sigmoid_arr < 0] = 0

    df = pd.DataFrame()
    df["eff"] = sigmoid_arr
    df["mag_i"] = mag
    df["i"]=df["mag_i"].astype(float).round(2)

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


def emcee_fitting(datsim, plots, path_plots, nameout):

    data = datsim

    # Initial params for fit
    low_bounds = [0.5, 2, -80]
    high_bounds = [2, 3.4, -43]
    lim_mag = 20.7

    xdata_tmp = np.array(data[data['x'] > lim_mag]['x'])
    ydata_tmp = np.array(data[data['x'] > lim_mag]['div'])
    err_tmp = np.array(data[data['x'] > lim_mag]['err'])

    # filling max efficiency for lower magnitudes
    mag_arr = np.arange(20, lim_mag, 0.2)
    eff_arr = np.ones(len(mag_arr))
    err_arr = np.divide(np.ones(len(mag_arr)), 100)
    xdata = np.concatenate((xdata_tmp, mag_arr))
    ydata = np.concatenate((ydata_tmp, eff_arr))
    err = np.concatenate((err_tmp, err_arr))

    d_param = {}
    d_param['A'] = (low_bounds[0], high_bounds[0])
    d_param['alpha'] = (low_bounds[1], high_bounds[1])
    d_param['beta'] = (low_bounds[2], high_bounds[2])

    print('>> Emcee: fitting a sigmoid to data/simulation in mag i bins')
    A_mcmc, alpha_mcmc, beta_mcmc = fit_MCMC(
        d_param, xdata, ydata, err, plots, path_plots)
    print('   Finished emcee')

    # write the selection function
    print('>> Write selection function %s'%(nameout))
    write_seleff(A_mcmc, alpha_mcmc, beta_mcmc, nameout)
    print('   Finished writting selection function %s'%(nameout))
