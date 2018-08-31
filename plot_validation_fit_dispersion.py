import matplotlib
matplotlib.use('Agg')
import pylab as plt
import pandas as pd
import numpy as np
import os
'''
Reading validation fits and plotting them
'''


def sigmoid_func(x, a, alph, bet):
    return a / (1 + np.exp((+alph * x) - bet))


def plot_speceff(df,input_sim_values,outname):
    # overplot all the speceff
    plt.clf()
    xx = np.linspace(20, 25, 200)
    for m in ['G10','C11']:
        model = df[df['fname'].str.contains(m)]
        model_color = colors[m]
        for i in range(len(model)):
            sel = model.iloc[i]
            for extra in ['_min','_max','']:
                yy = sigmoid_func(xx, sel['a' + extra] / sel['a' + extra],
                                  sel['alpha' + extra], sel['beta' + extra])
                plt.plot(xx,yy, color=model_color,alpha=.1)
    for m in ['G10','C11']:
        model = df[df['fname'].str.contains(m)]
        yy = sigmoid_func(xx, model['a'].mean() / model['a'].mean(),
                          model['alpha'].mean(), model['beta'].mean())
        plt.plot(xx,yy, color=colors_mean[m], label=m, lw=5)

    for m in ['G10','C11']:
        plt.scatter(input_sim_values[m]['i'].values, input_sim_values[m]['SPECEFF'].values,
                    color='black',marker=marker[m],label='input %s' % m)
    plt.legend()
    plt.ylim(-.1, 1.1)
    plt.xlim(20,24)
    plt.xlabel('mag')
    plt.ylabel('speceff')
    plt.savefig(outname)


def plot_rate_speceffs(df,input_sim_values,model,nameout):
    plt.clf()
    df = df[df['fname'].str.contains(m)]
    xx = input_sim_values[m]['i'].values
    stacked_yy = []
    for i in range(len(df)):
        sel = df.iloc[i]
        for extra in ['_min','_max','']:
            yy = np.divide(sigmoid_func(xx, sel['a' + extra] / sel['a' + extra],
                                        sel['alpha' + extra], sel['beta' + extra]), input_sim_values[model]['SPECEFF'].values)
            if extra == '':
                stacked_yy.append(yy)
                plt.plot(xx,yy, color=colors[model])
            else:
                plt.plot(xx,yy, color=colors[model],alpha=.1)
    plt.axhline(1,color='green',linewidth=2,label='input')
    # plot mean and contours
    ymeans = np.mean(stacked_yy, axis=0)
    ysigma = np.std(stacked_yy, axis=0)
    plt.plot(xx,ymeans,color='black',linewidth=3,label='mean')
    plt.plot(xx,ymeans - ysigma,color='cyan',linewidth=3,label='one sigma')
    plt.plot(xx,ymeans + ysigma,color='cyan',linewidth=3)
    if len(input_sim_values[m].keys()) > 3:
        plt.plot(xx, np.divide(input_sim_values[m]['SPECEFF_min'].values,input_sim_values[model]
                               ['SPECEFF'].values), color='grey',linewidth=3,label='sigma param')
        plt.plot(xx, np.divide(input_sim_values[m]['SPECEFF_max'].values,
                               input_sim_values[model]['SPECEFF'].values), color='grey',linewidth=3)
    plt.legend(loc='upper right')
    plt.xlabel('mag')
    plt.xlim(19,24)
    plt.ylim(0.4,1.6)
    plt.ylabel('FIT/INPUT SPECEFF')
    plt.savefig(nameout)

    return xx, ymeans,ysigma


def save_to_snana_format(df,study_type):

    for var in ['bias','onesigma']:
        for model in model_list:
            # to modify column names just
            # change "i" and "SPECEFF" here and in the rest of the function
            # to whatever you want
            df["i"] = df["x"].astype(float).round(2)
            df['SPECEFF'] = df["%s_%s" % (var, model)].astype(float).round(2)
            # column speceff
            new_i = df.fillna(method='bfill')
            charar = np.chararray(len(new_i), itemsize=8)
            charar[:] = 'SPECEFF:'
            df["VARNAMES:"] = charar
            df2 = pd.DataFrame(df, columns=['VARNAMES:', 'i', 'SPECEFF'])

            # save
            nameout = 'speceff_validation/SPECEFF_%s_%s_%s.DAT' % (study_type,var,model)
            fout = open(nameout, "w")
            fout.write("NVAR: 2 \n")
            df2.to_csv(fout, sep=" ", index=False)
            fout.close()


# init
model_list = ['G10','C11']
# plot settings
colors = {'G10':'orange','C11':'lightgreen'}
colors_mean = {'G10':'darkorange','C11':'green'}
linestyle = {'G10':'dashed','C11':'dashdot'}
marker = {'G10':'v','C11':'o'}
# Load
# - Speceff found during validation
# - input speceff for simulations
validation_values = pd.read_csv('validation_fit_values.txt',delimiter=' ',index_col=False)
DES_path = os.environ.get("DES_ROOT")
input_sim_values = {}
for m in model_list:
    input_sim_values[m] = pd.read_csv('%s/models/searcheff/SEARCHEFF_SPEC_DES_Moller_%s_v7.DAT' % (DES_path,m),
                                      skiprows=1,delimiter=' ')
    input_sim_values[m] = input_sim_values[m][input_sim_values[m]['i'] < 24]

good = validation_values[validation_values['a'] < 1.]
bad = validation_values[validation_values['a'] > 1.]
if len(bad) > 0:
    print('Fits that fail:',len(bad),'and succeded:',len(good))

# output for analysis
out_snana_df = pd.DataFrame()  # snana binning
out_smoothed_df = pd.DataFrame()  # smooth binning

# Plots
# Fitted specsel for validation
plot_speceff(good,input_sim_values,'speceff_validation/plots/dispersion_good_fits.png')
# ratio fitted specsel for validation / spec function provided to sim
xx_snana = {}
ymeans_snana = {}
ysigma_snana = {}
for m in model_list:
    xx_snana[m], ymeans_snana[m],ysigma_snana[m] = plot_rate_speceffs(
        good,input_sim_values,m,'speceff_validation/plots/rate_val_speceff_%s.png' % m)
    bias_corr = np.divide(input_sim_values[m]['SPECEFF'].values,ymeans_snana[m])
    onesigma_corr = np.multiply(input_sim_values[m]['SPECEFF'].values,ysigma_snana[m])
    out_snana_df['x'] = xx_snana[m]
    out_snana_df['bias_%s' % m] = bias_corr
    out_snana_df['onesigma_%s' % m] = onesigma_corr
    plt.clf()
    plt.plot(xx_snana[m],bias_corr)
    plt.plot(xx_snana[m],onesigma_corr,color='cyan')
    plt.xlabel('mag')
    plt.xlabel('speceff')
    plt.savefig('speceff_validation/plots/snana_%s.png' % m)

# saving snana binning
# out_snana_df.to_csv('speceff_validation/corrections_speceff_snana_binning.csv',index=False)
save_to_snana_format(out_snana_df,'snana')

# ratio fitted specsel for validation / sigmoid spec function provided to sim (direct parameters)
xx_smoothed = {}
ymeans_smoothed = {}
ysigma_smoothed = {}
for m in model_list:
    fname = 'SEARCHEFF_SPEC_DES_Moller_%s_v7_fitparams.txt' % m
    df = pd.read_csv(fname,delimiter=' ',index_col=False)
    xx = np.linspace(19, 25, 200)
    yy = sigmoid_func(xx, 1., df['alpha'].values[0], df['beta'].values[0])
    sigmoid_generator = {}
    sigmoid_generator[m] = pd.DataFrame()
    sigmoid_generator[m]['i'] = xx
    sigmoid_generator[m]['SPECEFF'] = yy
    sigmoid_generator[m]['SPECEFF_min'] = sigmoid_func(
        xx, 1., df['alpha_min'].values[0], df['beta_min'].values[0])
    sigmoid_generator[m]['SPECEFF_max'] = sigmoid_func(
        xx,1., df['alpha_max'].values[0], df['beta_max'].values[0])
    xx_smoothed[m],ymeans_smoothed[m],ysigma_smoothed[m] = plot_rate_speceffs(
        good,sigmoid_generator,m,'speceff_validation/plots/rate_val_speceff_sigmoid_directly_%s.png' % m)

    # creating outputs for analysis
    bias_corr = np.divide(yy,ymeans_smoothed[m])
    onesigma_corr = np.multiply(yy,ysigma_smoothed[m])
    out_smoothed_df['x'] = xx_smoothed[m]
    out_smoothed_df['bias_%s' % m] = bias_corr
    out_smoothed_df['onesigma_%s' % m] = onesigma_corr

    plt.clf()
    plt.plot(xx_smoothed[m],bias_corr)
    plt.plot(xx_smoothed[m],onesigma_corr,color='cyan')
    plt.xlabel('mag')
    plt.xlabel('speceff')
    plt.savefig('speceff_validation/plots/smoothed_%s.png' % m)

# out_smoothed_df.to_csv('speceff_validation/corrections_speceff_smoothed_binning.csv',index=False)
save_to_snana_format(out_smoothed_df,'smoothed')