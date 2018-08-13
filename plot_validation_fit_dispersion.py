import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
Reading validation fits and plotting them
'''


def sigmoid_func(x, a, alph, bet):
    return a / (1 + np.exp((+alph * x) - bet))


vals = pd.read_csv('validation_fit_values.txt',delimiter=' ',index_col=False)

good = vals[vals['a'] < 1.]
bad = vals[vals['a'] > 1.]
if len(bad)>0:
	print('Fits that fail:',len(bad))

plt.clf()
xx = np.linspace(20, 25, 200)
for i in range(len(good)):
    sel = good.iloc[i]
    plt.plot(xx, sigmoid_func(xx, 1.,sel['alpha_min'], sel['beta_min']), color='yellow')
    plt.plot(xx, sigmoid_func(xx, 1.,sel['alpha_max'], sel['beta_max']), color='yellow')
    plt.plot(xx, sigmoid_func(xx, 1.,sel['alpha'], sel['beta']), color='orange')
plt.ylim(-.1, 1.1)
plt.xlabel('mag')
plt.savefig('speceff_validation/plots/dispersion_good_fits.png')

plt.clf()
xx = np.linspace(20, 25, 200)
for i in range(len(bad)):
    sel = bad.iloc[i]
    plt.plot(xx, sigmoid_func(xx, 1.,sel['alpha_min'], sel['beta_min']), color='yellow')
    plt.plot(xx, sigmoid_func(xx, 1.,sel['alpha_max'], sel['beta_max']), color='yellow')
    plt.plot(xx, sigmoid_func(xx, 1.,sel['alpha'], sel['beta']), color='orange')
plt.xlabel('mag')
plt.savefig('speceff_validation/plots/dispersion_bad_fits.png')

a_std = good['a'].std()
alpha_std = good['alpha'].std()
beta_std = good['beta'].std()
a_mean = good['a'].mean()
alpha_mean = good['alpha'].mean()
beta_mean = good['beta'].mean()
print('mean and std')
print('a',a_mean, a_std)
print('alpha',alpha_mean, alpha_std)
print('beta',beta_mean, beta_std)