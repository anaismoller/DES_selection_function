import pandas as pd
import numpy as np
'''
sel eff with uniform magnitude binning
not the most beautiful code but it works
'''
#read latest emcee fit
emcee=pd.read_csv("emcee_function.txt",delimiter=" ")

#define sigmoid function
def sigmoid_fit_func(x, a, alph, bet):
    return a / (1 + np.exp((+alph * x) + bet))

#min magnitude of the seleff
min_mag=18

eff_r = {}
eff_i = {}
mag = np.arange(min_mag,25.9,0.1)

r_emcee=emcee[emcee['filter']=='r']
r_arr = np.around(sigmoid_fit_func(mag,r_emcee['A_mcmc'].values,r_emcee['alpha_mcmc'].values,r_emcee['beta_mcmc'].values),decimals=2)
r_arr[r_arr > 1] = 1
r_arr[r_arr < 0] = 0
eff_r["eff"] = r_arr
eff_r["mag_r"] = mag

i_emcee=emcee[emcee['filter']=='i']
i_arr = np.around(sigmoid_fit_func(mag,i_emcee['A_mcmc'].values,i_emcee['alpha_mcmc'].values,i_emcee['beta_mcmc'].values),decimals=2)
i_arr[i_arr > 1] = 1
i_arr[i_arr < 0] = 0
eff_i["eff"] = i_arr
eff_i["mag_i"] = mag

df_r = pd.DataFrame(eff_r)
df_i = pd.DataFrame(eff_i)

# filling from high eff to lower the magnitudes
new_r = df_r.fillna(method='bfill')
new_i = df_i.fillna(method='bfill')

bla_r = {}
bla_r['SPECEFF'] = new_r['eff']
bla_r["r"] = new_r['mag_r']
charar = np.chararray(len(new_r),itemsize=8)
charar[:] = 'SPECEFF:'
bla_r["VARNAMES:"] = charar
df_r = pd.DataFrame(bla_r,columns=['VARNAMES:','r','SPECEFF'])
df_r.to_csv("SPEC_EFF_NEFF_SEO1_r.DAT",sep=" ",index=False)

bla_i = {}
bla_i['SPECEFF'] = new_i['eff']
bla_i["i"] = new_i['mag_i']
charar = np.chararray(len(new_i),itemsize=8)
charar[:] = 'SPECEFF:'
bla_i["VARNAMES:"] = charar
df_i = pd.DataFrame(bla_i,columns=['VARNAMES:','i','SPECEFF'])
print bla_i
df_i.to_csv("SPEC_EFF_NEFF_SEO1_i.DAT",sep=" ",index=False)

fout=open("SPEC_EFF_NEFF_SEO1.DAT","w")
fout.write("# TABLE 1 -- r\n")
fout.write("NVAR: 2 \n")
for line in open("SPEC_EFF_NEFF_SEO1_r.DAT"):
    fout.write(line)
fout.write("\n")
fout.write("\n")
fout.write("# TABLE 2 -- i\n")
fout.write("NVAR: 2 \n")
for line in open("SPEC_EFF_NEFF_SEO1_i.DAT"):
    fout.write(line)
fout.close()