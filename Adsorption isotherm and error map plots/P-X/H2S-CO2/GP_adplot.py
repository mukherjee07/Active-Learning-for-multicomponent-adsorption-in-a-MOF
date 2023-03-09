#!/usr/bin/env python
# coding: utf-8

#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading the file
df = pd.read_csv('adsorption_XXX.csv',delimiter=',')
#df2 = pd.read_csv('co2_cummulative.csv',delimiter=',')
#df3 = pd.read_csv('ch4_cummulative.csv',delimiter=',')

#Reading the data
p = df.iloc[:,0].values
gp_co2 = df.iloc[:,1].values
gp_ch4 = df.iloc[:,2].values
real_co2 = df.iloc[:,3].values
error_co2 = df.iloc[:,5].values
#2sigma
error_co2 = 2*error_co2
error_ch4 = df.iloc[:,6].values
error_ch4 = 2*error_ch4
#Taking the error data as well
real_ch4 = df.iloc[:,4].values

#K4_Error = df.iloc[:,4].values
#print(Iast_co2)
#Doing the plotting
# set the font globally
#plt.rcParams.update({"font.family": "Times"})
marker_size = 55
fig1=plt.figure()
fig1.set_figwidth(15)
fig1.set_figheight(11)
plt.rc('xtick',labelsize=44)
plt.rc('ytick',labelsize=44)
#plt.rcParams.update({'errorbar.capsize':3})
#plotting for 100 K
#plt.plot(Pressure100,GP_100, marker = "H",color="r",Linestyle="solid",label="GP (T = 100 K)")

plt.plot(p,real_co2, marker = "o", markersize='10', markeredgecolor='k',linewidth='5',color = "cyan", linestyle = ":",label="$H_{2}S$ (GCMC)")
plt.plot(p,gp_co2, marker = "o", markersize='10', markeredgecolor='k', linewidth='5',color = "grey", linestyle = ":",label="${H_{2}S}$ (GP)")
plt.fill_between(p, real_co2 - (error_co2), real_co2 + error_co2,color='gray', alpha=0.2,label="95% Confidence")
#plt.errorbar(c, H_{2}S,  yerr = H_{2}S_Error, label="$\sigma_{GCMC}$",color = 'k', linestyle='')
#plt.xscale('log')
#plotting for 202 K
#plt.plot(Pressure202,GP_202, marker = "H",color="m",Linestyle="solid",label="GP (T = 202 K)")

plt.plot(p,real_ch4, marker = "s", markersize='10',markeredgecolor='k',linewidth='5', color = "red", linestyle = "-.",label="$CO_{2}$ (GCMC)")
plt.plot(p,gp_ch4, marker = "s", markersize='10', markeredgecolor='k',linewidth='5',color = "orange", linestyle = ":",label="$CO_{2}$ (GP)")
plt.fill_between(p, real_ch4 - (error_ch4), real_ch4 + (error_ch4),color='Grey', alpha=0.2)
#plt.errorbar(c, K4,  yerr = CH4_Error, label="$\sigma_{GCMC}$",color = 'b', linestyle='')
plt.xlabel("Pressure (in bar)", fontsize = "36")
plt.title("Adsorption isotherm at $X_{H_{2}S}$ = XXX",fontsize = "36")
plt.grid(visible=True,which='minor',color='#999999',linestyle='-',alpha=0.2)
#plt.ylabel("$CO_2$ adsorption in molecules/cylinder vol.($\AA^3$)", fontsize = "20")
plt.ylabel("$H_{2}S$ and $CO_{2}$ uptake in mg/g", fontsize = "36")
plt.legend(loc="best",prop={'size': 24})
plt.grid(True,axis='both',which='minor',linestyle='--', linewidth=2)
plt.grid(True,axis='both',which='major',linestyle='--', linewidth=1.5)
#plt.xscale('log')
plt.ylim(0,750)
plt.savefig('AL_limit_Adsorption_H_{2}S_XXX.png',dpi=300)
plt.show()


