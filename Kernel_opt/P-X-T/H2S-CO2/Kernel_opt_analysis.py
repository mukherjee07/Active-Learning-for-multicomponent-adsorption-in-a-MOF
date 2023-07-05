#!/usr/bin/env python
# coding: utf-8

##importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


#Reading the file
#df4 = pd.read_csv('CO2_CH4_ACCURACY/mean.csv',delimiter=',')
#df2 = pd.read_csv('CO2_CH4_Accuracy_0.01/mean.csv',delimiter=',')
#df3 = pd.read_csv('Xe-Kr-Regres_1/mean.csv',delimiter=',')
#df5 = pd.read_csv('CO2_CH4_Regree_Accuracy_1/mean.csv',delimiter=',')
df = pd.read_csv('Kernel_500_PTX.csv',delimiter=',')
#df5 = pd.read_csv('RBF+RQ/mean.csv',delimiter=',')
##df6 = pd.read_csv('RQ+RQ/mean.csv',delimiter=',')
#df7 = pd.read_csv('IE+CO2/mean.csv',delimiter=',')
#df8 = pd.read_csv('IE+CH4/mean.csv',delimiter=',')
#max_co2 = df.iloc[:,0].values
#max_ch4 = df.iloc[:,1].values
Iteration = df.iloc[:,0].values
#Taking the error data as well
MRE_ch4 = df.iloc[:,8].values
MRE_co2 = df.iloc[:,7].values
GPMRE_ch4 = df.iloc[:,2].values
GPMRE_co2 = df.iloc[:,1].values
AC_co2 = df.iloc[:,11].values
AC_ch4 = df.iloc[:,12].values

Del_ch4 = abs(MRE_ch4 - GPMRE_ch4)/np.max(abs(MRE_ch4 - GPMRE_ch4))
Del_co2 = abs(MRE_co2 - GPMRE_co2)/np.max(abs(MRE_co2 - GPMRE_co2))
Del_AC = (abs(AC_co2-AC_ch4)-np.min(abs(AC_co2-AC_ch4)))/(np.max(abs(AC_co2-AC_ch4))-np.min(abs(AC_co2-AC_ch4)))
mre_ch4 = (MRE_ch4)/(np.amax(MRE_ch4))
mre_co2 = (MRE_co2)/(np.amax(MRE_co2))
print(np.max(MRE_co2))
print("MRE",mre_ch4,mre_co2)
Delco2_90 = (abs(AC_co2-100)-np.min(abs(AC_co2-100)))/(np.max(abs(AC_co2-100))-np.min(abs(AC_co2-100)))
Delch4_90 = (abs(AC_ch4-100)-np.min(abs(AC_ch4-100)))/(np.max(abs(AC_ch4-100))-np.min(abs(AC_ch4-100)))
print(Delco2_90,Delch4_90)
#print(MRE_ch4,MRE_co2,Iteration)
Iteration = ((Iteration+54)/2499)*100
PERFORMANCE = (Delco2_90+Delch4_90+Del_AC+mre_co2+mre_ch4)/5
#PERFORMANCE = (MRE_co2+MRE_ch4)/(1+1)
#print(len(MRE_ch4),len(MRE_co2),len(Iteration))
#print(MRE_ch4,MRE_co2,Iteration)
#AGP_co2 = df.iloc[:,8].values
#AGPMRE_ch4 = df.iloc[:,1].values
#Taking the error data as well
#AGP_ch4 = df.iloc[:,9].values
'''
B_co2 = df2.iloc[:,6].values
B_ch4 = df2.iloc[:,7].values
BGP_co2 = df2.iloc[:,0].values
BGP_ch4 = df2.iloc[:,1].values

#X_std
C_co2 = df3.iloc[:,6].values
C_ch4 = df3.iloc[:,7].values
CGP_co2 = df3.iloc[:,0].values
CGP_ch4 = df3.iloc[:,1].values
#
#Y_std
D_co2 = df4.iloc[:,6].values
DGP_co2 = df4.iloc[:,0].values
#print(D_co2)
D_ch4 = df4.iloc[:,7].values
DGP_ch4 = df4.iloc[:,1].values
'''
#M_dot
'''
E_co2 = df5.iloc[:,6].values
#M_dot
E_ch4 = df5.iloc[:,7].values
EGP_co2 = df5.iloc[:,0].values
#M_dot
EGP_ch4 = df5.iloc[:,1].values
#print(E_co2)
#E_ch4 = df5.iloc[:,7].values
F_co2 = df6.iloc[:,6].values
#print(E_co2)
F_ch4 = df6.iloc[:,7].values
FGP_co2 = df6.iloc[:,0].values
#print(E_co2)
FGP_ch4 = df6.iloc[:,1].values
G_co2 = df7.iloc[:,3].values
#print(E_co2)
#G_ch4 = df7.iloc[:,7].values
GGP_co2 = df7.iloc[:,0].values
GGP_Max_co2 = df7.iloc[:,1].values
#print(E_co2)
#GGP_ch4 = df7.iloc[:,1].values
#I_co2 = df8.iloc[:,6].values
#print(E_co2)
I_ch4 = df8.iloc[:,3].values
#IGP_co2 = df8.iloc[:,0].values
#print(E_co2)
IGP_ch4 = df8.iloc[:,0].values
IGP_Max_ch4 = df8.iloc[:,1].values
'''

#CH4_Error = df.iloc[:,4].values
LEN=len(df)
#print(LEN)

Iter=np.zeros(LEN)
MAX = np.max(Iteration)
for i in range(LEN):
    Iter[i] = 1+i
    #print(Iter[i])
#print(len(Iter))    
    
'''
#Doing the plotting
marker_size = 55
fig1=plt.figure()
fig1.set_figwidth(12)
fig1.set_figheight(8)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
#plt.rcParams.update({'errorbar.capsize':3})
#plotting for 100 K
#plt.plot(Pressure100,GP_100, marker = "H",color="r",Linestyle="solid",label="GP (T = 100 K)")
#plt.plot(Iter,R2_co2, marker = "*", color = "b", Linestyle = "-.",label="Matern $R^{2}_{CO_2}$")
#plt.errorbar(c, CO2,  yerr = CO2_Error, label="$\sigma_{GCMC}$",color = 'k', linestyle='')
#plt.xscale('log')
#plotting for 202 K
#plt.plot(Pressure202,GP_202, marker = "H",color="m",Linestyle="solid",label="GP (T = 202 K)")
plt.plot(Iter,A_ch4[:I], marker = "*", color = "green", Linestyle = "-.",label="M+M --> Mixture space, No Submodel")
plt.plot(Iter,AGP_ch4[:I], marker = "*", color = "blue", Linestyle = "-.",label="M+M --> Mixture space, No Submodel")
#Dual Matern
#plt.plot(Iter,R2_co2_dk, marker = "1", color = "red", Linestyle = "-.",label="Dual Matern $\u03C3_{CO_2}$")
#plt.plot(Iter,B_ch4[:I], marker = "1", color = "orange", Linestyle = "--",label="M+RBF --> Mixture space, No Submodel")
#plt.plot(Iter,BGP_co2[:I], marker = "1", color = "green", Linestyle = "--",label="GP M+RBF --> Mixture space, No Submodel")
#X_std
#plt.plot(Iter,R2_co2_xs, marker = "2", color = "black", Linestyle = "-.",label="X_std $\u03C3_{CO_2}$")
#plt.plot(Iter,C_ch4[:I], marker = "2", color = "black", Linestyle = "-.",label="M+RQ --> Mixture space, No Submodel")
#plt.plot(Iter,CGP_co2[:I], marker = "2", color = "brown", Linestyle = "-.",label="GP M+RQ --> Mixture space, No Submodel")

#Y_norm
#plt.plot(Iter,R2_co2_ys, marker = "3", color = "cyan", Linestyle = "-.",label="Y_norm $\u03C3_{CO_2}$")
#plt.plot(Iter,D_ch4[:I], marker = "3", color = "cyan", Linestyle = "-.",label="RBF+RBF --> Mixture space, No Submodel")
#plt.plot(Iter,DGP_ch4[:I], marker = "3", color = "red", Linestyle = "-.",label="GP RBF+RBF --> Mixture space, No Submodel")
#kernel_dot
#plt.plot(Iter,R2_co2_md, marker = "8", color = "brown", Linestyle = "-.",label="Matern dot $\u03C3_{CO_2}$")
#plt.plot(Iter,E_ch4[:I], marker = "8", color = "brown", Linestyle = "-.",label="RBF+RQ --> Mixture space, No Submodel")
#plt.plot(Iter,F_ch4[:I], marker = "8", color = "red", Linestyle = "-.",label="RQ+RQ --> Mixture space, No Submodel")
#plt.plot(Iter,IGP_Max_ch4[:I], marker = "8", color = "brown", Linestyle = "-.",label="GP max Indepen. learning CO2 only --> Mixture space, No Submodel")
#plt.plot(Iter,IGP_ch4[:I], marker = "8", color = "k", Linestyle = "-.",label="GP Indepen. learning CO2 only --> Mixture space, No Submodel")
#plt.plot(Iter,I_ch4[:I], marker = "8", color = "green", Linestyle = "-.",label="Indepen. learning CH4 only --> Mixture space, No Submodel")
#plt.plot(Iter,IGP_ch4[:I], marker = "8", color = "k", Linestyle = "-.",label="GP Indepen. learning CH4 only --> Mixture space, No Submodel")
plt.xlabel("Number of Active learning iterations", fontsize = "28")
#plt.title("$CH_4$ adsorption isotherm with a 4-point prior sampled from Log-based LHS for Cu-BTC MOF",fontsize = "20")
plt.grid(b=True,which='minor',color='#999999',linestyle='-',alpha=0.2)
#plt.ylabel("$CO_2$ adsorption in molecules/cylinder vol.($\AA^3$)", fontsize = "20")
plt.ylabel("Accuracy for $CH_4$ (in %)", fontsize = "28")
plt.legend(loc="best",prop={'size': 18})
plt.ylim(0,100)
#plt.xlim(0,1000)
#plt.xscale('log')
#plt.savefig('Uncertaintyratio-LowP-CH4.png',dpi=600)
plt.show()
'''
# Create figure and axis #1

fig, host = plt.subplots(figsize=(17,9)) 
#ax.set_facecolor("yellow")
plt.rc('xtick',labelsize=34)# (width, height) in inches
plt.rc('ytick',labelsize=34)
plt.rc('ytick',labelsize=34)
# (see https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.subplots.html)
    
par1 = host.twinx()
par2 = host.twinx()
host.set_facecolor("white")
host.set_xlim(0, 39)
host.set_ylim(0.0, 1.0001)
par1.set_ylim(0, 1.0)
par2.set_ylim(0, 1.0)
    
host.set_xlabel("Kernel combinations",fontsize="34")
plt.rc('xtick',labelsize=28)
host.set_ylabel("AL Performance in P-X-T",fontsize="34",color="black")
par1.set_ylabel(r"All performance metrics",fontsize="34",color="black")
#par2.set_ylabel(r"${\overline{MRE}}$ for $CO_2$ in %",fontsize="28",color="black")

color1 = "k"
color2 = "green"
color3 = "blue"
p1, = host.plot(Iter, PERFORMANCE, markeredgecolor='k',linestyle='-.',marker='D',markersize='20',label=r"Performance = $\frac{{\overline{100-AC_{H_{2}S}}}+{\overline{100-AC_{CO_2}}}+{\overline{AC_{CL}}}+{\overline{MRE_{H_{2}S}}}+{\overline{MRE_{CO_2}}}}{5}$",color="silver",linewidth=5)
p11, = host.plot(Iter, Delco2_90,markeredgecolor='k', marker='*',linestyle=':',markersize='12',label=r"${\overline{100-AC_{H_{2}S}}}$",color="cyan",linewidth=5)
p12, = host.plot(Iter, Delch4_90,markeredgecolor='k', marker='H',linestyle='-.',markersize='12',label=r"${\overline{100-AC_{CO_2}}}$",color="turquoise",linewidth=5)
p13, = host.plot(Iter, Del_AC,markeredgecolor='k', marker='H',linestyle='-.',markersize='12',label=r"${\overline{AC_{CL}}}$",color="orange",linewidth=5)
p2, = par1.plot(Iter, mre_co2, markeredgecolor='k',linestyle='-',marker='o',markersize='12',label=r"${\overline{MRE}}$ for $H_{2}S$",color="lightgreen",linewidth=3)
#p22, = par1.plot(Iter, A_ch4[:I], linestyle=':',marker='o',label="MRE for $CH_4$ in %",color="cyan")
p3, = par2.plot(Iter, mre_ch4,linestyle=':',markeredgecolor='k', marker='o',markersize='12',label=r"${\overline{MRE}}$ for $CO_2$",color="salmon",linewidth=3)
#p33, = par2.plot(Iter, AGPMRE_ch4[:I], linestyle='-.',label="GP MRE for $CH_4$ in %",color="red")
lns = [p1,p11,p12,p2, p3]
host.legend(handles=lns)

# right, left, top, bottom
#par2.spines['right'].set_position(('outward', 90))

# no x-ticks                 
#par2.xaxis.set_ticks()

# Sometimes handy, same for xaxis
#par2.yaxis.set_ticks_position('right')

# Move "Velocity"-axis to the left
# par2.spines['left'].set_position(('outward', 60))
# par2.spines['left'].set_visible(True)
# par2.yaxis.set_label_position('left')
# par2.yaxis.set_ticks_position('left')

#host.yaxis.label.set_color(p1.get_color())
#par1.yaxis.label.set_color(p2.get_color())
#par2.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
major_ticks = np.arange(0, Iter[LEN-1], 2)
minor_ticks = np.arange(0, Iter[LEN-1], 0.25)
majory_ticks = np.arange(0.0, 1.0001, .10)
minory_ticks = np.arange(0.0, 1.0001, .02)
host.set_xticks(major_ticks)
host.set_xticks(minor_ticks, minor=True)
host.set_yticks(majory_ticks)
host.set_yticks(minory_ticks, minor=True)

# And a corresponding grid
host.grid(which='both')

# Or if you want different settings for the grids:
host.grid(which='minor', alpha=0.75,linestyle=':')
host.grid(which='major', alpha=1.5,linestyle='--')

host.tick_params(axis='y', colors="black", **tkw)
#host.grid(True,alpha=0.5)
par1.tick_params(axis='y', colors="black", **tkw)
par2.tick_params(axis='y', colors="black", **tkw)
host.tick_params(axis='x', colors="black")

lines = [p1,p11,p12,p13,p2 ,p3]

host.legend(lines, [l.get_label() for l in lines],fontsize="34", bbox_to_anchor=(1.1150,1) )
# Adjust spacings w.r.t. figsize
fig.tight_layout()
# Alternatively: bbox_inches='tight' within the plt.savefig function 
#                (overwrites figsize)

# Best for professional typesetting, e.g. LaTeX
plt.show()
# Fo
