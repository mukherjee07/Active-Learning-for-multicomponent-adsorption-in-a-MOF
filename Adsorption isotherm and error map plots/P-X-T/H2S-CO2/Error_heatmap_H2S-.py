#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:54:05 2022

@author: krishnendumukherjee
"""
#Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#a=np.linspace(10**(-6),100,21)
#print(round(a,2))
X=[0.02,0.116,0.212,0.308,0.404,0.5,0.596,0.692,0.788,0.884,0.98]
#'''
for i in range(len(X)):
   Y=round(1-X[i],2)
   #Importing dataframe
   df1 = pd.read_csv('error_'+str(X[i])+'.csv',delimiter=',')
   errors_co2 = df1.pivot("Temp","Pressure","error_h2s")
   fig, ax = plt.subplots(figsize=(31,13.5))
   ax = sns.heatmap(errors_co2,lw=4,linecolor='k',cmap='BrBG',vmax=20,vmin=-20)
   ax.set_ylabel('Temperature (in K) $\Longrightarrow$',size='48')
   ax.set_xlabel('Pressure (in bar) $\Longrightarrow$',size='48',rotation=0)
   ax.invert_yaxis()
   plt.text(24.00,2.5,'Relative Error (in %)', rotation=90,fontsize='48')
   ax.set_yticklabels(['200','220','240','260','280','300','320','340','360','380','400'],size='24',rotation=0)
   ax.set_xticklabels(['$10^{-6}$','5','10','15','20','25','30','35','40','45','50','55','60','65','70','75','80','85','90','95','100'],size='24',rotation=0)
   ax.set_title('Relative error heat map for $X_{H_{2}S}$ at $X_{H_{2}S}$ = '+str(X[i])+', and $X_{CO_{2}}$ = '+str(Y),size='44')
   plt.savefig('H2S_'+str(X[i])+'.png',dpi=300)
   errors_ch4 = df1.pivot("Temp","Pressure","error_co2")

   fig2, ax2 = plt.subplots(figsize=(31,13.5))
   ax2 = sns.heatmap(errors_ch4,lw=4,linecolor='k',cmap='BrBG',vmax=20,vmin=-20)
   plt.text(24.00,2.5,'Relative Error (in %)', rotation=90,fontsize='44')
   ax2.set_yticklabels(['200','220','240','260','280','300','320','340','360','380','400'],size='24',rotation=0)
   ax2.set_xticklabels(['$10^{-6}$','5','10','15','20','25','30','35','40','45','50','55','60','65','70','75','80','85','90','95','100'],size='24',rotation=0)
   #ax2.set_xticklabels(['$10^{-6}$','','30','','60','','90','','120','','150','','180','','210','','240','','270','','300'],rotation=0)
   ax2.set_ylabel('Temperature (in K) $\Longrightarrow$',size='44')
   ax2.set_xlabel('Pressure (in bar) $\Longrightarrow$',rotation=0,fontsize='44')
   ax2.set_title('Relative error heat map for $X_{CO_{2}}$ at $X_{H_{2}S}$ = '+str(X[i])+', and $X_{CO_{2}}$ = '+str(Y),size='44')
   ax2.invert_yaxis()
   plt.savefig('CO2_'+str(Y)+'.png',dpi=300)

#'''
'''
flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
ax = sns.heatmap(flights)
print(flights)
'''     