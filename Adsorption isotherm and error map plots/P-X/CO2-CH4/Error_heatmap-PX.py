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
#import numpy as np


#Importing dataframe
df1 = pd.read_csv('error.csv',delimiter=',')
errors_co2 = df1.pivot("Pressure","mole_fraction","Error_h2s")
fig, ax = plt.subplots(figsize=(37.5,16.5))
ax = sns.heatmap(errors_co2,cmap='BrBG',lw=1,linecolor='k',vmax=20.0,vmin=-20.0)
ax.set_ylabel('Pressure (in bar) $\Longrightarrow$',size='44')
ax.invert_yaxis()
ax.set_xlabel('Mole fraction $X_{CO_{2}}$ $\Longrightarrow$',size='44',rotation=0)
plt.text(56.00,12.5,'Relative Error (in %)  $\Longrightarrow$', rotation=90,fontsize='44')
plt.savefig('CO2_RR.png',dpi=300)
#df1 = pd.read_csv('error.csv',delimiter=',')
errors_ch4 = df1.pivot("Pressure","mole_fraction","Error_co2")
fig, ax = plt.subplots(figsize=(37.5,16.5))
ax = sns.heatmap(errors_ch4,cmap='BrBG',lw=1,linecolor='k',vmax=20.0,vmin=-20.0)
ax.invert_yaxis()
ax.set_ylabel('Pressure (in bar) $\Longrightarrow$',size='44')
ax.set_xlabel('Mole fraction $X_{CH_{4}}$  $\Longrightarrow$',size='44',rotation=0)
plt.text(56.00,12.5,'Relative Error (in %)  $\Longrightarrow$', rotation=90,fontsize='44')
plt.savefig('CH4_RR.png',dpi=300)
#ax.set_yticklabels(['200','220','240','260','280','300','320','340','360','380','400'],rotation=0)
#ax.set_xticklabels(['$10^{-6}$','15','30','45','60','75','90','105','120','135','150','165','180','195','210','225','240','255','270','275','300'],rotation=0)
'''
errors_ch4 = df1.pivot("Temp","Pressure","error_ch4")

fig2, ax2 = plt.subplots(figsize=(31,13.5))
ax2 = sns.heatmap(errors_ch4,lw=4,cmap='BrBG',vmax=20,vmin=-20)
plt.text(24.00,8.5,'Relative Error (in %)', rotation=90,fontsize='44')
ax2.set_yticklabels(['200','220','240','260','280','300','320','340','360','380','400'],rotation=0)
ax2.set_xticklabels(['$10^{-6}$','15','30','45','60','75','90','105','120','135','150','165','180','195','210','225','240','255','270','275','300'],rotation=0)
#ax2.set_xticklabels(['$10^{-6}$','','30','','60','','90','','120','','150','','180','','210','','240','','270','','300'],rotation=0)
#ax2.set_ylabel('Temperature (in K)',size='44')
#ax2.set_xlabel('Pressure (in bar)',rotation=0,fontsize='44')
plt.savefig('CH4_0.596.png',dpi=300)
'''
'''
flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
ax = sns.heatmap(flights)
print(flights)
'''     