#!/usr/bin/env python
# coding: utf-8

#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

#importing the ML libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel 
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.metrics import r2_score

#Reading the dataset
df = pd.read_csv('prior_Sample.csv',delimiter=',')
df2 = pd.read_csv('complete.csv',delimiter=',')

#### TESTING THE HEAD OF THE DATASET (Each testing section should be commented out) ####
#print(df.head())
#### TESTING DONE ####

#Limits of pressure array
p_min = 1.0e-6 #max and min pressure
p_max = 2.0e+2

#Converting pressure input to log base 10 for log-distribution
Y = np.log10(p_max)
X = np.log10(p_min)

#Temp range (in K)
T_min = 200
T_max = 400

#Mole fraction limits
X_min = 0.02
X_max = 0.98

#Sizes of input matrices
P_size = 21  #pressure vector size
T_size = 11  #temperature vector size
X1_size = 11  #mole fraction (methane) vector size
X2_size = 11  #mole fraction (carbon dioxide) vector size

#Unseen array
P_test = np.atleast_2d(np.linspace(p_min,p_max,P_size)).flatten().reshape(-1,1)
T_test = np.atleast_2d(np.linspace(T_min,T_max,T_size)).flatten().reshape(-1,1)
X1_test = np.atleast_2d(np.linspace(X_min,X_max,X1_size)).flatten().reshape(-1,1)
X2_test = np.atleast_2d(np.linspace(X_max,X_min,X2_size)).flatten().reshape(-1,1)

#### FOR TESTING ####
''' 
FOR generating Prior dataset for comparision 
for i in range(len(X_test)):
	os.system("echo "+str(X_test[i])+" >> Prior.csv")
'''
#### TESTING DONE ####

#Reading the data from prior
p = df.iloc[:,0].values
t = df.iloc[:,1].values
x1 = df.iloc[:,2].values
x2 = df.iloc[:,3].values
y1 = df.iloc[:,4].values
y2 = df.iloc[:,5].values

#Taking the error data as well
e1 = df.iloc[:,6].values
e2 = df.iloc[:,7].values

#From ground-truth dataset (*_g shows ground truth)
p_g = df2.iloc[:,0].values
t_g = df2.iloc[:,1].values
x1_g = df2.iloc[:,2].values
x2_g = df2.iloc[:,3].values
y1_g = df2.iloc[:,4].values
y2_g = df2.iloc[:,5].values
e1_g = df2.iloc[:,6].values
e2_g = df2.iloc[:,7].values

#Replacing y if some y value in zero
for i in range(len(y1)):
	if (y1[i] == 0):
		y1[i] = 0.0001

#For y2
for i in range(len(y2)):
	if (y2[i] == 0):
		y2[i] = 0.0001

#Transforming 1D arrays to 2D
p = np.atleast_2d(p).flatten().reshape(-1,1)
t = np.atleast_2d(t).flatten().reshape(-1,1)
x1 = np.atleast_2d(x1).flatten().reshape(-1,1)
x2 = np.atleast_2d(x2).flatten().reshape(-1,1)
y1 = np.atleast_2d(y1).flatten()
y2 = np.atleast_2d(y2).flatten()

#converting P to bars
p = p/(1.0e5)

#Taking logbase 10 of the input vector
p = np.log10(p)
t = np.log10(t)
y1 = np.log10(y1)
y2 = np.log10(y2)

#print(len(x),len(y))
#Taking the log of X_test
P_test = np.log10(P_test)
T_test = np.log10(T_test)

#Extracting the mean and std. dev for P_test
p_m = np.mean(P_test)
p_std = np.std(P_test,ddof=1)

#Extracting the mean and std. dev for T_test
t_m = np.mean(T_test)
t_std = np.std(T_test,ddof=1)

#Standardising p,t and y in log-space
p_s = (p - p_m)/p_std
t_s = (t - t_m)/t_std

#Standardising X_test in log-space
P_test = (P_test - p_m)/p_std
T_test = (T_test - t_m)/t_std

## x_s1 and x_s2 are scaled inputs to GP1 and GP2
x_s1 = np.zeros((len(p_s),3))
x_s2 = np.zeros((len(p_s),3))

#defining X_test size
test_size = (P_size)*(T_size)*(X1_size)
X_test_co2 = np.zeros((test_size,3))
X_test_ch4 = np.zeros((test_size,3))

#Filling all the data in training and prediction set
for i in range(len(p_s)):
	for k in range(3):
		#Inserting pressure for the first column
		if k == 0:
			x_s1[i,k] = p_s[i]
			x_s2[i,k] = p_s[i]
		elif k == 1:
		#Inserting temperature for the second column
			x_s1[i,k] = t_s[i]
			x_s2[i,k] = t_s[i]
		else:
		#Inserting methane and CO2 fraction for the second column
			x_s1[i,k] = (x1[i] - 0.50)*(25/12)
			x_s2[i,k] = (x2[i] - 0.50)*(25/12)

#for i in range(len(p_s)):
#	print(x_s1[i],x_s2[i])

Loc = 0
### X_test using CONVENTIONAL METHOD ###
for i in range(len(P_test)):
	for j in range(len(T_test)):
		for k in range(len(X1_test)):
			#Inserting pressure for the first column
			X_test_co2[Loc,0] = P_test[i]
			X_test_ch4[Loc,0] = P_test[i]
			X_test_co2[Loc,1] = T_test[j]
			X_test_ch4[Loc,1] = T_test[j]
			X_test_co2[Loc,2] = (X1_test[k]-0.50)*(25/12)
			X_test_ch4[Loc,2] = (X2_test[k]-0.50)*(25/12)
			#print(X_test[Loc])
			#Updating the location for next iteration
			Loc = Loc + 1

#print(x_s,y)
#Building the GP regresson 

# Instantiate a Gaussian Process model
# 3 kernels for three features
kernel = RBF(length_scale=1,length_scale_bounds=(1e-15,1e15))+RBF(length_scale=1,length_scale_bounds=(1e-15,1e15))+RBF(length_scale=1,length_scale_bounds=(1e-15,1e15))

#creating the GP instances for co2 and ch4
gp_co2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=1000, normalize_y=True)
gp_ch4 = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=1000, normalize_y=True)

#print(type(x_s),type(y.T))
#print(y.T)
#Fitting our standardized data to the GP model
gp_co2.fit(x_s1,y1.T)
gp_ch4.fit(x_s2,y2.T)

'''
### TEST FOR COMPARING SCALED INPUTS ###
### Testing for similarity in scaled prior input, uptake and X_test input ###
for i in range(len(x_s1)):
	print("Scaled input and output for CO_2",x_s1[i],y1[i])
	print("Scaled input and output for CH_4",x_s2[i],y2[i])

for i in range(len(X_test_co2)):
	print("X_test for CO2",X_test_co2[i])
	print("X_test for CH4",X_test_ch4[i])
### TESTING DONE ###
'''

# Make the prediction on the test data (ask for MSE as well)
y1_pred, sigma_co2 = gp_co2.predict(X_test_co2, return_std=True)
y2_pred, sigma_ch4 = gp_ch4.predict(X_test_ch4, return_std=True)

#Params = gp.get_params()
#print(Params)

### print TEST ###
#printing them
#print(y1_pred,sigma1,y2_pred,sigma2)
### Test DONE ###

#Declaring the error variables

# 1. Relative error
rel_error_co2 = np.zeros(len(X_test_co2))
rel_error_ch4 = np.zeros(len(X_test_ch4))

#finding the relative error—

#defining the accuracy parameter
AC_co2 = 0
AC_ch4 = 0

#Confident and un-confident in GPs
C_co2 = 0
NC_co2 = 0
C_ch4 = 0
NC_ch4 = 0

#finding the relative error—
for i in range(len(X_test_co2)):
    # Find GP-predicted relative error
    rel_error_co2[i] = abs(sigma_co2[i]/abs(y1_pred[i]))
    rel_error_ch4[i] = abs(sigma_ch4[i]/abs(y2_pred[i]))
    if (rel_error_co2[i] <= 0.02):
        C_co2 += rel_error_co2[i] 
    else:
        NC_co2 += rel_error_co2[i]
    if (rel_error_ch4[i] <= 0.02):
        C_ch4 += rel_error_ch4[i] 
    else:
        NC_ch4 += rel_error_ch4[i]

AC_co2 = 100*C_co2/(C_co2 + NC_co2)
AC_ch4 = 100*C_ch4/(C_ch4 + NC_ch4)

#defining the accuracy parameter-based on classification
ACCL_co2 = 0
ACCL_ch4 = 0

#Confident and un-confident in GPs
C_co2 = 0
NC_co2 = 0
C_ch4 = 0
NC_ch4 = 0

#finding the relative error—
for i in range(len(X_test_co2)):
    # Find GP-predicted relative error
    if (rel_error_co2[i] <= 0.05):
        C_co2 += 1 
    else:
        NC_co2 += 1
    if (rel_error_ch4[i] <= 0.05):
        C_ch4 += 1 
    else:
        NC_ch4 += 1

ACCL_co2 = 100*C_co2/(C_co2 + NC_co2)
ACCL_ch4 = 100*C_ch4/(C_ch4 + NC_ch4)
#converting back to original after calculating the relative error
y1_pred = 10**(y1_pred)
y2_pred = 10**(y2_pred)

# 2. Coefficient of correlation for co2 and ch4
r2_co2 = r2_score(y1_g,y1_pred)
r2_ch4 = r2_score(y2_g,y2_pred)

#define the limit for uncertainty
lim = 2

#Converting X_test to real values pf pressure and temperature for comparing them in future
# 1. Converting Pressure
X_test_co2[:,0] = (X_test_co2[:,0]*p_std) + p_m
X_test_ch4[:,0] = (X_test_ch4[:,0]*p_std) + p_m
X_test_co2[:,0] = 10**(X_test_co2[:,0])
X_test_ch4[:,0] = 10**(X_test_ch4[:,0])
X_test_co2[:,0] = 1e5*(X_test_co2[:,0])
X_test_ch4[:,0] = 1e5*(X_test_ch4[:,0])

# 2. Converting the temperature back to the original
X_test_co2[:,1] = (X_test_co2[:,1]*t_std) + t_m
X_test_ch4[:,1] = (X_test_ch4[:,1]*t_std) + t_m
X_test_co2[:,1] = 10**(X_test_co2[:,1])
X_test_ch4[:,1] = 10**(X_test_ch4[:,1])

#converting the mole-frac back to original
X_test_co2[:,2] = (X_test_co2[:,2]/(25/12)) + 0.50
X_test_ch4[:,2] = (X_test_ch4[:,2]/(25/12)) + 0.50
#Finding the maximum GP rel error value and the location

#converting P_test back
P_test = (P_test*p_std) + p_m
P_test = 10**(P_test)
T_test = (T_test*t_std) + t_m
T_test = 10**(T_test)
GP_co2 = np.ones((P_size,T_size,X1_size))
GP_ch4 = np.ones((P_size,T_size,X2_size))

#GP_ch4 = [[ 0 for i range(P_size)] for j in range(X1_size)]

#Counter for storing error values for 100K, 151K, 202K, 253K, 300K
C_100 = 0

#TEstinf X_test and y2, and y_pred
#print(len(X_test),len(y_pred))
for i in range(X1_size):
    for j in range(T_size):
        mol = str(X1_test[i])
        mol = mol.replace("]","")
        mol = mol.replace("[","")
        T = str(T_test[j])
        T = T.replace("[","")
        T = T.replace("]","")
        T = T.replace(".","")
        os.system("rm adsorption_"+str(mol)+"_"+str(T)+".csv")
        os.system("touch adsorption_"+str(mol)+"_"+str(T)+".csv")
        os.system("echo 'Pressure,Temperature,X_Xe,GP_Xe,GP_Kr,Actual_Xe,Actual_Kr,error_Xe,error_Kr' >> adsorption_"+str(mol)+"_"+str(T)+".csv")
        for k in range(P_size):
            Loc = (k*X1_size*T_size) + (j*T_size) + i
            GP_co2[k,j,i] = y1_pred[Loc]
            GP_ch4[k,j,i] = y2_pred[Loc]
            #print(Loc,X1_test[i],T_test[j],P_test[k],X_test_co2[Loc],y1_pred[Loc],y2_pred[Loc])
            os.system("echo "+str(P_test[k])+","+str(T_test[j])+","+str(X1_test[i])+","+str(GP_co2[k,j,i])+","+str(GP_ch4[k,j,i])+","+str(y1_g[Loc])+","+str(y2_g[Loc])+","+str(e1_g[Loc])+","+str(e2_g[Loc])+" >> adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
            os.system("sed -i 's/[][]//' adsorption_"+str(mol)+"_"+str(T)+".csv")
#### END OF CODE ####
