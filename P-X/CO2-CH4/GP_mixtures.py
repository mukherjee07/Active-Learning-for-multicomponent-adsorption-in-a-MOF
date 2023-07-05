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
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel 
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.metrics import r2_score

#Reading the dataset
df = pd.read_csv('training.csv',delimiter=',')
df2 = pd.read_csv('complete.csv',delimiter=',')

#### TESTING THE HEAD OF THE DATASET (Each testing section should be commented out) ####
#print(df.head())
#### TESTING DONE ####

#Limits of pressure array
p_min = 1.0e-6 #max and min pressure
p_max = 3.0e+2

#Converting pressure input to log base 10 for log-distribution
Y = np.log10(p_max)
X = np.log10(p_min)

#Mole fraction limits
X_min = 0.02
X_max = 0.98

#Sizes of input matrices
P_size = 51  #pressure vector size
X1_size = 49  #mole fraction (methane) vector size
X2_size = 49  #mole fraction (carbon dioxide) vector size

#Unseen array
P_test = np.atleast_2d(np.linspace(p_min,p_max,P_size)).flatten().reshape(-1,1)
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

#Replacing y if some y value in zero
for i in range(len(y1)):
	if (y1[i] == 0):
		y1[i] = 0.0001

#For y2
for i in range(len(y2)):
	if (y2[i] == 0):
		y2[i] = 0.0001

### Changing y's zero to small non-zero
'''
for i in range(len(y1_g)):
	if (y1_g[i] == 0):
		y1_g[i] = 0.0000000001

for i in range(len(y2_g)):
	if (y2_g[i] == 0):
		y2_g[i] = 0.0000000001
'''
	
#Transforming 1D arrays to 2D
p = np.atleast_2d(p).flatten().reshape(-1,1)
x1 = np.atleast_2d(x1).flatten().reshape(-1,1)
x2 = np.atleast_2d(x2).flatten().reshape(-1,1)
y1 = np.atleast_2d(y1).flatten()
y2 = np.atleast_2d(y2).flatten()

#converting P to bars
p = p/(1.0e5)

#Taking logbase 10 of the input vector
p = np.log10(p)
y1 = np.log10(y1)
y2 = np.log10(y2)

#print(len(x),len(y))
#Taking the log of X_test
P_test = np.log10(P_test)

#Extracting the mean and std. dev for P_test
p_m = np.mean(P_test)
p_std = np.std(P_test,ddof=1)

#Standardising p,t and y in log-space
p_s = (p - p_m)/p_std

#Standardising X_test in log-space
P_test = (P_test - p_m)/p_std

## x_s1 and x_s2 are scaled inputs to GP1 and GP2
x_s1 = np.zeros((len(p_s),2))
x_s2 = np.zeros((len(p_s),2))

#defining X_test size
test_size = (P_size)*(X1_size)
X_test_co2 = np.zeros((test_size,2))
X_test_ch4 = np.zeros((test_size,2))

#Filling all the data in training and prediction set
for i in range(len(p_s)):
	for k in range(2):
		#Inserting pressure for the first column
		if k == 0:
			x_s1[i,k] = p_s[i]
			x_s2[i,k] = p_s[i]
		else:
		#Inserting methane and CO2 fraction for the second column
			x_s1[i,k] = (x1[i]-0.50)*(25/12)
			x_s2[i,k] = (x2[i]-0.50)*(25/12)

#for i in range(len(p_s)):
#	print(x_s1[i],x_s2[i])

Loc = 0
### X_test using CONVENTIONAL METHOD ###
for i in range(len(P_test)):
	for k in range(len(X1_test)):
		#Inserting pressure for the first column
		X_test_co2[Loc,0] = P_test[i]
		X_test_ch4[Loc,0] = P_test[i]
		X_test_co2[Loc,1] = (X1_test[k]-0.50)*(25/12)
		X_test_ch4[Loc,1] = (X2_test[k]-0.50)*(25/12)
		#print(X_test[Loc])
		#Updating the location for next iteration
		Loc = Loc + 1

#print(x_s,y)
#Building the GP regresson 

# Instantiate a Gaussian Process model
# 3 kernels for three features
#kernel = RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-13,1e13),alpha_bounds=(1e-13,1e13))+RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-13,1e13),alpha_bounds=(1e-13,1e13))+ RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-13,1e13),alpha_bounds=(1e-13,1e13))

kernel_co2 = RBF(length_scale=1,length_scale_bounds=(1e-15,1e15)) 
kernel_ch4 = RBF(length_scale=1,length_scale_bounds=(1e-15,1e15)) 

#creating the GP instances for co2 and ch4
gp_co2 = GaussianProcessRegressor(kernel=kernel_co2, alpha=1e-4,n_restarts_optimizer=100, normalize_y=True)
gp_ch4 = GaussianProcessRegressor(kernel=kernel_ch4, alpha=1e-4,n_restarts_optimizer=100, normalize_y=True)

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

#defining the accuracy parameter
AC_co2 = 0
AC_ch4 = 0

#Confident and un-confident in GPs
C_co2 = 0
NC_co2 = 0
C_ch4 = 0
NC_ch4 = 0

#Factor to avoid diving by zero
factor=1E-3
#finding the relative error—
for i in range(len(X_test_co2)):
    # Find GP-predicted relative error
    rel_error_co2[i] = abs(sigma_co2[i]/(abs(y1_pred[i])))
    rel_error_ch4[i] = abs(sigma_ch4[i]/(abs(y2_pred[i])))
    if (rel_error_co2[i] <= 0.01):
        C_co2 += 1 
    else:
        NC_co2 += 1
    if (rel_error_ch4[i] <= 0.01):
        C_ch4 += 1
    else:
        NC_ch4 += 1

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
    if (rel_error_co2[i] <= 0.02):
        C_co2 += 1 
    else:
        NC_co2 += 1
    if (rel_error_ch4[i] <= 0.02):
        C_ch4 += 1 
    else:
        NC_ch4 += 1

ACCL_co2 = 100*C_co2/(C_co2 + NC_co2)
ACCL_ch4 = 100*C_ch4/(C_ch4 + NC_ch4)
#print("Accuracy of the learning process for $CO_2$ AND $CH_4$ IS ",AC_co2,AC_ch4)
#converting back to original after calculating the relative error
y1_pred = 10**(y1_pred)
y2_pred = 10**(y2_pred)

#'''
# 2. Coefficient of correlation for co2 and ch4
r2_co2 = r2_score(y1_g,y1_pred)
r2_ch4 = r2_score(y2_g,y2_pred)
#'''

#define the limit for uncertainty
lim = 0.5

#Converting X_test to real values pf pressure and temperature for comparing them in future
# 1. Converting Pressure
X_test_co2[:,0] = (X_test_co2[:,0]*p_std) + p_m
X_test_ch4[:,0] = (X_test_ch4[:,0]*p_std) + p_m
X_test_co2[:,0] = 10**(X_test_co2[:,0])
X_test_ch4[:,0] = 10**(X_test_ch4[:,0])
X_test_co2[:,0] = 1e5*(X_test_co2[:,0])
X_test_ch4[:,0] = 1e5*(X_test_ch4[:,0])

X_test_co2[:,1] = (X_test_co2[:,1]/(25/12)) + 0.50
X_test_ch4[:,1] = (X_test_ch4[:,1]/(25/12)) + 0.50
#Finding the maximum GP rel error value and the location

# CO2 adsorption
Max_co2 = np.amax(rel_error_co2)
index_co2 = np.argmax(abs(sigma_co2))

# CO2 adsorption
Max_ch4 = np.amax(rel_error_ch4)
index_ch4 = np.argmax(abs(sigma_ch4))

#### Error calculation block ####

#converting the relative errors to percentages
rel_error_co2 = 100*(rel_error_co2)
rel_error_ch4 = 100*(rel_error_ch4)

#### Error Printing block ####
#finding the mean of GP-predict relative error
rel_m1 = np.mean(rel_error_co2)
rel_m2 = np.mean(rel_error_ch4)

#Declaring error variables
rel_t_co2 = np.zeros(len(X_test_co2))
rel_t_ch4 = np.zeros(len(X_test_ch4))


#Error calculation for CO2
for i in range(len(rel_t_co2)):
	rel_t_co2[i] = 100*abs((y1_pred[i] - y1_g[i])/(y1_g[i]+factor))
		### TEST BLOCK ###
		#print(p_g[i],t_g[i],x1_g[i],rel_t_co2[i],y1_pred[i],y1_g[i])
		### TESTING DONE ###

#Error calculation for CH4		
for i in range(len(rel_t_ch4)):
	rel_t_ch4[i] = 100*abs((y2_pred[i] - (y2_g[i]))/(y2_g[i]+factor))
#Calculating the mean of error
rel_t_m_co2 = np.mean(rel_t_co2)
rel_t_m_ch4 = np.mean(rel_t_ch4)

#Converting fractional to in percentages
Max_co2 = Max_co2*100
Max_ch4 = Max_ch4*100

#### PRINTING MEAN ERRORS ####
#printing mean of GP-predict rel error and true relative error for each iteration in a separate mean.csv file
os.system("echo -n "+str(rel_m1)+","+str(rel_m2)+","+str(Max_co2)+","+str(Max_ch4)+","+str(round(r2_co2,3))+","+str(round(r2_ch4,3))+","+str(round(rel_t_m_co2,3))+","+str(round(rel_t_m_ch4,3))+","+str(round(AC_co2,3))+","+str(round(AC_ch4,3))+","+str(round(ACCL_co2,3))+","+str(round(ACCL_ch4,3))+" >> mean.csv")

#checking the whether the maximum uncertainty is less than out desired limit
#case 1 - CO2
if (Max_co2 >= lim or Max_ch4 >= lim):
	if (Max_co2 > Max_ch4):
		Data = str(X_test_co2[index_co2])
		Data = Data.replace("[","")
		Data = Data.replace("]","")
		#print(X_test_co2[index_co2,0],X_test_co2[index_co2,1])
		print("NOT_DONE")
		print(index_co2)
	else:
		Data = str(X_test_ch4[index_ch4])
		Data = Data.replace("[","")
		Data = Data.replace("]","")
		#print(X_test_ch4[index_ch4,0],X_test_ch4[index_ch4,1])
		print("NOT_DONE")
		print(index_ch4)
	
else:
	print("DONE")
	#### PRINTING LOCAL ERRORS ####
	##printing rel error and true rel. error in a error file
	for i in range(len(rel_error_co2)):
		#rounding off the error to 3 digits after decimals
		rel_error_ch4[i] = round(rel_error_ch4[i],3)
	
		#printing them in the .csv files for error
		os.system("echo -n "+str(rel_error_ch4[i])+" >> rel_ch4.csv")

		#This last condition makes sure that no comma is added at the row end
		if ( i != (len(rel_error_ch4) - 1)): 
			os.system("echo -n "+","+" >> rel_ch4.csv")

		#rounding off the error to 3 digits after decimals
		rel_error_co2[i] = round(rel_error_co2[i],3)
		
		#printing them in the .csv files for error
		os.system("echo -n "+str(rel_error_co2[i])+" >> rel_co2.csv")

		#This last condition makes sure that no comma is added at the row end
		if ( i != (len(rel_error_co2) - 1)): 
			os.system("echo -n "+","+" >> rel_co2.csv")

	#### Printing completed ####

	#### END OF CODE ####


#print(np.shape(X_test),np.shape(y_actual))


### This piece of code block below is simply for testing purposes, i.e. comparing the test set and grouth truth results


#### Converting X_test block ####
'''
# Printing the variables for comparision
for i in range(len(p_g)):
	print('Real Pressure and Temperature, Test Pressure, Temperature, CO2 frac and CH4 fraction, then predicted and actual uptake of CO2 and methane : ',p_g[i],t_g[i],x1_g[i],x2_g[i],X_test_co2[i,1],X_test_ch4[i,1],y1_g[i],y2_g[i],y1_pred[i],y2_pred[i])
'''



#### END OF CODE ####
