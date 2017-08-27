#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:34:50 2017

@author: ashish
"""

#simple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_data = pd.read_csv('Salary_Data.csv')
X = my_data.iloc[:,:-1].values
Y = my_data.iloc[:,1].values

# using sklearn library to split my_data dataset into test and training
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

#the regressor trains the machine learning model according to the dataset provided
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

#visualizing the test set result
plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_test, y_pred, color = 'red')
plt.title('variation of salary with experience')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X_new = dataset.iloc[:, :-1].values
Y_new = dataset.iloc[:, 4].values

# encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_new[:,3] = labelencoder_X.fit_transform(X_new[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X_new = onehotencoder.fit_transform(X_new).toarray()

#avoiding the dummy variable trap

X_new = X_new[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(X_new, Y_new, test_size = 0.2, random_state = 0)

#fitting the multiple linear regression model

regressor.fit(X_train_new, Y_train_new)

#predicting test set result

Y_pred_new = regressor.predict(X_test_new)

# bakward elimination regression

import statsmodels.formula.api as sm
X_new = np.append(arr = np.ones((50,1)).astype(int), values = X_new , axis = 1)
X_optimal = X_new[:,[0,1,2,3,4,5]]
regressorfromols = sm.OLS(endog = Y_new, exog = X_optimal).fit() # parameters are dependent and independent variable
regressorfromols.summary() #show the summary including p-values
X_optimal = X_optimal[:,[0,2,3,4]]
regressorfromols = sm.OLS(endog = Y_new, exog = X_optimal).fit()
regressorfromols.summary()
X_optimal = X_optimal[:,[0,1,3]]
regressorfromols = sm.OLS(endog = Y_new, exog = X_optimal).fit()
regressorfromols.summary()
X_optimal = X_optimal[:,[0,1]]
regressorfromols = sm.OLS(endog = Y_new, exog = X_optimal).fit()
regressorfromols.summary()