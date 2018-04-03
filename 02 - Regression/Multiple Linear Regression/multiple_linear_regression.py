# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 02:22:51 2018

@author: sudhe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values                                          #matrix of features(independent variables)
Y = dataset.iloc[:,4].values                                            #dependent vector


#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder           #importing LabelEncoder and OneHotEncoder class from preprocessing lib in scikit
labelencoder_state = LabelEncoder()                                   #Encoding the categorical values with numbers in the first column
X[:, 3] = labelencoder_state.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])               #Further diving the categorical values into 3 groups to eliminate order between the values
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the dummy variable trap
X = X[:, 1:]                                    #removing the first column of X inorder to avoid dummy variable trap


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #X and Y are arrays here, test_size 0.2 means 20%, and random_state is just a random number


# Fitting Multiple linear regression model to training set
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)


# Predicting test set values
Y_pred = linear_regressor.predict(X_test)


# Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0,1,2,3,4,5]]
ols_regressor = sm.OLS(endog = Y, exog = X_opt).fit()   #fitting the ordinarily leaast squares statmodel to create a new object that can be used for backward elimination
ols_regressor.summary()

X_opt = X[:, [0,1,3,4,5]]
ols_regressor = sm.OLS(endog = Y, exog = X_opt).fit()   #fitting the ordinarily leaast squares statmodel to create a new object that can be used for backward elimination
ols_regressor.summary()

X_opt = X[:, [0,3,4,5]]
ols_regressor = sm.OLS(endog = Y, exog = X_opt).fit()   #fitting the ordinarily leaast squares statmodel to create a new object that can be used for backward elimination
ols_regressor.summary()

X_opt = X[:, [0,3,5]]
ols_regressor = sm.OLS(endog = Y, exog = X_opt).fit()   #fitting the ordinarily leaast squares statmodel to create a new object that can be used for backward elimination
ols_regressor.summary()