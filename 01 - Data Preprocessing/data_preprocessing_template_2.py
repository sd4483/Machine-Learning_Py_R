# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:05:34 2018

@author: sudhe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values                                          #matrix of features(independent variables)
Y = dataset.iloc[:,3].values                                            #dependent vector


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #X and Y are arrays here, test_size 0.2 means 20%, and random_state is just a random number


##Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
