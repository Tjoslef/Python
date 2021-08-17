#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 22:01:27 2021

@author: kubrt
"""

#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")
print(df.head())
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = prod_per_year['year']
X = X.values.reshape(-1, 1)
y = prod_per_year['totalprod']
plt.scatter(y,X)
plt.show()
regr = linear_model.LinearRegression()
regr.fit(X,y)
y_predict = regr.predict(X)
plt.plot(y_predict,X)
plt.show()
print(regr.coef_)
print(regr.intercept_)
X_future = np.array(range(1, 11))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
plt.plot(X_future,future_predict)
plt.show()

    
