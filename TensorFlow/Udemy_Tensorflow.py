# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:54:26 2018

@author: Isaac
"""

# Tensorflow Udemy course

import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame as DF

# create dataset
np.random.seed(555)
X1 = np.random.normal(100, 15, 200).astype(int)
X2 = np.random.normal(10, 4.5, 200) # years of experience, 10 years, standard deviation of 4.5
X3 = np.random.normal(32, 4, 200).astype(int)
dob = np.datetime64('2018-10-31') - 365*X3 # feature engineering
b = 5
er = np.random.normal(0, 1.5, 200)

Y = np.array([0.3*x1 + 1.5*x2 + 0.83*x3 + b + e for x1,x2,x3,e in zip(X1,X2,X3,er)])


# Data cleaning
cols = ['iq', 'years_experience', 'dob']
df = DF(list(zip(X1,X2,dob)), columns=cols)
df['income'] = Y
df.info()

df.describe()

# get rid of negative years of experience
df = df[df.years_experience >= 0]
df.describe()



# EDA
df.describe(include=['datetime64'])

# visual exploration
import matplotlib.pyplot as plt
%matplotlib inline
pd.plotting.scatter_matrix(df, figsize=(16,9));

import seaborn as sns
plt.figure(figsize=(12,9))
sns.heatmap(df.corr());

# create the age and then go back to compare
from datetime import datetime as dt
df['age'] = df.dob.apply(lambda x: (dt.strptime('2017-10-31', '%Y-%m-%d') - x).days/365) #subtrack DOB from current date and then divide by 365 to get age
df.drop('dob', axis=1, inplace=True)
df.head()



# Train/Evaluate models
import tensorflow as tf
# train/test split
X = df.iloc[:, [0,1,3]]
Y = df.age

tr_idx = X.sample(frac=0.67).index
Xtr = X[X.index.isin(tr_idx)].values
Xts = X[-X.index.isin(tr_idx)].values

Ytr = Y[Y.index.isin(tr_idx)].values
Yts = Y[-Y.index.isin(tr_idx)].values