#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[18]:


#Importing the data
#In order to read the data as a time series, we have to pass special arguments to the read_csv command: 
import os
os.getcwd()
os.chdir('C:\\Users\\40101418\\OneDrive - Anheuser-Busch InBev\\Desktop\\Python\\app')

dataset = pd.read_csv('Sales.csv')
print (dataset.head())
print (dataset.dtypes)


# In[19]:


#Replacing missing values
dataset['rate'].fillna(0, inplace=True)

dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)
dataset.head()


# In[20]:


#Converting integers to words
X = dataset.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]
y


# In[21]:


#Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)


# In[22]:


#Saving the model
pickle.dump(regressor, open('model.pkl','wb'))


# In[23]:


#Reloading the model
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4, 300, 500]]))

