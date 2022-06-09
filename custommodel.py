import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests
import json



dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

pickle.dump(regressor, open('linmodel.pkl', 'wb'))


