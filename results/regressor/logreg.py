import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import TensorDataset
import ast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn import metrics


import matplotlib.pyplot as plt





"""
Toy script to see if log reg works

"""





patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')



outcomes = outcomes[['pat_id', 'org_id', 'survival']]
outcomes_noiseless = outcomes_noiseless[['pat_id', 'org_id', 'survival']]


patients = pd.get_dummies(patients)
organs = pd.get_dummies(organs)


merged = pd.concat([patients, organs], axis=1)
merged = merged.drop('pat_id', axis = 1)
merged = merged.drop('org_id', axis = 1)

outcomes = outcomes.iloc[:, 2]
outcomes_noiseless = outcomes_noiseless.iloc[:, 2]

X = merged
y = outcomes.values
y_noiseless = outcomes_noiseless.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_2, _, y_train_noiseless,  y_test_noiseless = train_test_split(X, y_noiseless, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


