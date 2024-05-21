import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\synthetic_data_generation")
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET")
sys.path.append(r"C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/regressor")
from synthetic_data_faster import SyntheticDataGenerator
from regressor import reg_sklearn
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, tab in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    generator = SyntheticDataGenerator(n=1000, m=1000, noise=0, complexity=2, TAB=tab, only_factual=True) 
    df_patients, df_organs, df_outcomes, df_outcomes_noiseless, _ = generator.generate_datasets()
    
    # Plotting outcomes data
    row = i // 3
    col = i % 3
    axes[row, col].hist(df_outcomes['survival_prob'].values.flatten(), bins=30, color='blue')
    axes[row, col].set_title(f"Outcomes Data (TAB={tab})")
    axes[row, col].set_xlabel("Value")
    axes[row, col].set_ylabel("Frequency")

plt.tight_layout()
plt.show()










# Analysis of outcomes data
print("Outcomes Data:")
print("Mean:")
print(df_outcomes['survival_prob'].mean())
print("Standard Deviation:")
print(df_outcomes['survival_prob'].std())

# Plotting the data
plt.figure(figsize=(10, 6))



# Plotting outcomes data
plt.hist(df_outcomes['survival_prob'].values.flatten(), bins=50, color='red')
plt.title("Outcomes Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# # Additional analysis for categorical variables
# categorical_columns = list(df_patients.columns)  # Replace with the actual column names
# categorical_columns.pop(0)

# fig, axes = plt.subplots(len(categorical_columns), 1, figsize=(6, 4*len(categorical_columns)))

# for i, column in enumerate(categorical_columns):
#     axes[i].bar(df_patients[column].value_counts().index, df_patients[column].value_counts().values)
#     axes[i].set_title(f"{column} Distribution")
#     axes[i].set_xlabel("Category")
#     axes[i].set_ylabel("Count")

# plt.tight_layout()
# plt.show()

