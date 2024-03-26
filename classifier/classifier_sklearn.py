
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import TensorDataset
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os




""" wirte a"""


def main():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # script_dir = os.path.dirname(__file__)  # __file__ is the pathname of the file from which the module was loaded

    # Load the synthetic data
    patients = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/patients.csv')
    organs = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/outcomes.csv')
    outcomes = outcomes.dropna()

    outcomes = outcomes.applymap(ast.literal_eval)

    

    #Delete the other outcomes. TODO: all outcomes
    outcomes = outcomes.applymap(lambda x: x['rejection'][2] if x and 'rejection' in x else None)
    outcomes = np.diag(outcomes.values)

    # number of nonzero values in the outcomes
    print(np.count_nonzero(outcomes))


    #Matching:
    merged = pd.concat([patients, organs], axis=1)

    merged = merged.drop('pat_id', axis = 1)
    merged = merged.drop('org_id', axis = 1)


    # Convert categorical variable into dummy/indicator variables
    merged = pd.get_dummies(merged)  


    # Step 2: Load the dataset

    
    X = merged
    y = outcomes

    # Step 3: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # Step 4: Create an instance of the classifier
    clf = LogisticRegression(random_state=0)

    # Step 5: Train the classifier
    clf.fit(X_train, y_train)

    # Step 6: Test the classifier
    predictions = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))



    


    





    





    
if __name__ == '__main__':
    main()
