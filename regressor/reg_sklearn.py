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


import matplotlib.pyplot as plt



class DataHandler:
    def __init__(self, patients:pd.DataFrame, organs:pd.DataFrame, outcomes:pd.DataFrame, outcomes_noiseless:pd.DataFrame, remote:bool = False):
        self.remote = remote
        self.patients = patients
        self.organs = organs
        self.outcomes = outcomes
        self.outcomes_noiseless = outcomes_noiseless
        self.merged = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_test_noiseless = None


    def get_features(self, pat_id, org_id):

        """
        Return the features of a patient and organ in the training set with a specific pat_id and org_id

        """
        patient = self.X_train[pat_id, :]


        return self.X_train, self.y_train, self.X_test, self.y_test, self.y_test_noiseless

    def load_data(self, factual:bool = True, outcome:str = 'eGFR_3', traintest_split:bool = True, scale:bool = True):
        # if self.remote:
        #     patients = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/patients.csv')
        #     organs = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/organs.csv')
        #     outcomes = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/outcomes.csv')
        #     outcomes_noiseless = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/outcomes_noiseless.csv')
        # else: 
        #     patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
        #     organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
        #     outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
        #     outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')

        # self.outcomes = pd.DataFrame(self.outcomes)
        # self.outcomes_noiseless = pd.DataFrame(self.outcomes_noiseless)

        self.outcomes = self.outcomes.dropna()
        self.outcomes_noiseless = self.outcomes_noiseless.dropna()

        # CAUTION: Uncomment this line if loading data from a csv file
        # self.outcomes = self.outcomes.map(ast.literal_eval)
        # self.outcomes_noiseless = self.outcomes_noiseless.map(ast.literal_eval)


        if outcome == 'eGFR_3':
            outcomes = self.outcomes.map(lambda x: x['eGFR'][2] if x and 'eGFR' in x else None)
            outcomes_noiseless = self.outcomes_noiseless.map(lambda x: x['eGFR'][2] if x and 'eGFR' in x else None)

        else:
            raise ValueError('Outcome not supported')
        


        patients = pd.get_dummies(self.patients)
        organs = pd.get_dummies(self.organs) 

        


        if factual:
            outcomes = np.diag(outcomes.values)
            outcomes_noiseless = np.diag(outcomes_noiseless.values)
            merged = pd.concat([patients, organs], axis=1)
            merged = merged.drop('pat_id', axis = 1)
            merged = merged.drop('org_id', axis = 1)
        else:

            """ If not factual, we match the first organ with all the patients
                reminder outcomes[i][j] is the outcome of the i-th patient with the j-th organ
            """
            outcomes_temp = outcomes.values[:, 0]
            outcomes_noiseless_temp = outcomes_noiseless.values[:, 0]

            # outcomes_temp[0] = outcomes.values[0,1]
            # outcomes_noiseless_temp[0] = outcomes_noiseless.values[0,1]

            # Create the second DataFrame
            df2 = pd.concat([organs.iloc[0, :]] * len(patients), axis=1).transpose()

            # Reset the index of 'df2' and drop the old index
            df2 = df2.reset_index(drop=True)

            # Concatenate 'patients' and 'df2'
            merged = pd.concat([patients, df2], axis=1)

            merged = merged.drop('pat_id', axis = 1)
            merged = merged.drop('org_id', axis = 1)

            outcomes = outcomes_temp
            outcomes_noiseless = outcomes_noiseless_temp
        




        X = merged
        y = outcomes
        y_noiseless = outcomes_noiseless

    #     Coumns of Merged:  ['age', 'weight', 'hla_a', 'hla_b', 'hla_c', 'cold_ischemia_time', 'dsa',
    #    'age_don', 'weight_don', 'hla_a_don', 'hla_b_don', 'hla_c_don',
    #    'sex_female', 'sex_male', 'blood_type_0', 'blood_type_A',
    #    'blood_type_AB', 'blood_type_B', 'rh_+', 'rh_-', 'blood_type_don_0',
    #    'blood_type_don_A', 'blood_type_don_AB', 'blood_type_don_B', 'rh_don_+',
    #    'rh_don_-', 'sex_don_female', 'sex_don_male'],
    #  

        if scale:
            X, y = self.scale_data(X, y)
            X, y_noiseless = self.scale_data(X, y_noiseless)

        if traintest_split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_2, _, _,  y_test_noiseless = train_test_split(X, y_noiseless, test_size=0.2, random_state=42)
            return X_train, y_train, X_test, y_test, y_test_noiseless
        else:
            X_train, y_train, X_test, y_test, y_test_noiseless = X, y, X, y, y_noiseless


 
        return X_train, y_train, X_test, y_test, y_test_noiseless
    


    def scale_data(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if y.ndim == 1:
            y_scaled = scaler.fit_transform(y.reshape(-1, 1))
        else:
            y_scaled = scaler.fit_transform(y)
        return X_scaled, y_scaled
    

    

    #TODO: Standarize also outcomes!!!!!


class RegressionModel:
    def __init__(self, patients:pd.DataFrame, organs:pd.DataFrame, outcomes:pd.DataFrame, outcomes_noiseless:pd.DataFrame, remote:bool = False, split:bool = True, scale:bool = True):
        
        self.data_handler = DataHandler(patients, organs, outcomes, outcomes_noiseless, remote=False)
        self.X_train, self.y_train, self.X_test, self.y_test, self.y_test_noiseless = self.data_handler.load_data(factual=True, outcome='eGFR_3', traintest_split=split, scale=scale)

    def train_model(self, model):
        """
        Train a regression model.

        :param model: The regression model to train
        """
        # Train the model
        return model.fit(self.X_train, self.y_train)

    def predict(self, model, X):
        """
        Predict the outcome of a regression model.

        :param model: The regression model to predict
        :param X: The input data
        :return: The predicted outcome
        """
        return model.predict(X)

    def evaluate_model(self, model):
        """
        Evaluate the performance of a regression model.

        :param model: The regression model to evaluate
        :return: Model performance metrics
        """
        # Predictions
        y_pred = model.predict(self.X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        rmse_noiseless = np.sqrt(mean_squared_error(self.y_test_noiseless, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        r2_noiseless = r2_score(self.y_test_noiseless, y_pred)

        return rmse, r2, rmse_noiseless, r2_noiseless

    def run_regression(self):
        # Linear Regression
        linear_model = LinearRegression()
        self.train_model(linear_model)
        mse_lr, r2_lr, mse_noiseless_lr, r2_noiseless_lr = self.evaluate_model(linear_model)

        # Ridge Regression
        ridge_model = Ridge(alpha=1.0)
        self.train_model(ridge_model)
        mse_rr, r2_rr, mse_noiseless_rr, r2_noiseless_rr = self.evaluate_model(ridge_model)

        # Random Forest Regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.train_model(rf_model)
        mse_rf, r2_rf, mse_noiseless_rf, r2_noiseless_rf = self.evaluate_model(rf_model)

        # Create a table
        results = pd.DataFrame({
            'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest Regression'],
            'root MSE': [mse_lr, mse_rr, mse_rf],
            'R2': [r2_lr, r2_rr, r2_rf],
            'root MSE Noiseless': [mse_noiseless_lr, mse_noiseless_rr, mse_noiseless_rf],
            'R2 Noiseless': [r2_noiseless_lr, r2_noiseless_rr, r2_noiseless_rf]
        })

        return results

# if __name__ == '__main__':
#     patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
#     organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
#     outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
#     outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')
#     regression_model = RegressionModel(patients, organs, outcomes, outcomes_noiseless, remote=False, split=True, scale=True)
#     model = LinearRegression()
#     regression_model.train_model(model)
#     print(regression_model.evaluate_model(model))

#     results = regression_model.run_regression()
#     print(results)

