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
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn import metrics
import sys
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\synthetic_data_generation")
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET")
sys.path.append(r"C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/regressor")
sys.path.append(r"C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/Clustering")

from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from econml.inference import BootstrapInference


from synthetic_data_fast import SyntheticDataGenerator

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor


import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier
import warnings

# # from causalml.inference.meta import XGBTLearner, MLPTLearner
# from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
# from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier
# from causalml.inference.meta import LRSRegressor
# from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
# from causalml.propensity import ElasticNetPropensityModel
# from causalml.dataset import *
# from causalml.metrics import *

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# imports from package
import logging
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import statsmodels.api as sm
from copy import deepcopy

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

from reg_sklearn import DataHandler

from kmeans import Clustering_kmeans









class S_Learner:
    def __init__(self, patients, organs, outcomes, outcomes_noiseless, effects,  split=bool, scale=True, trainfac=bool, evalfac=bool, outcome='eGFR'):
        self.split = split
        self.scale = scale
        self.trainfac = trainfac
        self.evalfac = evalfac
        self.outcome = outcome
        self.effects = effects
        self.n = len(patients)
        self.patients = patients
        self.organs = organs
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=1)




        #this doesnt work! Keep in mind WE dont merge the features here. All the treatment features must be reduced to 1 dimension!
        self.data_handler = DataHandler(patients, organs, outcomes, outcomes_noiseless, remote=False)
        self.X_train, self.y_train, self.X_test, self.y_test, self.y_train_noiseless, self.y_test_noiseless = self.data_handler.load_data(trainfac = self.trainfac, evalfac = self.evalfac, outcome=self.outcome, traintest_split=split)
        



        #Take out the pat_id colums s.t. it doesnt affect clustering or training
        self.X_train = self.X_train.drop(columns=['pat_id', 'org_id'])
        self.X_test = self.X_test.drop(columns=['pat_id', 'org_id'])


        # #Do the clustering
        # self.clustering = Clustering_kmeans(self.organs, 4)
        # self.clusters = self.clustering.fit_and_encode()

        #Fit the model
        self.model.fit(self.X_train, self.y_train)


        #Get the treatment effect
        #First get factual features
        self.organs = pd.get_dummies(self.organs)
        factual_organs = self.organs.loc[self.organs.index.repeat(self.n)]
        factual_patients = self.X_test.iloc[:, 0:13]

        factual_organs = factual_organs.reset_index(drop=True)
        factual_patients = factual_patients.reset_index(drop=True)

        factual = pd.concat([factual_patients, factual_organs], axis=1)

        # factual.drop(columns=['org_id'], inplace=True)
        


        #get counterfactual features
        # self.X_test = self.X_test.drop(columns=['org_id'])
        caounterfactual = self.X_test


        






    def get_pehe(self):
        
        #Get the treatment effect
        #First get factual features
        self.organs = pd.get_dummies(self.organs)
        factual_organs = self.organs.loc[self.organs.index.repeat(self.n)]
        factual_patients = self.X_test.iloc[:, 0:12]

        factual_organs = factual_organs.reset_index(drop=True)
        factual_patients = factual_patients.reset_index(drop=True)

        factual = pd.concat([factual_patients, factual_organs], axis=1)

        factual.drop(columns=['org_id'], inplace=True)
        


        #get counterfactual features
        #self.X_test = self.X_test.drop(columns=['org_id'])
        caounterfactual = self.X_test

        #Get the treatment effects
        te_slearner = self.model.predict(factual) - self.model.predict(caounterfactual)
        pehe = np.sqrt(np.mean((te_slearner - self.effects['eGFR'])**2))

        return pehe










            





if __name__ == "__main__":
    patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
    outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')
    effects = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/effects.csv')


    tlearner = S_Learner(patients, organs, outcomes, outcomes_noiseless, effects,split=True, scale=True, trainfac=True, evalfac=False, outcome='eGFR')

    print(tlearner.get_pehe())