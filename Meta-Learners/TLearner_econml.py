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


from synthetic_data_faster import SyntheticDataGenerator

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor


import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier
import warnings
from expert_clustering import Clustering_expert

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







class T_Learner:
    def __init__(self, patients, organs, outcomes, outcomes_noiseless, effects, models,   split=bool, scale=True, trainfac=bool, evalfac=bool, outcome='eGFR'):
        self.split = split
        self.scale = scale
        self.trainfac = trainfac
        self.evalfac = evalfac
        self.outcome = outcome
        self.effects = effects
        self.n = len(patients)
        self.patients = patients
        self.organs = organs
        self.models = models




        #this doesnt work! Keep in mind WE dont merge the features here. All the treatment features must be reduced to 1 dimension!
        self.data_handler = DataHandler(patients, organs, outcomes, outcomes_noiseless, remote=False)
        self.X_train, self.y_train, self.X_test, self.y_test, self.y_train_noiseless, self.y_test_noiseless = self.data_handler.load_data(trainfac = self.trainfac, evalfac = self.evalfac, outcome=self.outcome, traintest_split=split)
        
        #Fix: trim the unnecessary columns and import organs dataset to cluster afterwards

        #Trim the organs columns
        self.X_train = self.X_train.iloc[:, 0:14]
        self.X_test = self.X_test.iloc[:, 0:14]

        #Take out the pat_id colums s.t. it doesnt affect clustering or training
        self.X_train = self.X_train.drop(columns=['pat_id'])
        self.X_test = self.X_test.drop(columns=['pat_id'])


        #Do the clustering
        
        # self.clustering = Clustering_kmeans(self.organs, 4)
        # self.clusters = self.clustering.fit_and_encode()

        self.clustering = Clustering_expert(self.organs)
        self.clusters = self.clustering.fit_and_encode()



        # Instantiate T learner
        #self.models = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=1)


        self.T_learner = TLearner(models = self.models)


        # Train T_learner
        self.T_learner.fit(self.y_train, T = self.clusters, X=self.X_train)


    def get_pehe(self):

        # Get the base treatments vectors (to compare to the effects)
        indices_factual = [i*len(self.patients) + (i) for i in range(0, len(self.organs))]
        factual_treatments = self.effects['org_id'].iloc[indices_factual] #these are the org_ids of the factual treatments
        factual_treatments_encoded = self.clusters[factual_treatments] #these are the clusters of the factual treatments

        factual_treatments_encoded = factual_treatments_encoded.repeat(self.n)

        #Get the alternative treatments vector

        count_treatments = self.effects['org_id']
        count_treatments = self.clusters[count_treatments]


        #Prepare the features for the T-learner
        patient_features = self.patients.loc[self.patients.index.repeat(self.n)]
        patient_features  = patient_features.drop(columns=['pat_id'])
        patient_features = pd.get_dummies(patient_features)
        


        


        # Estimate treatment effects on test data
        T_te = self.T_learner.effect(patient_features, T0 = factual_treatments_encoded, T1 = count_treatments)






        #calculate the PEHE
        #TODO(1) automate this!!

        ## TODO(1): think how to accomodate classification and regression outcomes --> store also the log of the survival probabbilities in the data-generation
        pehe = np.mean((T_te - self.effects[self.outcome])**2)

        T_te_2 = self.T_learner.effect_inference(patient_features, T0 = factual_treatments_encoded, T1 = count_treatments)



        return pehe


            





if __name__ == "__main__":
    patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
    outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')
    effects = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/effects.csv')


    tlearner = T_Learner(patients, organs, outcomes, outcomes_noiseless, effects, models = LinearRegression(), split=True, scale=True, trainfac=True, evalfac=False, outcome='eGFR')

    print(tlearner.get_pehe())