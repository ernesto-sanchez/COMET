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


from kmeans_clustering import Clustering_kmeans


config_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config'))
sys.path.append(config_path)
from config import config  # noqa: E402




class DataHandler:
    def __init__(self, patients:pd.DataFrame, organs:pd.DataFrame, outcomes:pd.DataFrame, outcomes_noiseless:pd.DataFrame, effects:pd.DataFrame):
        self.patients = patients
        self.organs = organs
        self.outcomes = outcomes
        self.outcomes_noiseless = outcomes_noiseless
        self.merged = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_noiseless = None
        self.y_test_noiseless = None
        self.effects = effects

    def load_data(self) -> dict:


        outcomes = self.outcomes[['pat_id', 'org_id', config['evaluation']['outcome']]]
        outcomes_noiseless = self.outcomes_noiseless[['pat_id', 'org_id', config['evaluation']['outcome']]]

        outcomes = outcomes.dropna()
        outcomes_noiseless = outcomes_noiseless.dropna()

        effects = self.effects[['pat_id', 'org_id', config['evaluation']['outcome']]]
        effects = effects.dropna()

        columns_to_dummy_patients = self.patients.columns.drop(['pat_id', 'age', 'weight'])
        self.patients = pd.get_dummies(self.patients, columns=columns_to_dummy_patients)






        if not config['evaluation']['split']:
            raise ValueError('Not using the The train test split is not supported for the T-Learner')
        



        



        #1) We split the data into train and test sets. Need a to shuffle as the last patients tend to get worse organs!
        #Keep the organ and patients ids bc it gets messy due to the shuffle

        #Hard-coded-flag! -> change the 0.8
        training_patients_ids = self.patients['pat_id'].sample(frac = 0.8, random_state=42)
        training_organs_ids = self.organs['org_id'].sample(frac = 0.8, random_state=42)

        test_patients_ids = self.patients['pat_id'][~self.patients['pat_id'].isin(training_patients_ids)]
        test_organs_ids = self.organs['org_id'][~self.organs['org_id'].isin(training_organs_ids)]

        training_patients = self.patients[self.patients['pat_id'].isin(training_patients_ids)]
        training_organs = self.organs[self.organs['org_id'].isin(training_organs_ids)]
        test_patients = self.patients[self.patients['pat_id'].isin(test_patients_ids)]
        test_organs = self.organs[self.organs['org_id'].isin(test_organs_ids)]


        #2) Do the clustering of the organs
        

        if config['evaluation']['clustering_type'] == 'kmeans':
            #Do the clustering
            self.clustering = Clustering_kmeans(training_organs.drop(columns=['org_id']), config['evaluation']['clustering_n_clusters'])
            self.clustering.fit_and_encode()
        


            #indices_organs = np.stack([self.organs['org_id'], self.clusters], axis = 1) -> not used!
            



        elif config['evaluation']['clustering_type'] == 'expert':


            #Do the clustering
            self.clustering = Clustering_expert(training_organs.drop(columns=['org_id']))
            self.clustering.fit_and_encode()


        elif config['evaluation']['clustering_type'] is None:
            raise ValueError('Clustering must be done for the T-Learner')
            


        #3) Add the organs/clusters to the training/test data. 

        X_train_factual = training_patients
        treatments_train_factual = self.clustering.encode(data = training_organs.drop(columns=['org_id'])) #dont forget to frop the 'org_id column as it shouldnt inlfuence the clustering!
        
        X_test_factual = test_patients
        treatments_test_factual = self.clustering.encode(data = test_organs.drop(columns=['org_id'])) #dont forget to frop the 'org_id column as it shouldnt inlfuence the clustering!


        #get the counterfactual training data. Need to create a dataset 

        indices_train = self.effects.loc[self.effects['pat_id'].isin(training_patients_ids)].iloc[:, 0:2]
        X_train_count = self.patients.reindex(indices_train['pat_id'])

        #now append the organ cluster to the training data also
        orgs = self.organs.set_index('org_id')
        treatments_train_count = self.clustering.encode(orgs.reindex(indices_train['org_id']))


        #get the counterfactual test data. Need to create a dataset
        indices_test = self.effects.loc[self.effects['pat_id'].isin(test_patients_ids)].iloc[:, 0:2]
        X_test_count = self.patients.reindex(indices_test['pat_id'])

        #now append the organ cluster to the training data also
        treatments_test_count = self.clustering.encode(orgs.reindex(indices_test['org_id']))


    
  



        #4) get the train and test labels
        #TODO: How to get the factual labels
        factual_indices_train = pd.DataFrame({'pat_id': training_patients_ids, 'org_id': training_organs_ids})
        y_train_factual = pd.merge(outcomes, factual_indices_train).iloc[:,2]
        y_train_noiseless_factual = pd.merge(outcomes_noiseless, factual_indices_train).iloc[:,2]



        y_train_count = outcomes[outcomes['pat_id'].isin(training_patients_ids)].iloc[:,2]
        
        y_train_noiseless_count = outcomes_noiseless[outcomes_noiseless['pat_id'].isin(training_patients_ids)].iloc[:,2]

        factual_indices_test = pd.DataFrame({'pat_id': test_patients_ids, 'org_id': test_organs_ids})
        y_test_factual = pd.merge(outcomes, factual_indices_test).iloc[:,2]
        y_test_noiseless_factual = pd.merge(outcomes_noiseless, factual_indices_test).iloc[:,2]

        y_test_count = outcomes[outcomes['pat_id'].isin(test_patients_ids)].iloc[:,2]
        y_test_noiseless_count = outcomes_noiseless[outcomes_noiseless['pat_id'].isin(test_patients_ids)].iloc[:,2]

        #5) get the true train and test effects (there is no factual/counterfactual distinction here)
        effects_train = effects[effects['pat_id'].isin(training_patients_ids)].iloc[:,2]
        effects_test = effects[effects['pat_id'].isin(test_patients_ids)].iloc[:,2]



        #6) Make sure there is no 'pat_id' or 'org_id' in the data
        X_train_factual = X_train_factual.drop(columns=['pat_id']).reset_index(drop=True)
        X_train_count = X_train_count.drop(columns=['pat_id']).reset_index(drop=True)
        X_test_factual = X_test_factual.drop(columns=['pat_id']).reset_index(drop=True)
        X_test_count = X_test_count.drop(columns=['pat_id']).reset_index(drop=True)


        #7) Convert treatment to dataframes
        treatments_train_factual = pd.DataFrame(treatments_train_factual)
        treatments_test_factual = pd.DataFrame(treatments_test_factual)
        treatments_train_count = pd.DataFrame(treatments_train_count)
        treatments_test_count = pd.DataFrame(treatments_test_count)
        


        return {
            'X_train_factual': X_train_factual,
            'y_train_factual': y_train_factual,
            'X_test_factual': X_test_factual,
            'y_test_factual': y_test_factual,
            'X_train_count': X_train_count,
            'y_train_count': y_train_count,
            'X_test_count': X_test_count,
            'y_test_count': y_test_count,
            'y_train_noiseless_factual': y_train_noiseless_factual,
            'y_test_noiseless_factual': y_test_noiseless_factual,
            'y_train_noiseless_count': y_train_noiseless_count,
            'y_test_noiseless_count': y_test_noiseless_count, 
            'effects_train': effects_train,
            'effects_test': effects_test,
            'treatments_train_factual': treatments_train_factual, 
            'treatments_test_factual': treatments_test_factual,
            'treatments_train_count': treatments_train_count,
            'treatments_test_count': treatments_test_count
        }



class T_Learner:
    def __init__(self):
        self.split = config['evaluation']['split']  
        self.scale = config['evaluation']['scale']
        self.trainfac = config['evaluation']['trainfac']
        self.evalfac = config['evaluation']['evalfac']
        self.outcome = config['evaluation']['outcome']
        self.effects = pd.read_csv(config['data']['path_effects'])
        self.patients = pd.read_csv(config['data']['path_patients'])
        self.organs = pd.read_csv(config['data']['path_organs'])
        self.outcomes = pd.read_csv(config['data']['path_outcomes'])
        self.outcomes_noiseless = pd.read_csv(config['data']['path_outcomes_noiseless'])
        self.model = config['evaluation']['model']
        self.clustering_type = config['evaluation']['clustering_type']
        self.n = len(self.patients)





        #prepare the data for the T-learner (This data Handler class is not the same as the one of the S-learner)
        self.data_handler = DataHandler(self.patients, self.organs, self.outcomes, self.outcomes_noiseless, self.effects)
        self.processed_data = self.data_handler.load_data()



  



        # Instantiate T learner
        models  = [eval(config['evaluation']['model']) for _ in range(len(np.unique(self.processed_data['treatments_train_factual'])))]
        self.T_learner = TLearner(models = models)


        # Train T_learner
        X_train = self.processed_data['X_train_factual']
        y_train = self.processed_data['y_train_factual']
        t_train = self.processed_data['treatments_train_factual']


        self.T_learner.fit(Y = y_train, T = t_train, X=X_train)


    def get_pehe(self):

        """
        Returns the RMSE of the estmiated effect and the true effects in the counterfactual test data.
        """

        #true effects
        effects = self.processed_data['effects_test']

        #estimated effects

        m = len(self.patients)
        #get the patient features
        features_test_factual = self.processed_data['X_test_factual'].loc[self.processed_data['X_test_factual'].index.repeat(m)].reset_index(drop=True)


        #get the factual treatments
        factual_treatments = self.processed_data['treatments_test_factual'].loc[self.processed_data['treatments_test_factual'].index.repeat(m)].reset_index(drop=True)

        est_effects = self.T_learner.effect(X = features_test_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_test_count'])



        #calculate the PEHE
        pehe = np.sqrt(np.mean((effects - est_effects)**2))

        return pehe
    

    
    def get_pehe_train_factual(self):
            
        """
        Returns 0
        """

        return 0

    def get_pehe_test_factual(self):

        """
        Returns 0
        """

        return 0
    
    def get_pehe_train_count(self):
        """ 
        Return MSE of the predicted vs tru efect on already seen patients

        """
        #true effects
        effects = self.processed_data['effects_train']

        #estimated effects

        m = len(self.patients)
        #get the patient features
        features_train_factual = self.processed_data['X_train_factual'].loc[self.processed_data['X_train_factual'].index.repeat(m)].reset_index(drop=True)


        #get the factual treatments
        factual_treatments = self.processed_data['treatments_train_factual'].loc[self.processed_data['treatments_train_factual'].index.repeat(m)].reset_index(drop=True)

        est_effects = self.T_learner.effect(X = features_train_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_train_count'])



        #calculate the PEHE
        pehe = np.sqrt(np.mean((effects - est_effects)**2))

        return pehe
       

        


            





if __name__ == "__main__":


    tlearner = T_Learner()
    print(tlearner.get_pehe())