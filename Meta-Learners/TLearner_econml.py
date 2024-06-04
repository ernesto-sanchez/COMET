# imports from package
import logging
import statsmodels.api as sm
from econml.metalearners import TLearner
import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import os
import matplotlib.pyplot as plt
import sys
import configparser

project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_path, 'Clustering'))
sys.path.append(os.path.join(project_path, 'Meta-Learners'))


from expert_clustering import Clustering_expert
from kmeans_clustering import Clustering_kmeans
from SLearner import DataHandler_SLearner





# Create a config parser
config = configparser.ConfigParser()

config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config.ini'))

# Read the config file
config.read(config_file)


warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.4f' % x)



logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)





class DataHandler_TLearner:
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






        if not bool(config['evaluation']['split'] == 'True'):
            raise ValueError('Not using the The train test split is not supported for the T-Learner')
        


        #1) We split the data into train and test sets. Need a to shuffle as the last patients tend to get worse organs!
        #Keep the organ and patients ids bc it gets messy due to the shuffle

        #Hard-coded-flag! -> change the 0.8
        training_patients_ids = self.patients['pat_id'].sample(frac = float(config['evaluation']['split_proportion']), random_state=42)
        training_organs_ids = self.organs['org_id'].sample(frac = float(config['evaluation']['split_proportion']), random_state=42)

        test_patients_ids = self.patients['pat_id'][~self.patients['pat_id'].isin(training_patients_ids)]
        test_organs_ids = self.organs['org_id'][~self.organs['org_id'].isin(training_organs_ids)]

        training_patients = self.patients[self.patients['pat_id'].isin(training_patients_ids)]
        training_organs = self.organs[self.organs['org_id'].isin(training_organs_ids)]
        test_patients = self.patients[self.patients['pat_id'].isin(test_patients_ids)]
        test_organs = self.organs[self.organs['org_id'].isin(test_organs_ids)]


        #2) Do the clustering of the organs
        

        if config['evaluation']['clustering_type'] == 'kmeans':
            #Do the clustering
            self.clustering = Clustering_kmeans(training_organs.drop(columns=['org_id']), int(config['evaluation']['clustering_n_clusters']))
            self.clustering.fit_and_encode()
        


            #indices_organs = np.stack([self.organs['org_id'], self.clusters], axis = 1) -> not used!
            



        elif config['evaluation']['clustering_type'] == 'expert':


            #Do the clustering
            self.clustering = Clustering_expert(training_organs.drop(columns=['org_id']))
            self.clustering.fit_and_encode()


        elif config['evaluation']['clustering_type'] == 'None':
            raise ValueError('Clustering must be done for the T-Learner')
            


        #3) Add the organs/clusters to the training/test data. 
        if config['evaluation']['clustering_type'] != 'None':


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

          
        else: 
            #Do the organ dummy encoding here! 
            columns_to_dummy_organs = self.organs.columns.drop(['org_id', 'cold_ischemia_time', 'age_don', 'weight_don', 'height_don', 'creatinine_don'])

            self.organs = pd.get_dummies(self.organs, columns=columns_to_dummy_organs) 

            X_train_factual = training_patients
            treatments_train_factual = self.organs.loc[training_organs_ids].reset_index(drop = True)

            X_test_factual = test_patients.reset_index(drop = True)
            treatments_test_factual = self.organs.loc[test_organs_ids].reset_index(drop = True)


            indices_train = self.effects.loc[self.effects['pat_id'].isin(training_patients_ids)].iloc[:, 0:2]



            X_train_count = self.patients.reindex(indices_train['pat_id']).reset_index(drop = True)
            treatments_train_count = self.organs.reindex(indices_train['org_id']).reset_index(drop = True)


            indices_test = self.effects.loc[self.effects['pat_id'].isin(test_patients_ids)].iloc[:, 0:2]

            X_test_count = self.patients.reindex(indices_test['pat_id']).reset_index(drop = True)
            treatments_test_count = self.organs.reindex(indices_test['org_id']).reset_index(drop = True)


    
  



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
            'treatments_test_count': treatments_test_count, 
            'indices_train': indices_train,
            'indices_test': indices_test
        }


def from_cate_to_outcomes_train(predicted_cates):

    """
    This function is used to convert the estimated CATE to outcomes (in the train set). It is model agnostic, and it assumes linear in treatment models.
    We fit a baseline model b(X,t_0) that estimates factual outcomes conditional on patients and treatment features. 
    Then we add the estimated CATE te(x, t_0, t_1) between the factual and counterfactual treatment to the baseline model to get the counterfactual outcome.

    Inputs:
    - Implicitly (through config): Patients, organs, effects, and outcomes, the S-Learner data-handler

    - predicted cates: A dataframe with the predicted CATEs. The dataframe should have the same indices as the DataHandler's dataframes.

    Outputs: 
    - A dataframe with the estimated counterfactual outcomes. The dataframe should have the same indices as the DataHandler's dataframes.

    CAREFUL WITH: 
        - The indices fo the training and test patients in the S-Learner shoulb be the same as the ones use in the cate prediction
    """

    effects = pd.read_csv(config['data']['path_effects'])
    patients = pd.read_csv(config['data']['path_patients'])
    organs = pd.read_csv(config['data']['path_organs'])
    outcomes = pd.read_csv(config['data']['path_outcomes'])
    outcomes_noiseless = pd.read_csv(config['data']['path_outcomes_noiseless'])

    processed_data = DataHandler_SLearner(patients, organs, outcomes, outcomes_noiseless, effects).load_data()

    #fit the baseline model
    X_train_factual = processed_data['X_train_factual']
    y_train_factual = processed_data['y_train_factual']

    model = eval(config['evaluation']['model_to_outcome'])
    model.fit(X_train_factual, y_train_factual)

    # predict the baseline outcomes
    predicted_outcomes = model.predict(processed_data['X_train_count'])

    #add the estimated cate to the baseline outcomes
    predicted_outcomes = predicted_outcomes + predicted_cates

    return predicted_outcomes










def from_cate_to_outcomes(predicted_cates):

    """
    This function is used to convert the estimated CATE to outcomes. It is model agnostic, and it assumes linear in treatment models.
    We fit a baseline model b(X,t_0) that estimates factual outcomes conditional on patients and treatment features. 
    Then we add the estimated CATE te(x, t_0, t_1) between the factual and counterfactual treatment to the baseline model to get the counterfactual outcome.

    Inputs:
    - Implicitly (through config): Patients, organs, effects, and outcomes, the S-Learner data-handler
    - predicted cates: A dataframe with the predicted CATEs. The dataframe should have the same indices as the DataHandler's dataframes.

    Outputs: 
    - A dataframe with the estimated counterfactual outcomes. The dataframe should have the same indices as the DataHandler's dataframes.
    """

    effects = pd.read_csv(config['data']['path_effects'])
    patients = pd.read_csv(config['data']['path_patients'])
    organs = pd.read_csv(config['data']['path_organs'])
    outcomes = pd.read_csv(config['data']['path_outcomes'])
    outcomes_noiseless = pd.read_csv(config['data']['path_outcomes_noiseless'])

    processed_data = DataHandler_SLearner(patients, organs, outcomes, outcomes_noiseless, effects).load_data()

    #fit the baseline model
    X_train_factual = processed_data['X_train_factual']
    y_train_factual = processed_data['y_train_factual']

    model = eval(config['evaluation']['model_to_outcome'])
    model.fit(X_train_factual, y_train_factual)


    # predict the baseline outcomes
    predicted_outcomes = model.predict(processed_data['X_test_count'])

    #add the estimated cate to the baseline outcomes
    predicted_outcomes = predicted_outcomes + predicted_cates

    return predicted_outcomes









class T_Learner:
    def __init__(self):
        self.split = bool(config['evaluation']['split'] == 'True')  
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
        self.data_handler = DataHandler_TLearner(self.patients, self.organs, self.outcomes, self.outcomes_noiseless, self.effects)
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
    
    def get_pairwise_cate_train(self):

        #true effects
        effects = self.processed_data['effects_train']

        #estimated effects

        m = len(self.patients)
        #get the patient features
        features_train_factual = self.processed_data['X_train_factual'].loc[self.processed_data['X_train_factual'].index.repeat(m)].reset_index(drop=True)


        #get the factual treatments
        factual_treatments = self.processed_data['treatments_train_factual'].loc[self.processed_data['treatments_train_factual'].index.repeat(m)].reset_index(drop=True)

        est_effects = self.T_learner.effect(X = features_train_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_train_count'])

        
        return est_effects
    

    def get_pairwise_cate_test(self):

        #true effects
        effects = self.processed_data['effects_test']

        #estimated effects

        m = len(self.patients)
        #get the patient features
        features_test_factual = self.processed_data['X_test_factual'].loc[self.processed_data['X_test_factual'].index.repeat(m)].reset_index(drop=True)


        #get the factual treatments
        factual_treatments = self.processed_data['treatments_test_factual'].loc[self.processed_data['treatments_test_factual'].index.repeat(m)].reset_index(drop=True)

        est_effects = self.T_learner.effect(X = features_test_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_test_count'])
        
        return est_effects

    def get_outcome_error_train_count(self):

        """
        Calculates the outcome error in the train set after after estimating the cate from the outcome
        """

        #get the estimated outcome of the test data
        est_outcome = from_cate_to_outcomes_train(self.get_pairwise_cate_train())
        
        #get the ground truth effects
        true_outcome = self.processed_data['y_train_noiseless_count']
        
        #return the desired metric
        if config['evaluation']['metric'] == 'RMSE':

            error = np.sqrt(np.mean((true_outcome - est_outcome)**2))

        elif config['evaluation']['metric'] == 'AUROC':

            error = roc_auc_score(true_outcome, est_outcome)

        elif config['evaluation']['metric'] == 'MSE':
                
            error = np.mean((true_outcome - est_outcome)**2)

        elif config['evaluation']['metric'] == 'AUPRC':
                
            error = average_precision_score(true_outcome, est_outcome)

        else:
            raise ValueError('The metric is not supported')
        
        return error
    
    def get_outcome_error_test_count(self):
        """
        Calculates the outcome error in the test set after after estimating the cate from the outcome
        """

        #get the estimated outcome of the test data
        est_outcome = from_cate_to_outcomes(self.get_pairwise_cate_test())
        
        #get the ground truth effects
        true_outcome = self.processed_data['y_test_noiseless_count']
        
        #return the desired metric
        if config['evaluation']['metric'] == 'RMSE':

            error = np.sqrt(np.mean((true_outcome - est_outcome)**2))

        elif config['evaluation']['metric'] == 'AUROC':

            error = roc_auc_score(true_outcome, est_outcome)

        elif config['evaluation']['metric'] == 'MSE':
                
            error = np.mean((true_outcome - est_outcome)**2)

        elif config['evaluation']['metric'] == 'AUPRC':
                
            error = average_precision_score(true_outcome, est_outcome)

        else:
            raise ValueError('The metric is not supported')
        
        return error

    def get_outcome_error_train_factual(self):
        """
        Returns 0: The cate for the factual train set is 0, so it would be essentially ony using the S-Learner
        """
        return 0    
    

    def get_outcome_error_test_factual(self):
        """
        Returns 0: The cate for the factual test set is 0, so it would be essentially ony using the S-Learner
        """
        return 0
        


            





if __name__ == "__main__":


    tlearner = T_Learner()
    # print(tlearner.get_pehe())




    predicted_cates_train = tlearner.get_pairwise_cate_train()
    predicted_cates_test = tlearner.get_pairwise_cate_test()

    print(from_cate_to_outcomes_train(predicted_cates_train))
    print(from_cate_to_outcomes(predicted_cates_test))
