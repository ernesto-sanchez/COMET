import pandas as pd
import numpy as np
import os
import sys
import configparser
project_path = os.path.dirname(os.path.dirname(__file__))


# Create a config parser
config = configparser.ConfigParser()

config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config1.ini'))

# Read the config file
config.read(config_file)



sys.path.append(os.path.join(project_path, 'Clustering'))
sys.path.append(os.path.join(project_path, 'Meta-Learners'))

from econml.dml import DML
from econml.dr import DRLearner

from sklearn.metrics import roc_auc_score, average_precision_score



from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

from kmeans_clustering import Clustering_kmeans
from expert_clustering import Clustering_expert



from TLearner_econml import DataHandler_TLearner
from SLearner import DataHandler_SLearner








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





class DoubleML:
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
        self.data_handler = DataHandler_TLearner(self.patients, self.organs, self.outcomes, self.outcomes_noiseless, self.effects)
        self.processed_data = self.data_handler.load_data()

        # Instantiate the DML estimator
        model_t = eval(config['evaluation']['model_t'])
        model_y = eval(config['evaluation']['model_y'])
        model_propensity = eval(config['evaluation']['model_propensity'])
        model_regression = eval(config['evaluation']['model_regression'])
        model_final = eval(config['evaluation']['model'])

        discrete_outcome = False if config['evaluation']['outcome_type'] == 'continuous' else True

        if config['evaluation']['learner'] == 'DoubleML()':   
            self.estimator = DML(model_t  = model_t, model_y = model_y, model_final = model_final, discrete_outcome = discrete_outcome, discrete_treatment = True, categories = 'auto')
        elif config['evaluation']['learner'] == 'DRLearner()':
            self.estimator = DRLearner(model_propensity= 'auto', model_regression= model_regression, model_final = model_final, discrete_outcome = discrete_outcome, categories = 'auto')
        else:
            raise ValueError('The learner is not supported')


        # Train T_learner
        X_train = self.processed_data['X_train_factual']
        y_train = self.processed_data['y_train_factual']
        t_train = self.processed_data['treatments_train_factual']


        self.estimator.fit(Y = y_train, T = t_train, X=X_train)

    
    
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

        est_effects = self.estimator.effect(X = features_test_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_test_count'])



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

        est_effects = self.estimator.effect(X = features_train_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_train_count'])



        #calculate the PEHE
        pehe = np.sqrt(np.mean((effects - est_effects)**2))

        return pehe
    
    def get_pairwise_cate_train(self):


        #estimated effects

        m = len(self.patients)
        #get the patient features
        features_train_factual = self.processed_data['X_train_factual'].loc[self.processed_data['X_train_factual'].index.repeat(m)].reset_index(drop=True)


        #get the factual treatments
        factual_treatments = self.processed_data['treatments_train_factual'].loc[self.processed_data['treatments_train_factual'].index.repeat(m)].reset_index(drop=True)

        est_effects = self.estimator.effect(X = features_train_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_train_count'])

        
        return est_effects
    

    def get_pairwise_cate_test(self):



        #estimated effects

        m = len(self.patients)
        #get the patient features
        features_test_factual = self.processed_data['X_test_factual'].loc[self.processed_data['X_test_factual'].index.repeat(m)].reset_index(drop=True)


        #get the factual treatments
        factual_treatments = self.processed_data['treatments_test_factual'].loc[self.processed_data['treatments_test_factual'].index.repeat(m)].reset_index(drop=True)

        est_effects = self.estimator.effect(X = features_test_factual, T0 = factual_treatments, T1 = self.processed_data['treatments_test_count'])
        
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
        
    




if __name__ == '__main__':
    model = DoubleML()
    print(model.get_pehe())