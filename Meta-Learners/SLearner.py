import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
import configparser




project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_path, 'Clustering'))


from kmeans_clustering import Clustering_kmeans
from expert_clustering import Clustering_expert


# Create a config parser
config = configparser.ConfigParser()

config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config.ini'))

# Read the config file
config.read(config_file)



class DataHandler_SLearner:
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
            raise ValueError('Not using the The train test split is not supported for the S-Learner')
        



        



        #1) We split the data into train and test sets. Need a to shuffle as the last patients tend to get worse organs!
        #Keep the organ and patients ids bc it gets messy due to the shuffle

        #Hard-coded-flag! -> change the 0.8
        training_patients_ids = self.patients['pat_id'].sample(frac = float(config['evaluation']['split_proportion']) , random_state=42)
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
            pass


        #3) Add the organs/clusters to the training/test data. 

        if config['evaluation']['clustering_type'] != 'None':
            X_train_factual = training_patients
            X_train_factual['organ_cluster'] = self.clustering.encode(data = training_organs.drop(columns=['org_id'])) #dont forget to frop the 'org_id column as it shouldnt inlfuence the clustering!
            

            X_test_factual = test_patients
            X_test_factual['organ_cluster'] = self.clustering.encode(data = test_organs.drop(columns=['org_id'])) #dont forget to frop the 'org_id column as it shouldnt inlfuence the clustering!


            #get the counterfactual training data. Need to create a dataset 

            indices_train = self.effects.loc[self.effects['pat_id'].isin(training_patients_ids)].iloc[:, 0:2]
            X_train_count = self.patients.reindex(indices_train['pat_id'])

            #now append the organ cluster to the training data also
            orgs = self.organs.set_index('org_id')
            X_train_count['organ_cluster'] = self.clustering.encode(orgs.reindex(indices_train['org_id']))


            #get the counterfactual test data. Need to create a dataset
            indices_test = self.effects.loc[self.effects['pat_id'].isin(test_patients_ids)].iloc[:, 0:2]
            X_test_count = self.patients.reindex(indices_test['pat_id'])

            #now append the organ cluster to the training data also
            X_test_count['organ_cluster'] = self.clustering.encode(orgs.reindex(indices_test['org_id']))


    
  
        else: 
            #Do the organ dummy encoding here! 
            #TODO (include columns that are used in the expert_clustering){}
            columns_to_dummy_organs = self.organs.columns.drop(['org_id', 'cold_ischemia_time', 'age_don', 'weight_don', 'height_don', 'creatinine_don'])

            self.organs = pd.get_dummies(self.organs, columns=columns_to_dummy_organs) 

            X_train_factual = pd.concat([training_patients.reset_index(drop = True), self.organs.loc[training_organs_ids].reset_index(drop = True)], axis=1)
            X_test_factual = pd.concat([test_patients.reset_index(drop = True), self.organs.loc[test_organs_ids].reset_index(drop = True)], axis=1)


            indices_train = self.effects.loc[self.effects['pat_id'].isin(training_patients_ids)].iloc[:, 0:2]



            X_train_count = pd.concat([self.patients.reindex(indices_train['pat_id']).reset_index(drop = True),
                                    self.organs.reindex(indices_train['org_id']).reset_index(drop = True)],
                                    axis=1)

            indices_test = self.effects.loc[self.effects['pat_id'].isin(test_patients_ids)].iloc[:, 0:2]

            X_test_count = pd.concat([self.patients.reindex(indices_test['pat_id']).reset_index(drop = True),
                                      self.organs.reindex(indices_test['org_id']).reset_index(drop = True)],
                                      axis=1)
            
            #drop the org_id column
            X_train_factual = X_train_factual.drop(columns=['org_id'])
            X_test_factual = X_test_factual.drop(columns=['org_id'])
            X_train_count = X_train_count.drop(columns=['org_id'])
            X_test_count = X_test_count.drop(columns=['org_id'])



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
            'indices_train': indices_train,
            'indices_test': indices_test
        }



class S_Learner:
    def __init__(self):
        self.split = bool(config['evaluation']['split']  == 'True')
        self.scale = bool(config['evaluation']['scale'] == 'True')
        self.trainfac = bool(config['evaluation']['trainfac'] == 'True')
        self.evalfac = bool(config['evaluation']['evalfac'] == 'True')
        self.outcome = config['evaluation']['outcome']
        self.effects = pd.read_csv(str(config['data']['path_effects']))
        self.patients = pd.read_csv(config['data']['path_patients'])
        self.organs = pd.read_csv(config['data']['path_organs'])
        self.outcomes = pd.read_csv(config['data']['path_outcomes'])
        self.outcomes_noiseless = pd.read_csv(config['data']['path_outcomes_noiseless'])
        self.model = config['evaluation']['model']
        self.clustering_type = config['evaluation']['clustering_type']
        self.clustering = None
        self.n = len(self.patients)





        self.data_handler = DataHandler_SLearner(self.patients, self.organs, self.outcomes, self.outcomes_noiseless, self.effects)
        self.processed_data = self.data_handler.load_data()

        #fit the model
        self.X_train = self.processed_data['X_train_factual']
        self.y_train = self.processed_data['y_train_factual']

        self.model = eval(config['evaluation']['model'])
        self.model.fit(self.X_train, self.y_train)
        

    def get_pehe(self):
        """
        train model on factual training data and evaluate treatment effect on test counterfactual data
        """




        #Get the estimated treatment effect of the test data: E(y| X (test), factual_treatment(test)) - E(y| X(test), counterfactual_treatment(test))
        m = len(self.organs)
        features_test_factual = self.processed_data['X_test_factual'].loc[self.processed_data['X_test_factual'].index.repeat(m)].reset_index(drop=True)
        features_test_count = self.processed_data['X_test_count']

        est_effects = self.model.predict(features_test_factual) - self.model.predict(features_test_count)

        #get the found truth effects

        true_effects = self.processed_data['effects_test']

        pehe = np.sqrt(np.mean((true_effects - est_effects)**2))

        return pehe
    
    
    def get_pehe_train_count(self):
        """
        Train model on factual training data and evaluate treatment effect on train factual data as well ("MSE treatement effect train")

        """




        #get the estimated treatment effect of the train data: E(y| X(train), factual_treatment(train)) - E(y| X(train), counterfactual_treatment(train))
        m = len(self.organs)
        features_train_factual = self.processed_data['X_train_factual'].loc[self.processed_data['X_train_factual'].index.repeat(m)].reset_index(drop=True)
        features_train_count = self.processed_data['X_train_count']

        est_effects = self.model.predict(features_train_factual) - self.model.predict(features_train_count)

        #get the found truth effects

        true_effects = self.processed_data['effects_train']

        pehe = np.sqrt(np.mean((true_effects - est_effects)**2))

        return pehe
    

    def get_pairwise_cate_train(self):

        #get the estimated treatment effect of the train data: E(y| X(train), factual_treatment(train)) - E(y| X(train), counterfactual_treatment(train))
        m = len(self.organs)
        features_train_factual = self.processed_data['X_train_factual'].loc[self.processed_data['X_train_factual'].index.repeat(m)].reset_index(drop=True)
        features_train_count = self.processed_data['X_train_count']

        est_effects = self.model.predict(features_train_factual) - self.model.predict(features_train_count)

        #Append the correct indexing to make it more useful
        indices_train = self.processed_data['indices_train']

        est_effects = pd.concat([indices_train.reset_index(drop=True), pd.DataFrame(est_effects, columns=['cate'])], axis=1)
        
        return est_effects
    

    def get_pairwise_cate_test(self):

        #get the estimated treatment effect of the test data: E(y| X(test), factual_treatment(test)) - E(y| X(test), counterfactual_treatment(test))
        m = len(self.organs)
        features_test_factual = self.processed_data['X_test_factual'].loc[self.processed_data['X_test_factual'].index.repeat(m)].reset_index(drop=True)
        features_test_count = self.processed_data['X_test_count']

        est_effects = self.model.predict(features_test_factual) - self.model.predict(features_test_count)

        #Append the correct indexing to make it more useful
        indices_test = self.processed_data['indices_test']

        est_effects = pd.concat([indices_test.reset_index(drop=True), pd.DataFrame(est_effects, columns=['cate'])], axis=1)
        
        return est_effects

    def get_pehe_train_factual(self):

        return 0


    def get_pehe_test_factual(self):
        return 0



    def get_outcome_error_train_factual(self):
        """
        Train model on factual training data and evaluate outcome on train factual data as well ("MSE outcome train")

        """


        #get the estimated outcome of the train data
        est_outcome = self.model.predict(self.X_train)

        #get the ground truth effects
        true_outcome = self.processed_data['y_train_noiseless_factual']

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






    def get_outcome_error_test_factual(self):
        """
        Train model on factual training data and evaluate outcome on test factual data as well ("MSE outcome test")

        """


        #get the estimated outcome of the test data
        est_outcome = self.model.predict(self.processed_data['X_test_factual'])
        
        #get the ground truth effects
        true_outcome = self.processed_data['y_test_noiseless_factual']
        
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


    def get_outcome_error_train_count(self):
        """
        Train model on factual training data and evaluate outcome on train counterfactual data as well ("MSE outcome train")

        """



        #get the estimated outcome of the test data
        est_outcome = self.model.predict(self.processed_data['X_train_count'])
        
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
        Train model on factual training data and evaluate outcome on test counterfactual data as well ("MSE outcome test")

        """
        #fit the model

        #get the estimated outcome of the test data
        est_outcome = self.model.predict(self.processed_data['X_test_count'])
        
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


    










            





if __name__ == "__main__":
    # patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
    # organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
    # outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
    # outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')
    # effects = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/effects.csv')


    slearner = S_Learner()
    print(slearner.get_pairwise_cate_train())
    print(slearner.get_pairwise_cate_test())

    print(slearner.get_pehe())
    print(slearner.get_pehe_train_factual())