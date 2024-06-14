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



def clustering_patients():
    pass


class DataHandler_PLearner:
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

        # Create a config parser
        self.config = configparser.ConfigParser()

        config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config.ini'))

        # Read the config file
        self.config.read(config_file)

    def load_data(self) -> dict:


        outcomes = self.outcomes[['pat_id', 'org_id', self.config['evaluation']['outcome']]]
        outcomes_noiseless = self.outcomes_noiseless[['pat_id', 'org_id', self.config['evaluation']['outcome']]]

        outcomes = outcomes.dropna()
        outcomes_noiseless = outcomes_noiseless.dropna()

        effects = self.effects[['pat_id', 'org_id', self.config['evaluation']['outcome']]]
        effects = effects.dropna()

        columns_to_dummy_patients = self.patients.columns.drop(['pat_id', 'age', 'weight'])
        self.patients = pd.get_dummies(self.patients, columns=columns_to_dummy_patients)

        columns_to_dummy_organs = self.organs.columns.drop(['org_id', 'cold_ischemia_time', 'age_don', 'weight_don', 'height_don', 'creatinine_don'])

        self.organs = pd.get_dummies(self.organs, columns=columns_to_dummy_organs) 





        if not bool(self.config['evaluation']['split'] == 'True'):
            raise ValueError('Not using the The train test split is not supported for the T-Learner')
        


        #1) We split the data into train and test sets. Need a to shuffle as the last patients tend to get worse organs!
        #Keep the organ and patients ids bc it gets messy due to the shuffle

        training_patients_ids = self.patients['pat_id'].sample(frac = float(self.config['evaluation']['split_proportion']), random_state=42)
        training_organs_ids = self.organs['org_id'].sample(frac = float(self.config['evaluation']['split_proportion']), random_state=42)

        test_patients_ids = self.patients['pat_id'][~self.patients['pat_id'].isin(training_patients_ids)]
        test_organs_ids = self.organs['org_id'][~self.organs['org_id'].isin(training_organs_ids)]

        training_patients = self.patients[self.patients['pat_id'].isin(training_patients_ids)]
        training_organs = self.organs[self.organs['org_id'].isin(training_organs_ids)]
        test_patients = self.patients[self.patients['pat_id'].isin(test_patients_ids)]
        test_organs = self.organs[self.organs['org_id'].isin(test_organs_ids)]


        #2) Do the clustering of the patients
        

        if self.config['evaluation']['clustering_type'] == 'kmeans':
            #Do the clustering
            self.clustering = Clustering_kmeans(training_patients.drop(columns=['pat_id']), int(self.config['evaluation']['clustering_n_clusters']))
            self.clustering.fit_and_encode()
        else: 
            raise ValueError('Clustering type not supported')
        


            

        #3) Add the organs/clusters to the training/test data. 
    


        X_train_factual = training_organs
        patients_train_factual = self.clustering.encode(data = training_patients.drop(columns=['pat_id'])) #dont forget to frop the 'org_id column as it shouldnt inlfuence the clustering!
        
        X_test_factual = test_organs
        patients_test_factual = self.clustering.encode(data = test_patients.drop(columns=['pat_id'])) #dont forget to frop the 'org_id column as it shouldnt inlfuence the clustering!


        #get the counterfactual training data. Need to create a dataset 

        indices_train_count = self.effects.loc[self.effects['org_id'].isin(training_organs_ids) &
                                        self.effects['pat_id'].isin(training_patients_ids)].iloc[:, 0:2]
        
        self.organs.set_index('org_id', inplace=True)
        X_train_count = self.organs.reindex(indices_train_count['org_id'])

        #now append the patients cluster to the training data also
        self.patients.set_index('pat_id', inplace=True)
        patients_train_count = self.clustering.encode(self.patients.reindex(indices_train_count['pat_id']))


        #get the counterfactual test data. Need to create a dataset
        indices_test_count = self.effects.loc[self.effects['org_id'].isin(test_organs_ids) &
                                              self.effects['pat_id'].isin(test_patients_ids)].iloc[:, 0:2]
        
        X_test_count = self.organs.reindex(indices_test_count['org_id'])

        #now append the organ cluster to the training data also
        patients_test_count = self.clustering.encode(self.patients.reindex(indices_test_count['pat_id']))

          

  



        #4) get the train and test labels
        #TODO: How to get the factual labels
        factual_indices_train = pd.DataFrame({'pat_id': training_patients_ids, 'org_id': training_organs_ids})
        y_train_factual = pd.merge(outcomes, factual_indices_train).iloc[:,2]
        y_train_noiseless_factual = pd.merge(outcomes_noiseless, factual_indices_train).iloc[:,2]



        y_train_count = outcomes[outcomes['org_id'].isin(training_organs_ids) &
                                  outcomes['pat_id'].isin(training_patients_ids)].iloc[:,2]
        
        y_train_noiseless_count = outcomes_noiseless[outcomes_noiseless['org_id'].isin(training_organs_ids) &
                                                     outcomes_noiseless['pat_id'].isin(training_patients_ids)].iloc[:,2]

        factual_indices_test = pd.DataFrame({'pat_id': test_patients_ids, 'org_id': test_organs_ids})
        y_test_factual = pd.merge(outcomes, factual_indices_test).iloc[:,2]
        y_test_noiseless_factual = pd.merge(outcomes_noiseless, factual_indices_test).iloc[:,2]

        y_test_count = outcomes[outcomes['pat_id'].isin(test_patients_ids) &
                                outcomes['org_id'].isin(test_organs_ids)].iloc[:,2]
        
        y_test_noiseless_count = outcomes_noiseless[outcomes_noiseless['pat_id'].isin(test_patients_ids) &
                                                    outcomes_noiseless['org_id'].isin(test_organs_ids)].iloc[:,2]

        #5) get the true train and test effects (there is no factual/counterfactual distinction here)
        effects_train = effects[effects['pat_id'].isin(training_patients_ids)&
                                effects['org_id'].isin(training_organs_ids)].iloc[:,2]
        
        effects_test = effects[effects['org_id'].isin(test_organs_ids)&
                               effects['pat_id'].isin(test_patients_ids)].iloc[:,2]



        #6) Make sure there is no 'pat_id' or 'org_id' in the data
        X_train_factual = X_train_factual.drop(columns=['org_id']).reset_index(drop=True)
        #X_train_count = X_train_count.drop(columns=['org_id']).reset_index(drop=True)
        X_test_factual = X_test_factual.drop(columns=['org_id']).reset_index(drop=True)
        #X_test_count = X_test_count.drop(columns=['org_id']).reset_index(drop=True)


        #7) Convert treatment to dataframes
        patients_train_factual = pd.DataFrame(patients_train_factual)
        patients_test_factual = pd.DataFrame(patients_test_factual)
        patients_train_count = pd.DataFrame(patients_train_count)
        patients_test_count = pd.DataFrame(patients_test_count)
        


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
            'patients_train_factual': patients_train_factual, 
            'patients_test_factual': patients_test_factual,
            'patients_train_count': patients_train_count,
            'patients_test_count': patients_test_count, 
            'indices_train': indices_train_count,
            'indices_test': indices_test_count
        }




class PLearner():

    def __init__(self):
        # Create a config parser
        self.config = configparser.ConfigParser()

        config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config.ini'))

        # Read the config file
        self.config.read(config_file)


        self.split = bool(self.config['evaluation']['split'] == 'True')  
        self.scale = self.config['evaluation']['scale']
        self.trainfac = self.config['evaluation']['trainfac']
        self.evalfac = self.config['evaluation']['evalfac']
        self.outcome = self.config['evaluation']['outcome']
        self.effects = pd.read_csv(self.config['data']['path_effects'])
        self.patients = pd.read_csv(self.config['data']['path_patients'])
        self.organs = pd.read_csv(self.config['data']['path_organs'])
        self.outcomes = pd.read_csv(self.config['data']['path_outcomes'])
        self.outcomes_noiseless = pd.read_csv(self.config['data']['path_outcomes_noiseless'])
        self.model = self.config['evaluation']['model']
        self.clustering_type = self.config['evaluation']['clustering_type']
        self.n = len(self.patients)





        #prepare the data for the T-learner (This data Handler class is not the same as the one of the S-learner)
        self.data_handler = DataHandler_PLearner(self.patients, self.organs, self.outcomes, self.outcomes_noiseless, self.effects)
        self.processed_data = self.data_handler.load_data()

        # Instantiate P learner. The ith model corresponds to the ith patient cluster
        self.models  = [eval(self.config['evaluation']['model']) for _ in range(len(np.unique(self.processed_data['patients_train_factual'])))]


        # Train T_learner
        X_train = self.processed_data['X_train_factual']
        y_train = self.processed_data['y_train_factual']
        p_train = self.processed_data['patients_train_factual']


        for i, model in enumerate(self.models):
            model.fit(X_train.loc[(p_train == i).values, :], y_train.loc[(p_train == i).values])


    def get_pehe(self):

        """
        Returns the RMSE of the estimated effect and the true effects in the counterfactual test data.
        """

        #true effects
        effects = self.processed_data['effects_test']

        #estimated effects: outcome of the counterfactual organ - the factual organ
        #we get the features of the counterfactual organs directly from the data handler
        #in order to just perform a substraction from the predicted outcomes of the counterfactual organs, we make a new dataset called factual_features
        #which is the features of the factual organs repeated for each cluster

        m = len(self.processed_data['X_test_factual'])

        factual_patients = self.processed_data['patients_test_factual'].reindex(np.repeat(self.processed_data['patients_test_factual'].index.values, m), method=None)

        factual_features = self.processed_data['X_test_factual'].reindex(np.repeat(self.processed_data['X_test_factual'].index.values, m), method=None)



        patients_test_count = self.processed_data['patients_test_count'].values
        X_test_count = self.processed_data['X_test_count'].values
        factual_patients = factual_patients.values
        factual_features = factual_features.values

        est_effects = [float(self.models[int(patients_test_count[i])].predict([X_test_count[i]]) - self.models[int(factual_patients[i])].predict([factual_features[i]]))
               for i in range(m**2)]

        

        
        #Make this not a loop
    

        m = len(self.organs)




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
        Returns 0
        """

        return 0
    







if __name__ == '__main__':
    plearner = PLearner()
    print(plearner.get_pehe())



