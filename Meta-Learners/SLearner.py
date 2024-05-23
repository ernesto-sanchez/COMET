import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\synthetic_data_generation")
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET")
sys.path.append(r"C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/regressor")
sys.path.append(r"C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/Clustering")





from sklearn.ensemble import  GradientBoostingRegressor


from kmeans_clustering import Clustering_kmeans
from expert_clustering import Clustering_expert

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


        self.patients = pd.get_dummies(self.patients)
        self.organs = pd.get_dummies(self.organs) 

        self.patients = pd.get_dummies(self.patients, columns=['hla_a', 'hla_b', 'hla_c'])
        self.organs = pd.get_dummies(self.organs, columns=['hla_a_don', 'hla_b_don', 'hla_c_don'])


        if not config['evaluation']['split']:
            raise ValueError('Not using the The train test split is not supported for the S-Learner')
        

        #1) Do the clustering of the organs
        

        if config['evaluation']['clustering_type'] == 'kmeans':
            #Do the clustering
            self.clustering = Clustering_kmeans(self.organs, config['evaluation']['clustering_n_clusters'])
            self.clusters = self.clustering.fit_and_encode()


            #indices_organs = np.stack([self.organs['org_id'], self.clusters], axis = 1) -> not used!
            



        elif config['evaluation']['clustering_type'] == 'expert':


            #Do the clustering
            self.clustering = Clustering_expert(self.organs)
            self.clusters = self.clustering.fit_and_encode()


        elif config['evaluation']['clustering_type'] is None:
            pass

        



        #2) We split the data into train and test sets and get the dummies. Need a to shuffle as the last patients tend to get worse organs!
        #Keep the organ and patients ids bc it gets messy due to the shuffle

        #Hard-coded-flag! -> change the 0.8
        training_patients_ids = self.patients['pat_id'].sample(frac = 0.8, random_state=42)
        training_organs_ids = self.organs['org_id'].sample(frac = 0.8, random_state=42)

        test_patients_ids = self.patients['pat_id'][~self.patients['pat_id'].isin(training_patients_ids)]
        test_organs_ids = self.organs['org_id'][~self.organs['org_id'].isin(training_organs_ids)]

        training_patients = self.patients[self.patients['pat_id'].isin(training_patients_ids)]
        test_patients = self.patients[self.patients['pat_id'].isin(test_patients_ids)]


        #3) Add the organs/clusters to the training/test data. 

        if config['evaluation']['clustering_type'] != 'None':
            X_train_factual = training_patients
            X_train_factual['organ_cluster'] = self.clusters[training_patients_ids]
            
            X_test_factual = test_patients
            X_test_factual['organ_cluster'] = self.clusters[test_patients_ids]


            #get the counterfactual training data. Need to create a dataset 

            indices_train = self.effects.loc[self.effects['pat_id'].isin(training_patients_ids)].iloc[:, 0:2]
            X_train_count = self.patients.reindex(indices_train['pat_id'])
            X_train_count['organ_cluster'] = pd.DataFrame(self.clusters).reindex(indices_train['pat_id'])


            #get the counterfactual test data. Need to create a dataset
            indices_test = self.effects.loc[self.effects['pat_id'].isin(test_patients_ids)].iloc[:, 0:2]
            X_test_count = self.patients.reindex(indices_test['pat_id'])
            X_test_count['organ_cluster'] = pd.DataFrame(self.clusters).reindex(indices_test['pat_id'])


    
  
        else: 
            X_train_factual = pd.concat([training_patients, self.organs.loc[training_patients_ids,:]], axis=1)
            X_test_factual = pd.concat([test_patients, self.organs.loc[test_patients_ids, :]], axis=1)



            X_train_count = pd.concat([self.patients.reindex(indices_train['pat_id']), self.organs.reindex(indices_train['org_id'])], axis=1)
            X_test_count = pd.concat([self.patients.reindex(indices_train['pat_id']), self.organs.reindex(indices_train['org_id'])], axis=1)


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
            'y_test_noiseless_count': y_test_noiseless_count
        }



class S_Learner:
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
        self.clustering = None
        self.n = len(patients)





        self.data_handler = DataHandler(patients, organs, outcomes, outcomes_noiseless, effects)
        self.processed_data = self.data_handler.load_data()
        





        






    def get_pehe(self, outcome ):
        
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
        pehe = np.mean((te_slearner - self.effects['eGFR'])**2)

        return pehe










            





if __name__ == "__main__":
    patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
    outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')
    effects = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/effects.csv')


    slearner = S_Learner()

    print(slearner.get_pehe())