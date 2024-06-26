# imports from package


import pandas as pd
import numpy as np

import os
import sys
import configparser
from sklearn import preprocessing


project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_path, 'Clustering'))



from expert_clustering import Clustering_expert
from kmeans_clustering import Clustering_kmeans


data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ACIC2018')
print(data_path)




INDEX_COL_NAME = "sample_id"
COUNTERFACTUAL_FILE_SUFFIX = "_cf"
FILENAME_EXTENSION = ".csv"
DELIMITER = ","



class DataHandler_DiffPO:
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






        if not bool(self.config['evaluation']['split'] == 'True'):
            raise ValueError('Not using the The train test split is not supported for the T-Learner')
        

        # Dont split the data, as it is already done in the DIFFPO code?
        #TODO: Check if the data is already split in the DIFFPO code




        training_patients_ids = self.patients['pat_id']
        training_organs_ids = self.organs['org_id']



        #2) Do the clustering of the organs
        

        if self.config['evaluation']['clustering_type'] == 'kmeans':
            #Do the clustering
            self.clustering = Clustering_kmeans(self.organs.drop(columns=['org_id']), int(self.config['evaluation']['clustering_n_clusters']))
            self.clustering.fit_and_encode()
        


            #indices_organs = np.stack([self.organs['org_id'], self.clusters], axis = 1) -> not used!
            



        elif self.config['evaluation']['clustering_type'] == 'expert':


            #Do the clustering
            self.clustering = Clustering_expert(self.organs.drop(columns=['org_id']))
            self.clustering.fit_and_encode()


        elif self.config['evaluation']['clustering_type'] == 'None':
            raise ValueError('Clustering must be done for the T-Learner')
            


        #3) Add the organs/clusters to the training/test data. 
        if self.config['evaluation']['clustering_type'] != 'None':

            m = len(self.organs)

            #get the features
            features = self.patients.reindex(np.repeat(self.patients.index.values, m), method=None).reset_index(drop=True)

            #get the treatments
            organs = self.organs.reindex(np.tile(self.organs.index.values, m)).reset_index(drop=True)
            treatments = self.clustering.encode(organs.drop(columns=['org_id']))
            treatments = pd.DataFrame(treatments, columns = ['z'])

            #get the outcomes
            outcome = self.config['evaluation']['outcome']


            #their y1(treated) id our counterfactual
            outcomes_y1 = self.outcomes[outcome]
            outcomes_y1 = outcomes_y1.rename('y1')

            #their y0(untreated) is our factual
            factual_indices = pd.DataFrame({'pat_id': training_patients_ids, 'org_id': training_organs_ids})
            outcomes_factual = pd.merge(outcomes, factual_indices).iloc[:,2]
            outcomes_factual = outcomes_factual.reindex(np.repeat(outcomes_factual.index.values, m), method=None).reset_index(drop=True)




            outcomes_y0 = outcomes_factual
            outcomes_y0 = outcomes_y0.rename('y0')
            outcomes_y = outcomes_y0.copy()
            outcomes_y = outcomes_y.rename('y')



          
        else: 
            raise ValueError('Clustering must be done for DiffPO')
        
        #4) Scale Data

        cov_scalar = preprocessing.StandardScaler()
        y_out_scaler = preprocessing.StandardScaler()
        mu_out_scaler = preprocessing.StandardScaler()


        x = cov_scalar.fit_transform(features)
        print(np.min(x), np.max(x))
        y_out_scaler.fit(np.concatenate([outcomes_y0.to_numpy(), outcomes_y1.to_numpy()]).reshape(-1, 1))
        y_0 = y_out_scaler.transform(outcomes_y0.to_numpy().reshape(-1, 1))
        y_1 = y_out_scaler.transform(outcomes_y1.to_numpy().reshape(-1, 1))


        mu_out_scaler = preprocessing.StandardScaler()

        mu_out_scaler.fit(np.concatenate([outcomes_y0.to_numpy(), outcomes_y1.to_numpy()]).reshape(-1, 1))
        mu_0 = y_out_scaler.transform(outcomes_y0.to_numpy().reshape(-1, 1))
        mu_1 = y_out_scaler.transform(outcomes_y1.to_numpy().reshape(-1, 1))

        self.merged = np.concatenate((treatments,y_0,y_1,mu_0,mu_1, x) , axis=1)

        self.merged = pd.DataFrame(self.merged)
            



        #get the masks

        factual_indices = [i*len(patients) + (i) for i in range(0, int(np.sqrt(len(organs))))]

        self.masked = self.merged.copy()
        self.masked = pd.DataFrame(np.ones(self.masked.shape))
        self.masked.iloc[:,3:5] = 0 #mask mu_0, mu_1
        self.masked.loc[~self.masked.index.isin(factual_indices), 2] = 0
        self.masked.iloc[factual_indices, 1] = 0 #unmask mu_1




        self.merged.to_csv(os.path.join(data_path ,"merged", "synthetic_merged.csv"),index = False, sep=DELIMITER)
        self.masked.to_csv(os.path.join(data_path ,"masked", "synthetic_masked.csv"),index = False, sep=DELIMITER)


        #4) SGet the data necessary for propnet

        #concatenate featres to encoded organs only using the factual organs
        self.propnet_data = pd.concat([pd.Series(self.clustering.encode(self.organs)),
                                    self.patients],
                                    axis=1)
        
        self.propnet_data.to_csv(os.path.join(data_path ,"propnet", "synthetic_propnet.csv"),index = False, sep=DELIMITER)



        



        

    








if __name__ == '__main__':


    # Create a config parser
    config = configparser.ConfigParser()

    config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config1.ini'))

    # Read the config file
    config.read(config_file)


    split = bool(config['evaluation']['split'] == 'True')  
    scale = config['evaluation']['scale']
    trainfac = config['evaluation']['trainfac']
    evalfac = config['evaluation']['evalfac']
    outcome = config['evaluation']['outcome']
    effects = pd.read_csv(config['data']['path_effects'])
    patients = pd.read_csv(config['data']['path_patients'])
    organs = pd.read_csv(config['data']['path_organs'])
    outcomes = pd.read_csv(config['data']['path_outcomes'])
    outcomes_noiseless = pd.read_csv(config['data']['path_outcomes_noiseless'])
    model = config['evaluation']['model']
    clustering_type = config['evaluation']['clustering_type']
    n = len(patients)

    
    data_handler = DataHandler_DiffPO(patients, organs, outcomes, outcomes_noiseless, effects)
    processed_data = data_handler.load_data()