

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import umap






class Clustering_umap():
    def __init__(self, data, combination_method):
        self.data = data
        self.clust_model = None
        self.numerical_data = None
        self.categorical_data = None
        self.combination_method = combination_method
        self.map = None


    def prepare_data(self) -> None:
        """
        Since Umap only supports either categrical or numerical data, we need to preprocess the data dividing the categorical and numerical data

        """
        #drop 'org_id' column
        self.data = self.data.drop(columns = ['org_id'])


        #drop other columns that are not needed
        self.data = self.data.drop(columns = [ 'cold_ischemia_time', 'dsa',  'blood_type_don', 'rh_don','hla_a_don', 'hla_b_don', 'hla_c_don'])

    

        #convert to dummy
        encoded_data = pd.get_dummies(self.data)
        self.numerical_data = encoded_data.select_dtypes(exclude=['bool', 'int64'])
        self.categorical_data = encoded_data.select_dtypes(include=['bool', 'int64'])


        #Scale numerical data

        self.numerical_data_scaled = RobustScaler().fit_transform(self.numerical_data)








    def fit_and_encode(self)-> tuple:
        """
        Fits the clustering model(s) and returns the enconding of the data. As data is mixed, we need to fit 2 models and then combine the 2 models to get the final encoding
        """
        self.prepare_data()

        numeric_mapper = umap.UMAP(n_neighbors=2).fit(self.numerical_data_scaled)
        ordinal_mapper = umap.UMAP(metric="manhattan", n_neighbors=2).fit(self.categorical_data)


        if self.combination_method == 'intersection':
            self.map = numeric_mapper * ordinal_mapper
        elif self.combination_method == 'union':
            self.map = numeric_mapper + ordinal_mapper
        elif self.combination_method == 'contrast':
            self.map = numeric_mapper - ordinal_mapper
        elif self.combination_method == 'intersect_union':
            union_mapper = numeric_mapper + ordinal_mapper
            self.map = umap.UMAP(random_state=42, n_neighbors=2).fit(self.numerical_data) * union_mapper

        return self.map, self.map.embedding_

    
    def encode(self, data):

        raise NotImplementedError('Method not implemented yet')


        return self.clust_model.predict(data, categorical=catColumnsPos)




    



if __name__ == "__main__":

    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')

    clustering = Clustering_umap(organs, combination_method= 'intersection')
    print(clustering.fit_and_encode())

