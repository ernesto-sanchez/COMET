

from kmodes.kprototypes import KPrototypes
import pandas as pd




class Clustering_kmeans():
    def __init__(self, data, n_clusters):
        self.data = data  # should be the data of the training organs only!!!
        self.n_clusters = n_clusters
        self.clust_model = None

    def fit_and_encode(self):
        #cluster the organs treatment into n clusters



        # Assuming 'df' is your DataFrame
        # Specify the columns of your categorical features
        selected_columns = ['age_don', 'weight_don', 'blood_type_don', 'rh_don', 'hla_a_don', 'hla_b_don', 'hla_c_don']
        self.data = self.data[selected_columns]
        catColumnsPos = [self.data.columns.get_loc(col) for col in list(self.data.select_dtypes('bool').columns)]
        catColumnsPos += [self.data.columns.get_loc(col) for col in list(self.data.select_dtypes('object').columns)]

        self.clust_model = KPrototypes(n_clusters=self.n_clusters, init='Cao', verbose=0)
        clusters = self.clust_model.fit_predict(self.data, categorical=catColumnsPos)

         # Create a new DataFrame that includes the cluster assignments
        df_clusters = self.data.copy()
        df_clusters['Cluster'] = clusters

        return clusters
    
    def encode(self, data):

        selected_columns = ['age_don', 'weight_don', 'blood_type_don', 'rh_don', 'hla_a_don', 'hla_b_don', 'hla_c_don']
        data = data[selected_columns]

        
        catColumnsPos = [data.columns.get_loc(col) for col in list(data.select_dtypes('bool').columns)]
        catColumnsPos += [data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]


        

        return self.clust_model.predict(data, categorical=catColumnsPos)




    



if __name__ == "__main__":

    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')

    n_clusters = 3
    clustering = Clustering_kmeans(organs, n_clusters)
    print(clustering.fit_and_encode())

