import expert_clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap_clustering


import umap.plot
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import kmeans_clustering
import umap_clustering
import expert_clustering










class visualize_clustering():
    def __init__(self, data, outcomes:None):
        """
        data: np.array, shape (n_samples, n_features), Data to be clustered
        labels: np.array, shape (n_samples,), True labels of the data, must be categorical!!
        """
        self.data = data
        self.outcomes = outcomes
        
        # UMAP
        self.map, self.embedding = umap_clustering.Clustering_umap(self.data, 'intersection').fit_and_encode()

    
        

    def visualize_kmeans(self, n_clusters):



        # plt.scatter(embedding[:, 0], embedding[:, 1], s= 5, c=self.labels, cmap='Spectral')
        # plt.show()

        labels = kmeans_clustering.Clustering_kmeans(self.data, n_clusters).fit_and_encode()
        
        umap.plot.points(self.map, labels=labels, cmap="viridis")


    def visualize_expert(self):
        labels = expert_clustering.Clustering_expert(self.data).fit_and_encode()
        umap.plot.points(self.map, labels=labels, cmap="viridis")

    def visualize_egfr(self):
        umap.plot.points(self.map, values=self.outcomes['eGFR'], cmap="viridis")

    def visualize_survival(self):
        umap.plot.points(self.map, values=self.outcomes['survival_prob'], cmap="viridis")







if __name__ == "__main__":

    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')


    visual = visualize_clustering(organs, outcomes)
    visual.visualize_kmeans(n_clusters=10)
    visual.visualize_expert()
    visual.visualize_egfr()
    visual.visualize_survival()



