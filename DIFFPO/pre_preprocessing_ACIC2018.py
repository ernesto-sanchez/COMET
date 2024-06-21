import os
import pandas as pd
import numpy as np



## This script has 2 parts:
## 1. Create a new big dataset where all the data lies
## 2. Create a the so-called mask datasets with 0s and 1s in apporpriate places.
##  - 


#1. CReate a dataset whiw the following columns: [‘sample_id’, ‘z’, ‘y’, ‘y_0’, ‘y_1’, covariates]

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ACIC2018')
print(data_path)




INDEX_COL_NAME = "sample_id"
COUNTERFACTUAL_FILE_SUFFIX = "_cf"
FILENAME_EXTENSION = ".csv"
DELIMITER = ","


def combine(covariate_file_path, factual_dir_path, counterfactual_dir_path):

    covariates = pd.read_csv(covariate_file_path, index_col=INDEX_COL_NAME, header=0, sep=DELIMITER)
    for file in os.listdir(factual_dir_path):

        factuals = pd.read_csv(os.path.join(factual_dir_path, file), index_col=INDEX_COL_NAME, header=0, sep=DELIMITER)
        cf_file = file.replace('.csv', '_cf.csv')

        counterfactuals = pd.read_csv(os.path.join(counterfactual_dir_path, cf_file), index_col=INDEX_COL_NAME, header=0, sep=DELIMITER)

        # treatments = factuals['z']
        n = factuals.shape[0]
        treatments = pd.DataFrame(np.random.uniform(size=n), columns=['z'], index=factuals.index)
        


        outcomes = factuals['y']

        
        dataset = pd.DataFrame(outcomes).join(covariates, how="inner")
        dataset = counterfactuals.join(dataset, how="inner").drop('Unnamed: 0', axis=1)
        dataset = pd.DataFrame(treatments).join(dataset, how="inner")
        dataset.reset_index(drop=True, inplace=True)

        
        # dataset.fillna(0,inplace=True)

        dataset.to_csv(os.path.join(data_path ,"merged", file.replace(FILENAME_EXTENSION, "_merged" + FILENAME_EXTENSION)),index = False, sep=DELIMITER)

    return 0






#2. Create a mask dataset with the following columns: [‘z’,‘y_0’, ‘y_1’, ‘y’, "sample_id", covariates]


def get_masks(merged_path):
    for file in os.listdir(merged_path):
        merged = pd.read_csv(os.path.join(merged_path, file), sep=DELIMITER)
        merged.iloc[:, 4:] = 1
        treatments = merged['z']
        merged['y0'] = treatments < 0.5
        merged['y1'] = 1 - merged["y0"]
        merged.iloc[:,0] = 1
        merged.iloc[:,3:] = 1

        merged.to_csv(os.path.join(data_path, "masked", file.replace(FILENAME_EXTENSION, "_masked" + FILENAME_EXTENSION)), index = False, sep=DELIMITER)
    return 0


if __name__ == "__main__":
    covariate_path = r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\ACIC2018\x.csv"
    factual_dir = r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\ACIC2018\censoring"
    counterfactual_dir = r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\ACIC2018\censoring_cf"
    merged_dir = r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\ACIC2018\merged"

    # a = combine_covariates_with_observed(covariate_path, factual_dir)
    combine(covariate_path, factual_dir, counterfactual_dir)
    get_masks(merged_dir)