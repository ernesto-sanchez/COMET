import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


class SyntheticDataGenerator:
    def __init__(self, n:int, m:int, complexity: int, only_factual: bool,TAB:float,  bias:bool = False, noise:int = 0 ) -> None:
        self.n = n  # number of patients
        self.m = m  # number of organs
        self.noise = noise
        self.complexity = complexity
        self.only_factual = only_factual
        self.bias = bias
        self.alpha = TAB



    def generate_patients(self) -> pd.DataFrame:
        """
        Generate synthetic data for transplant patients. See comments for details on the data generation process.
        """
        n = self.n
        df = pd.DataFrame(index=range(n), columns=['pat_id', 'age', 'sex', 'blood_type', 'rh', 'weight', 'hla_a', 'hla_b', 'hla_c'])
        df['pat_id'] = range(n)
        df['age'] = np.maximum(0, np.round(np.random.normal(40, 5, size=n)))
        df['sex'] = np.random.choice(["male", "female"], size=n)
        df['blood_type'] = np.random.choice(["A", "B", "AB", "0"], size=n, p=[0.45, 0.09, 0.05, 0.41])
        rh = np.random.choice(["+", "-"], size=n)
        df['rh'] = rh
        df['weight'] = np.maximum(40, np.minimum(120, np.round(np.random.normal(75, 10, size=n), 2)))
        df['hla_a'] = np.random.choice(range(1, 3), size=n)
        df['hla_b'] = np.random.choice(range(1, 3), size=n)
        df['hla_c'] = np.random.choice(range(1, 3), size=n)

        return df



    def generate_organs(self) -> pd.DataFrame:
        """
        Generate synthetic data for transplant organs. See comments for details on the data generation process.

    
        """
        m = self.m
        df = pd.DataFrame(index=range(m), columns=['org_id', 'cold_ischemia_time', 'dsa', 'blood_type_don', 'rh_don', 'age_don', 'sex_don', 'weight_don', 'hla_a_don', 'hla_b_don', 'hla_c_don'])
        df['org_id'] = range(m)
        df['cold_ischemia_time'] = np.maximum(0, np.round(np.random.normal(7, 2, size=m), 2))
        df['dsa'] = np.random.choice([0, 1], size=m)
        df['blood_type_don'] = np.random.choice(["A", "B", "AB", "0"], size=m, p=[0.45, 0.09, 0.05, 0.41])
        df['rh_don'] = np.random.choice(["+", "-"], size=m)
        df['age_don'] = np.maximum(0, np.round(np.random.normal(40, 5, size=m)))
        df['sex_don'] = np.random.choice(["male", "female"], size=m)
        df['weight_don'] = np.maximum(40, np.minimum(120, np.round(np.random.normal(75, 10, size=m), 2)))
        df['hla_a_don'] = np.random.choice(range(1, 3), size=m)
        df['hla_b_don'] = np.random.choice(range(1, 3), size=m)
        df['hla_c_don'] = np.random.choice(range(1, 3), size=m)

        return df
    


    def pat_org_matching(self, patients: pd.DataFrame, organs: pd.DataFrame, alpha: float) -> tuple:

        """
        Function get as input a dataframe with patients and a dataframe with organs. Returns the same data frames but with different ordering of the rows.
        The ith patient corresponds to the ith organ in the new dataframes. The new dataframes are obtained by sampling from the original dataframes.

        When alpha == 0, the best organ is given to each patient, this is measured by euclidena distance between selected features. For other values of alpha, a random organ from the alpha*m nearest organs is smapled.         
        
        """

        n = self.n #patients
        m = self.m #organs


        ## Compute distance matrix
        dist_matrix = np.zeros((n, m))
        #Use broadcasting: take first the age difference, then the weight difference, then the blood type difference, then the HLA difference
        age_pat = patients['age'].values.reshape(-1, 1)
        weight_pat = patients['weight'].values.reshape(-1, 1)
        blood_type_pat = patients['blood_type'].values.reshape(-1, 1)
        rh_pat = patients['rh'].values.reshape(-1, 1)
        hla_a_pat = patients['hla_a'].values.reshape(-1, 1)
        hla_b_pat = patients['hla_b'].values.reshape(-1, 1)
        hla_c_pat = patients['hla_c'].values.reshape(-1, 1)

        age_org = organs['age_don'].values.reshape(1, -1)
        weight_org = organs['weight_don'].values.reshape(1, -1)
        blood_type_org = organs['blood_type_don'].values.reshape(1, -1)
        rh_org = organs['rh_don'].values.reshape(1, -1)
        hla_a_org = organs['hla_a_don'].values.reshape(1, -1)
        hla_b_org = organs['hla_b_don'].values.reshape(1, -1)
        hla_c_org = organs['hla_c_don'].values.reshape(1, -1)

        dist_matrix = np.abs(age_pat - age_org) + np.abs(weight_pat - weight_org) + 10*(blood_type_pat != blood_type_org) + 10*(rh_pat != rh_org) + 5* np.abs(hla_a_pat != hla_a_org) + 2*np.abs(hla_b_pat != hla_b_org) + np.abs(hla_c_pat != hla_c_org)


        ## Create indices of the best organ for each patient

        best_organ = np.zeros(n, dtype=int)

        dist_matrix = pd.DataFrame(dist_matrix)
        #print(dist_matrix)
      
        for patient in range(n):
            m = dist_matrix.shape[1]
            best_organ[patient] = np.random.choice(list(dist_matrix.iloc[patient].nsmallest(int((1-alpha)*(m - 1) +1)).index), size = 1)[0]
            dist_matrix.drop(columns = best_organ[patient], inplace = True)
    


        ## Create new dataframes with the best organ for each patient
        #print(best_organ)

        organs = organs.iloc[best_organ]

        return patients, organs


    def generate_outcomes(self, patients: pd.DataFrame, organs: pd.DataFrame) -> tuple:
        """
        Generate (factual and counterfactual) outcomes for transplant patients. Have 3 complexities for generating the data.

        Complexity 1 consists on a completely linear data generating process, and complexity 2 and 3 are non-linear processes.

        We produce the following outcomes:
        - eGFR: estimated glomerular filtration rate
        - survival: survival after transplantation. A probability is generated and then a binary outcome is produced by sampling from a Bernoulli distribution with that probability.



        """
        noise = self.noise
        if self.only_factual:
            outcomes = pd.DataFrame(index=range(self.n), columns=['pat_id', 'org_id','eGFR', 'survival'] )
            outcomes_noiseless = pd.DataFrame(index=range(self.n), columns=['pat_id', 'org_id', 'eGFR', 'survival'] )

            patients = patients.reset_index(drop=True)
            organs = organs.reset_index(drop=True)

            outcomes['pat_id'] = patients['pat_id']
            outcomes['org_id'] = organs['org_id']

            outcomes_noiseless['pat_id'] = patients['pat_id']
            outcomes_noiseless['org_id'] = organs['org_id']







            if self.complexity == 1:
                outcomes['eGFR'] = 100*np.ones(self.n) - organs['age_don'] - 0.5*patients["age"] - 0.1*organs['weight_don'] - 0.1*patients['weight'] + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n) - organs['age_don'] - 0.5*patients["age"] - 0.1*organs['weight_don'] - 0.1*patients['weight']

                outcomes['survival'] = 1/(1 + np.exp(- 82 +organs['age_don'] + 0.5*patients["age"] + 0.2*organs['weight_don'] + 0.1*patients['weight'] + np.random.normal(0, noise, self.n)))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(- 82 +organs['age_don'] + 0.5*patients["age"] + 0.2*organs['weight_don'] + 0.1*patients['weight']))

                outcomes['survival'] = np.random.binomial(1, outcomes['survival'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival'])

            if self.complexity == 2:
                outcomes['eGFR'] = 100*np.ones(self.n)  - 2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']) + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n)  - 2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don'])

                outcomes['survival'] = 1/(1 + np.exp(-(23  - abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']) + np.random.normal(0, noise, self.n))))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(-(-2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']))))

                outcomes['survival'] = np.random.binomial(1, outcomes['survival'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival'])

            if self.complexity == 3:
                outcomes['eGFR'] = 100*np.ones(self.n) - 15 * (abs(patients['age'] - organs['age_don']) >= 15) - 10 * (abs(patients['weight'] - organs['weight_don']) >=15) + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n) - 15 * (abs(patients['age'] - organs['age_don']) >= 15) - 10 * (abs(patients['weight'] - organs['weight_don']) >=15)
        else:
            patients = patients.reset_index(drop=True)
            organs = organs.reset_index(drop=True)
            outcomes = pd.DataFrame(index=range(self.n * self.m), columns=['pat_id', 'org_id', 'eGFR', 'survival'])
            outcomes_noiseless = pd.DataFrame(index=range(self.n * self.m), columns=['pat_id', 'org_id', 'eGFR', 'survival'])
            outcomes['pat_id'] = np.repeat(np.array(patients['pat_id']), self.m)
            outcomes['org_id'] = np.tile(np.array(organs['org_id']), self.n)
            tiled_indices = np.tile(np.arange(self.n), self.m)
            outcomes_noiseless['pat_id'] = np.repeat(np.array(patients['pat_id']), self.m)
            outcomes_noiseless['org_id'] = np.tile(np.array(organs['org_id']), self.n)

            if self.complexity == 1:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m) - organs['age_don'].values[tiled_indices] - 0.5*patients["age"].values[outcomes['pat_id']] - 0.1*organs['weight_don'].values[tiled_indices] - 0.1*patients['weight'].values[outcomes['pat_id']] + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m) - organs['age_don'].values[tiled_indices] - 0.5*patients["age"].values[outcomes['pat_id']] - 0.1*organs['weight_don'].values[tiled_indices] - 0.1*patients['weight'].values[outcomes['pat_id']]

                outcomes['survival'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'].values[tiled_indices] + 0.05*patients["age"].values[outcomes['pat_id']] + 0.01*organs['weight_don'].values[tiled_indices] + 0.01*patients['weight'].values[outcomes['pat_id']] + np.random.normal(0, noise, self.n * self.m)))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'].values[tiled_indices] + 0.05*patients["age"].values[outcomes['pat_id']] + 0.01*organs['weight_don'].values[tiled_indices] + 0.01*patients['weight'].values[outcomes['pat_id']]))

                outcomes['survival'] = np.random.binomial(1, outcomes['survival'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival'])

            if self.complexity == 2:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m)  - 2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices]) + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m)  - 2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - 0.1*patients['weight'].values[outcomes['pat_id']] - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices])

                outcomes['survival'] = 1/(1 + np.exp(-(-2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices]) + np.random.normal(0, noise, self.n * self.m))))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(-(-2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices]) )))

            if self.complexity == 3:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m) - 15 * (abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) >= 15) - 10 * (abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) >=15) + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m) - 15 * (abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) >= 15) - 10 * (abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) >=15)

            return outcomes, outcomes_noiseless


        return outcomes, outcomes_noiseless






    def generate_datasets(self) -> tuple:
        """
        Generate synthetic datasets for transplant patients, organs and outcomes. See comments for details on the data generation process.
        """
        patients = self.generate_patients()
        organs = self.generate_organs()
        patients, organs = self.pat_org_matching(patients, organs, self.alpha)
        outcomes, outcomes_noiseless = self.generate_outcomes(patients, organs)

        return patients, organs, outcomes, outcomes_noiseless









if __name__ == '__main__':
    generator = SyntheticDataGenerator(n=500, m =500, noise=0, complexity=1, TAB = 1,  only_factual=False)
    df_patients, df_organs, df_outcomes, df_outcomes_noiseless = generator.generate_datasets()




    script_dir = os.path.dirname(__file__)
    df_patients.to_csv(script_dir + "/patients.csv", index=False)
    df_organs.to_csv(script_dir + "/organs.csv", index=False)
    df_outcomes.to_csv(script_dir + "/outcomes.csv", index=False)
    df_outcomes_noiseless.to_csv(script_dir + "/outcomes_noiseless.csv", index=False)



##### TODO: See how many pairs of patients we get, whether eGFR crosses at some point and 