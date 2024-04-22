import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


class SyntheticDataGenerator:
    def __init__(self, n:int, m:int, complexity: int, only_factual: bool, bias:bool = False, noise:int = 0 ) -> None:
        self.n = n  # number of patients
        self.m = m  # number of organs
        self.noise = noise
        self.complexity = complexity
        self.only_factual = only_factual
        self.bias = bias



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



    

    def generate_outcomes(self, patients: pd.DataFrame, organs: pd.DataFrame) -> dict:
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

            outcomes['pat_id'] = range(self.n)
            outcomes['org_id'] = range(self.m)

            outcomes_noiseless['pat_id'] = range(self.n)
            outcomes_noiseless['org_id'] = range(self.m)







            if self.complexity == 1:
                outcomes['eGFR'] = 100*np.ones(self.n) - organs['age_don'] - 0.5*patients["age"] - 0.1*organs['weight_don'] - 0.1*patients['weight'] + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n) - organs['age_don'] - 0.5*patients["age"] - 0.1*organs['weight_don'] - 0.1*patients['weight']

                outcomes['survival'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'] + 0.05*patients["age"] + 0.01*organs['weight_don'] + 0.01*patients['weight'] + np.random.normal(0, noise, self.n)))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'] + 0.05*patients["age"] + 0.01*organs['weight_don'] + 0.01*patients['weight']))

                outcomes['survival'] = np.random.binomial(1, outcomes['survival'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival'])

            if self.complexity == 2:
                outcomes['eGFR'] = 100*np.ones(self.n)  - 2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']) + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n)  - 2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don'])

                outcomes['survival'] = 1/(1 + np.exp(-(-2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']) + np.random.normal(0, noise, self.n))))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(-(-2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']))))

                outcomes['survival'] = np.random.binomial(1, outcomes['survival'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival'])

            if self.complexity == 3:
                outcomes['eGFR'] = 100*np.ones(self.n) - 15 * (abs(patients['age'] - organs['age_don']) >= 15) - 10 * (abs(patients['weight'] - organs['weight_don']) >=15) + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n) - 15 * (abs(patients['age'] - organs['age_don']) >= 15) - 10 * (abs(patients['weight'] - organs['weight_don']) >=15)
        else:
            outcomes = pd.DataFrame(index=range(self.n * self.m), columns=['pat_id', 'org_id', 'eGFR', 'survival'])
            outcomes_noiseless = pd.DataFrame(index=range(self.n * self.m), columns=['pat_id', 'org_id', 'eGFR', 'survival'])
            outcomes['pat_id'] = np.repeat(range(self.n), self.m)
            outcomes['org_id'] = np.tile(range(self.m), self.n)
            outcomes_noiseless['pat_id'] = np.repeat(range(self.n), self.m)
            outcomes_noiseless['org_id'] = np.tile(range(self.m), self.n)

            if self.complexity == 1:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m) - organs['age_don'].values[outcomes['org_id']] - 0.5*patients["age"].values[outcomes['pat_id']] - 0.1*organs['weight_don'].values[outcomes['org_id']] - 0.1*patients['weight'].values[outcomes['pat_id']] + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m) - organs['age_don'].values[outcomes['org_id']] - 0.5*patients["age"].values[outcomes['pat_id']] - 0.1*organs['weight_don'].values[outcomes['org_id']] - 0.1*patients['weight'].values[outcomes['pat_id']]

                outcomes['survival'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'].values[outcomes['org_id']] + 0.05*patients["age"].values[outcomes['pat_id']] + 0.01*organs['weight_don'].values[outcomes['org_id']] + 0.01*patients['weight'].values[outcomes['pat_id']] + np.random.normal(0, noise, self.n * self.m)))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'].values[outcomes['org_id']] + 0.05*patients["age"].values[outcomes['pat_id']] + 0.01*organs['weight_don'].values[outcomes['org_id']] + 0.01*patients['weight'].values[outcomes['pat_id']]))

                outcomes['survival'] = np.random.binomial(1, outcomes['survival'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival'])

            if self.complexity == 2:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m)  - 2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[outcomes['org_id']]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[outcomes['org_id']]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[outcomes['org_id']]) + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m)  - 2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[outcomes['org_id']]) - 0.1*patients['weight'].values[outcomes['pat_id']]

                outcomes['survival'] = 1/(1 + np.exp(-(-2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[outcomes['org_id']]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[outcomes['org_id']]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[outcomes['org_id']]) + np.random.normal(0, noise, self.n * self.m))))
                outcomes_noiseless['survival'] = 1/(1 + np.exp(-(-2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[outcomes['org_id']]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[outcomes['org_id']]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[outcomes['org_id']]) )))

            if self.complexity == 3:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m) - 15 * (abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[outcomes['org_id']]) >= 15) - 10 * (abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[outcomes['org_id']]) >=15) + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m) - 15 * (abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[outcomes['org_id']]) >= 15) - 10 * (abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[outcomes['org_id']]) >=15)

            return outcomes, outcomes_noiseless


        return outcomes, outcomes_noiseless






    def generate_datasets(self) -> tuple:
        """
        Generate synthetic datasets for transplant patients, organs and outcomes. See comments for details on the data generation process.
        """
        patients = self.generate_patients()
        organs = self.generate_organs()
        outcomes, outcomes_noiseless = self.generate_outcomes(patients, organs)

        return patients, organs, outcomes, outcomes_noiseless









if __name__ == '__main__':
    generator = SyntheticDataGenerator(n=2000, m =2000, noise=1, complexity=1, only_factual=True)
    df_patients, df_organs, df_outcomes, df_outcomes_noiseless = generator.generate_datasets()




    script_dir = os.path.dirname(__file__)
    df_patients.to_csv(script_dir + "/patients.csv", index=False)
    df_organs.to_csv(script_dir + "/organs.csv", index=False)
    df_outcomes.to_csv(script_dir + "/outcomes.csv", index=False)
    df_outcomes_noiseless.to_csv(script_dir + "/outcomes_noiseless.csv", index=False)



##### TODO: See how many pairs of patients we get, whether eGFR crosses at some point and 