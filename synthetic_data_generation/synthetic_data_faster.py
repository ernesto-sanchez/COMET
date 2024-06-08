import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import configparser
project_path = os.path.dirname(os.path.dirname(__file__))


# Create a config parser
config = configparser.ConfigParser()

config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config0.ini'))


# Read the config file
config.read(config_file)

class SyntheticDataGenerator:

    """
    Class for generating synthetic data for transplant patients. Patients and organs are generated independently. 
    After generating the data, we match patients with organs based on their features. We then generate outcomes and effects for the matched patients and organs.

    Parameters:
    - n: number of patients
    - m: number of organs
    - complexity: complexity of the data generating process. We have 2 levels of complexity. (level 1: linear, level 2: non-linear)
    - only_factual: if True, we only generate factual outcomes. If False, we generate both factual and counterfactual outcomes.
    - TAB: Parameter controlling Treatment Assignment Bias (0 = random treatment assignment, 1 = Near-optimal tratment assignemnt according to outcome-generating mechanism).
    - noise: standard deviation of the noise added to the outcomes

    """
    def __init__(self) -> None:

        self.n = int(config['synthetic_data']['n'])
        self.m = int(config['synthetic_data']['m'])
        self.noise = float(config['synthetic_data']['noise'])
        self.complexity = int(config['synthetic_data']['complexity'])
        self.only_factual = bool(config['synthetic_data']['only_factual'] == 'True')
        self.alpha = float(config['synthetic_data']['TAB'])
        if self.n <= 0:
            raise ValueError("n must be a positive integer")
        if self.m <= 0:
            raise ValueError("m must be a positive integer")
        if self.n != self.m:
            raise ValueError("n must be equal to m")
        if self.complexity not in [1, 2]:
            raise ValueError("complexity must be either 1 or 2")
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("TAB must be between 0 and 1")
        if self.noise < 0:
            raise ValueError("noise must be a positive float")
        
        # self.n = n  # number of patients
        # self.m = m  # number of organs
        # self.noise = noise
        # self.complexity = complexity
        # self.only_factual = only_factual
        # self.alpha = TAB

        



    def generate_patients(self) -> pd.DataFrame:
        """
        Generate synthetic data for transplant patients. 

        """
        #TODO(3): Take emprical age dist. in switzerland instead of normal.

        n = self.n
        df = pd.DataFrame(index=range(n), columns=['pat_id', 'age', 'sex', 'blood_type', 'rh', 'weight', 'hla_a', 'hla_b', 'hla_c'])

        df['pat_id'] = range(n)
        df['age'] = np.maximum(0, np.round(np.random.normal(40, 5, size=n)))
        df['sex'] = np.random.choice(["male", "female"], size=n)
        df['blood_type'] = np.random.choice(["A", "B", "AB", "0"], size=n, p=[0.45, 0.09, 0.05, 0.41])
        df['rh'] = np.random.choice(["+", "-"], size=n)
        df['weight'] = np.maximum(40, np.minimum(120, np.round(np.random.normal(75, 10, size=n), 2)))
        df['hla_a'] = np.random.choice(range(1, 4), size=n)
        df['hla_b'] = np.random.choice(range(1, 4), size=n)
        df['hla_c'] = np.random.choice(range(1, 4), size=n)

        return df



    def generate_organs(self) -> pd.DataFrame:
        """
        Generate synthetic data for transplant organs. 

        """
        #TODO(3): Take empirical distributions for age and hla features instead of normal. (low prio)

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
        df['hla_a_don'] = np.random.choice(range(1, 4), size=m)
        df['hla_b_don'] = np.random.choice(range(1, 4), size=m)
        df['hla_c_don'] = np.random.choice(range(1, 4), size=m)
        df['height_don'] = np.maximum(140, np.minimum(220, np.round(np.random.normal(170, 10, size=m), 2)))
        df['race_don'] = np.random.choice([0,1], size=m, p=[0.9, 0.1]) # 0 is non-black, 1 is black
        df['hypertension_don'] = np.random.choice([0, 1], size=m, p=[0.8, 0.2])
        df['diabetes_don'] = np.random.choice([0, 1], size=m, p=[0.9, 0.1])
        df['death_cerebrovascular'] = np.random.choice([0, 1], size=m, p=[0.6, 0.4])
        df['creatinine_don'] = np.maximum(0, np.minimum(40, np.round(np.random.normal(0.9, 1, size=m), 2)))
        df['HCV_don'] = np.random.choice([0, 1], size=m, p=[0.99, 0.01])
        df['DCD_don'] = np.random.choice([0, 1], size=m, p=[0.6, 0.4])

        return df
    


    def pat_org_matching(self, patients: pd.DataFrame, organs: pd.DataFrame, alpha: float) -> tuple:

        """

        Perform the patient-organ matching with controlled Treatment Assignment Bias (TAB).

        Parameters:
        - patients: dataframe with patients
        - organs: dataframe with organs
        - alpha: parameter controlling the Treatment Assignment Bias (TAB). alpha = 0 corresponds to no TAB, alpha = 1 corresponds to maximum TAB.

        
        Mathcing Mechanism:
        - When alpha == 0, the best organ is given to each patient, this is measured by euclidean distance between selected features. 
        - For other values of alpha, a random organ from the (alpha*m) best organs is smapled.     
        - It is made sure that no 2 organs are assigned to the same patient.    
        
        Returns the same data frames but with different ordering of the rows. The ith patient corresponds to the ith organ in the new dataframes.

        """

        n = self.n #patients
        m = self.m #organs


        ## Compute distance matrix
        dist_matrix = np.zeros((n, m))

        #Faster: Use broadcasting: Shape organ and aptients features along different dimensions and take their difference. 
        #TODO(1): Maybe incorporate other features??? --> Ask clinician!!!!!!

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



        # Select an organ for the patient by randomly choosing from the smallest distances
        # The number of smallest distances considered is determined by `(1-alpha)*(m - 1) +1`
        # drop the column to make sure the same treatment is not sampled twice.

        #TODO(1): Think of a way to avoid the loop (not obvious) -> before next meeting!!
        for patient in range(n):
            m = dist_matrix.shape[1]
            np.random.seed(1)
            best_organ[patient] = np.random.choice(list(dist_matrix.iloc[patient].nsmallest(int((1-alpha)*(m - 1) +1)).index), size = 1)[0]
            dist_matrix.drop(columns = best_organ[patient], inplace = True)

        organs = organs.iloc[best_organ]
    



        return patients, organs


    def generate_outcomes(self, patients: pd.DataFrame, organs: pd.DataFrame) -> tuple:
        """
        Generate (factual and counterfactual) outcomes for transplant patients. Have 2 complexities for generating the data.

        Complexity 1 consists on a completely linear data generating process, and complexity 2 is a non-linear processes.

        We produce the following outcomes:
        - eGFR -> regression: estimated glomerular filtration rate
        - survival -> classification: survival after transplantation. A probability is generated and then a binary outcome is produced by sampling from a Bernoulli distribution with that probability.
        
        
        #TODO(3): Add more outcomes


        """
        noise = self.noise

        if self.only_factual:

            # Generate outcomes for factual patient-organ pairs --> Use vectorized operations couse loop takes too long

            outcomes = pd.DataFrame(index=range(self.n), columns=['pat_id', 'org_id','eGFR', 'survival_prob','survival_log_prob', 'survival'] )
            outcomes_noiseless = pd.DataFrame(index=range(self.n), columns=['pat_id', 'org_id', 'eGFR','survival_prob', 'survival_log_prob', 'survival'] )

            patients = patients.reset_index(drop=True)
            organs = organs.reset_index(drop=True)

            outcomes['pat_id'] = patients['pat_id']
            outcomes['org_id'] = organs['org_id']

            outcomes_noiseless['pat_id'] = patients['pat_id']
            outcomes_noiseless['org_id'] = organs['org_id']





            if self.complexity == 1:
                outcomes['eGFR'] = 100*np.ones(self.n) - organs['age_don'] - 0.5*patients["age"] - 0.1*organs['weight_don'] - 0.1*patients['weight']  + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n) - organs['age_don'] - 0.5*patients["age"] - 0.1*organs['weight_don'] - 0.1*patients['weight']

                outcomes['survival_prob'] = 1/(1 + np.exp(- 82 +organs['age_don'] + 0.5*patients["age"] + 0.2*organs['weight_don'] + 0.1*patients['weight'] + np.random.normal(0, noise, self.n)))
                outcomes_noiseless['survival_prob'] = 1/(1 + np.exp(- 82 +organs['age_don'] + 0.5*patients["age"] + 0.2*organs['weight_don'] + 0.1*patients['weight']))

                outcomes['survival_log_prob'] = np.log(outcomes['survival_prob'])
                outcomes_noiseless['survival_log_prob'] = np.log(outcomes_noiseless['survival_prob'])

                outcomes['survival'] = np.random.binomial(1, outcomes['survival_prob'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival_prob'])

            if self.complexity == 2:
                outcomes['eGFR'] = 100*np.ones(self.n)  - 2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']) + np.random.normal(0, noise, self.n)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n)  - 2 * abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don'])

                outcomes['survival_prob'] = 1/(1 + np.exp(-0.25*(20  - 2* abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']) -  5* (patients['hla_a'] != organs['hla_a_don']) + np.random.normal(0, noise, self.n))))
                outcomes_noiseless['survival_prob'] = 1/(1 + np.exp(-0.25*(20  - 2* abs(patients['age'] - organs['age_don']) - abs(patients['weight'] - organs['weight_don']) - 10*(patients['blood_type'] != organs['blood_type_don']) -  5* (patients['hla_a'] != organs['hla_a_don']) )))

                outcomes['survival_log_prob'] = np.log(outcomes['survival_prob'])
                outcomes_noiseless['survival_log_prob'] = np.log(outcomes_noiseless['survival_prob'])

                outcomes['survival'] = np.random.binomial(1, outcomes['survival_prob'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival_prob'])

        else:

            # Generate outcomes for each patient-organ pair --> 
            # 1st: Extend the patients and organs dataframes to have n*m rows with tile and repeat
            # 2nd: Generate outcomes for each patient-organ with vectorized oerations --> carefully index the extended dataset




            patients = patients.reset_index(drop=True)
            organs = organs.reset_index(drop=True)

            outcomes = pd.DataFrame(index=range(self.n * self.m), columns=['pat_id', 'org_id', 'eGFR', 'survival_prob', 'survival_log_prob' , 'survival'])
            outcomes_noiseless = pd.DataFrame(index=range(self.n * self.m), columns=['pat_id', 'org_id', 'eGFR', 'survival_prob', 'survival_log_prob' , 'survival'])

            outcomes['pat_id'] = np.repeat(np.array(patients['pat_id']), self.m)
            outcomes['org_id'] = np.tile(np.array(organs['org_id']), self.n)

            tiled_indices = np.tile(np.arange(self.n), self.m)

            outcomes_noiseless['pat_id'] = np.repeat(np.array(patients['pat_id']), self.m)
            outcomes_noiseless['org_id'] = np.tile(np.array(organs['org_id']), self.n)

            if self.complexity == 1:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m) - organs['age_don'].values[tiled_indices] - 0.5*patients["age"].values[outcomes['pat_id']] - 0.1*organs['weight_don'].values[tiled_indices] - 0.1*patients['weight'].values[outcomes['pat_id']] + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m) - organs['age_don'].values[tiled_indices] - 0.5*patients["age"].values[outcomes['pat_id']] - 0.1*organs['weight_don'].values[tiled_indices] - 0.1*patients['weight'].values[outcomes['pat_id']]

                outcomes['survival_prob'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'].values[tiled_indices] + 0.05*patients["age"].values[outcomes['pat_id']] + 0.01*organs['weight_don'].values[tiled_indices] + 0.01*patients['weight'].values[outcomes['pat_id']] + np.random.normal(0, noise, self.n * self.m)))
                outcomes_noiseless['survival_prob'] = 1/(1 + np.exp(- 5 +0.03*organs['age_don'].values[tiled_indices] + 0.05*patients["age"].values[outcomes['pat_id']] + 0.01*organs['weight_don'].values[tiled_indices] + 0.01*patients['weight'].values[outcomes['pat_id']]))

                outcomes['survival_log_prob'] = np.log(outcomes['survival_prob'])
                outcomes_noiseless['survival_log_prob'] = np.log(outcomes_noiseless['survival_prob'])

                outcomes['survival'] = np.random.binomial(1, outcomes['survival_prob'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival_prob'])

            if self.complexity == 2:
                outcomes['eGFR'] = 100*np.ones(self.n * self.m)  - 2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices]) + np.random.normal(0, noise, self.n * self.m)
                outcomes_noiseless['eGFR'] = 100*np.ones(self.n * self.m)  - 2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - 0.1*patients['weight'].values[outcomes['pat_id']] - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices])

                outcomes['survival_prob'] = 1/(1 + np.exp(-0.25*(20 -2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices]) - 5* (patients['hla_a'].values[outcomes['pat_id']] != organs['hla_a_don'].values[tiled_indices]) + np.random.normal(0, noise, self.n * self.m))))
                outcomes_noiseless['survival_prob'] = 1/(1 + np.exp(-0.25* (20 - 2 * abs(patients['age'].values[outcomes['pat_id']] - organs['age_don'].values[tiled_indices]) - abs(patients['weight'].values[outcomes['pat_id']] - organs['weight_don'].values[tiled_indices]) - 10*(patients['blood_type'].values[outcomes['pat_id']] != organs['blood_type_don'].values[tiled_indices])  - 5* (patients['hla_a'].values[outcomes['pat_id']] != organs['hla_a_don'].values[tiled_indices]))))

                outcomes['survival_log_prob'] = np.log(outcomes['survival_prob'])
                outcomes_noiseless['survival_log_prob'] = np.log(outcomes_noiseless['survival_prob'])

                outcomes['survival'] = np.random.binomial(1, outcomes['survival_prob'])
                outcomes_noiseless['survival'] = np.random.binomial(1, outcomes_noiseless['survival_prob'])

            return outcomes, outcomes_noiseless


        return outcomes, outcomes_noiseless
    

    def generate_effects(self, patients: pd.DataFrame, organs: pd.DataFrame, outcomes: pd.DataFrame, outcomes_noiseless: pd.DataFrame) -> pd.DataFrame:
        """
        Generate treatment effects for transplant patients. The treatment effect is the difference between the factual and counterfactual outcomes.

        """
        #extend datasets
        effects = pd.DataFrame(index=range(self.n * self.m), columns=['pat_id', 'org_id', 'eGFR', 'survival_prob', 'survival'])
        patients = patients.reset_index(drop=True)
        organs = organs.reset_index(drop=True)
        effects['pat_id'] = np.repeat(np.array(patients['pat_id']), self.m)
        effects['org_id'] = np.tile(np.array(organs['org_id']), self.n)

        #get factual indices
        factual_indices = [i*len(patients) + (i) for i in range(0, len(organs))]

        factual_outcomes = outcomes.loc[factual_indices]
        factual_outcomes = factual_outcomes.loc[factual_outcomes.index.repeat(self.m)].reset_index(drop=True)


        #hard-coded flag -> careful if more features added!
        #get treatment effect
        effects['eGFR'] = outcomes['eGFR'] - factual_outcomes['eGFR']
        effects['survival_prob'] = outcomes['survival_prob'] - factual_outcomes['survival_prob']
        effects['survival'] = outcomes['survival'] - factual_outcomes['survival']
        
        return effects









    def generate_datasets(self) -> tuple:
        """
        Generate synthetic datasets for transplant patients, organs outcomes and effects.
        """
        patients = self.generate_patients()
        organs = self.generate_organs()
        patients, organs = self.pat_org_matching(patients, organs, self.alpha)
        outcomes, outcomes_noiseless = self.generate_outcomes(patients, organs)
        if not self.only_factual:
            effects = self.generate_effects(patients, organs, outcomes, outcomes_noiseless)
        else:
            effects = None


        return patients, organs, outcomes, outcomes_noiseless, effects









if __name__ == '__main__':
    generator = SyntheticDataGenerator()
    df_patients, df_organs, df_outcomes, df_outcomes_noiseless, df_effects = generator.generate_datasets()




    script_dir = os.path.dirname(__file__)
    try:
        df_patients.to_csv(script_dir + "/patients.csv", index=False)
        df_organs.to_csv(script_dir + "/organs.csv", index=False)
        df_outcomes.to_csv(script_dir + "/outcomes.csv", index=False)
        df_outcomes_noiseless.to_csv(script_dir + "/outcomes_noiseless.csv", index=False)
        df_effects.to_csv(script_dir + "/effects.csv", index=False)
    except Exception as e:
        print(f"An error occurred while writing the data to CSV files: {e}")

