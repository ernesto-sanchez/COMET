from econml.dml import CausalForestDML
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LassoCV
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split


remote = False









# Load the synthetic data
if remote:
    patients = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/patients.csv')
    organs = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/outcomes.csv')
    outcomes_noiseless = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/outcomes_noiseless.csv')
else: 
    patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
    outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')

outcomes = outcomes.dropna()
outcomes_noiseless = outcomes_noiseless.dropna()

outcomes = outcomes.map(ast.literal_eval)
outcomes_noiseless = outcomes_noiseless.map(ast.literal_eval)

# Delete the other outcomes. TODO: all outcomes
outcomes = outcomes.map(lambda x: x['eGFR'][2] if x and 'eGFR' in x else None)
outcomes = np.array(outcomes)
outcomes = np.diag(outcomes)

outcomes_noiseless = outcomes_noiseless.map(lambda x: x['eGFR'][2] if x and 'eGFR' in x else None)
outcomes_noiseless = np.array(outcomes_noiseless)
outcomes_noiseless = np.diag(outcomes_noiseless)

# number of nonzero values in the outcomes
print(np.count_nonzero(outcomes))

patients = pd.get_dummies(patients)
organs = pd.get_dummies(organs)






# Step 3: Split the dataset
Patients_train, Patients_test, organs_train, organs_test, outcomes_train, outcomes_test = train_test_split(patients, organs, outcomes, test_size=0.2, random_state=42)





est = CausalForestDML()
# Or specify hyperparameters
est = CausalForestDML(criterion='het', n_estimators=100,       
                      min_samples_leaf=10, 
                      max_depth=10, max_samples=0.5,
                      discrete_treatment=False,
                      model_t=MultiTaskLassoCV(), model_y= LassoCV())
est.fit(outcomes_train, organs_train, X=Patients_train, W=None, cache_values= True)


est.score(outcomes_test, organs_test, X=Patients_test, W=None, sample_weight=None)
# Confidence intervals via Bootstrap-of-Little-Bags for forests
lb, ub = est.effect_interval(Patients_test, alpha=0.05)