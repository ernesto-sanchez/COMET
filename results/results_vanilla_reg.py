import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\synthetic_data_generation")
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET")
sys.path.append(r"C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/regressor")
from synthetic_data import SyntheticDataGenerator
from regressor import reg_sklearn
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

my_path = Path(__file__).resolve().parent



"""
First we generate data for 100 patients and 100 organs and different noise levels for the outcomes.
We compare the accuracy of the three regression methods for different noise levels. 
"""
class ComparisonVaryingNoise:
    def __init__(self, noise_levels, n, m, n_simulations, complexity = 1):
        self.noise_levels = noise_levels
        self.n = n
        self.m = m
        self.n_simulations = n_simulations
        self.results = []
        self.complexity = complexity

    def run_comparison(self):
        for _ in range(self.n_simulations):
            simulation_results = {}
            for i in range(len(self.noise_levels)):
                generator = SyntheticDataGenerator(self.n, self.m, self.noise_levels[i], complexity=self.complexity)
                df_patients, df_organs, df_outcomes, df_outcomes_noiseless = generator.generate_datasets()
                simulation_results[i] = reg_sklearn.RegressionModel(df_patients, df_organs, df_outcomes, df_outcomes_noiseless, remote=False).run_regression()
                print(f"Done with noise level {self.noise_levels[i]}")
            self.results.append(simulation_results)

    def plot_results(self):
        linear_mse = [np.mean([self.results[j][i]['root MSE'][0] for j in range(self.n_simulations)]) for i in range(len(self.noise_levels))]
        ridge_mse = [np.mean([self.results[j][i]['root MSE'][1] for j in range(self.n_simulations)]) for i in range(len(self.noise_levels))]
        random_forests_mse = [np.mean([self.results[j][i]['root MSE'][2] for j in range(self.n_simulations)]) for i in range(len(self.noise_levels))]

        linear_noiseless_mse = [np.mean([self.results[j][i]['root MSE Noiseless'][0] for j in range(self.n_simulations)]) for i in range(len(self.noise_levels))]
        ridge_noiseless_mse = [np.mean([self.results[j][i]['root MSE Noiseless'][1] for j in range(self.n_simulations)]) for i in range(len(self.noise_levels))]
        random_forests_noiseless_mse = [np.mean([self.results[j][i]['root MSE Noiseless'][2] for j in range(self.n_simulations)]) for i in range(len(self.noise_levels))]

        # Plot the mean MSE values
        plt.plot(self.noise_levels, linear_mse, label='Mean Linear Regression')
        plt.plot(self.noise_levels, ridge_mse, label='Mean Ridge Regression')
        plt.plot(self.noise_levels, random_forests_mse, label='Mean Random Forests')
        plt.xlabel('Noise Level')
        plt.ylabel('Mean root MSE')
        plt.title('Mean root MSE for Different Noise Levels and Models, n={}, m={}'.format(self.n, self.m))
        plt.legend()
        plt.savefig(os.path.join(my_path , '{}.png'.format(plt.gca().get_title())))
        plt.show()

        # Plot the mean noiseless MSE values
        plt.plot(self.noise_levels, linear_noiseless_mse, label='Mean Linear Regression Noiseless')
        plt.plot(self.noise_levels, ridge_noiseless_mse, label='Mean Ridge Regression Noiseless')
        plt.plot(self.noise_levels, random_forests_noiseless_mse, label='Mean Random Forests Noiseless')
        plt.xlabel('Noise Level')
        plt.ylabel('Mean Noiseless  root MSE')
        plt.title('Mean Noiseless  root MSE for Different Noise Levels, n={}, m={}'.format(self.n, self.m))
        plt.legend()
        plt.savefig(os.path.join(my_path , '{}.png'.format(plt.gca().get_title())))
        plt.show()

        # Combine the two plots into one
        plt.figure(figsize=(10, 5))

        # Plot the mean MSE values
        plt.subplot(1, 2, 1)
        plt.plot(self.noise_levels, linear_mse, label='Mean Linear Regression')
        plt.plot(self.noise_levels, ridge_mse, label='Mean Ridge Regression')
        plt.plot(self.noise_levels, random_forests_mse, label='Mean Random Forests')
        plt.xlabel('Noise Level')
        plt.ylabel('Mean MSE')
        plt.title('Mean MSE for Different Noise Levels and Models, n={}, m={}'.format(self.n, self.m))
        plt.legend()

        # Plot the mean noiseless MSE values
        plt.subplot(1, 2, 2)
        plt.plot(self.noise_levels, linear_noiseless_mse, label='Mean Linear Regression Noiseless')
        plt.plot(self.noise_levels, ridge_noiseless_mse, label='Mean Ridge Regression Noiseless')
        plt.plot(self.noise_levels, random_forests_noiseless_mse, label='Mean Random Forests Noiseless')
        plt.xlabel('Noise Level')
        plt.ylabel('Mean Noiseless MSE')
        plt.title('Mean Noiseless MSE for Different Noise Levels, n={}, m={}'.format(self.n, self.m))
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(my_path , 'combined_plots.png'))
        plt.show()

# comparison = ComparisonVaryingNoise(noise_levels=[0, 0.5, 1, 3, 5, 10, 20, 50], n=100, m=100, n_simulations=4, complexity=1)
# comparison.run_comparison()
# comparison.plot_results()



class ComparisonVaryingObservations:
    def __init__(self, n_values, m_values, noise, n_simulations, complexity = 2):
        self.n_values = n_values
        self.m_values = m_values
        self.noise = noise
        self.n_simulations = n_simulations
        self.results = []
        self.complexity = complexity

    def run_comparison(self):
        for _ in range(self.n_simulations):
            simulation_results = {}
            for i in range(len(self.n_values)):
                generator = SyntheticDataGenerator(self.n_values[i], self.m_values[i], self.noise, complexity=self.complexity)
                df_patients, df_organs, df_outcomes, df_outcomes_noiseless = generator.generate_datasets()
                simulation_results[i] = reg_sklearn.RegressionModel(df_patients, df_organs, df_outcomes, df_outcomes_noiseless, remote=False, scale= True).run_regression()
                print(f"Done with {self.n_values[i]} observations and {self.m_values[i]} organs.")
            self.results.append(simulation_results)

    def plot_results(self):
        linear_mse = [np.mean([self.results[j][i]['root MSE'][0] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        ridge_mse = [np.mean([self.results[j][i]['root MSE'][1] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        random_forests_mse = [np.mean([self.results[j][i]['root MSE'][2] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        svm_mse = [np.mean([self.results[j][i]['root MSE'][3] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]

        # linear_noiseless_mse = [np.mean([self.results[j][i]['root MSE Noiseless'][0] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        # ridge_noiseless_mse = [np.mean([self.results[j][i]['root MSE Noiseless'][1] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        # random_forests_noiseless_mse = [np.mean([self.results[j][i]['root MSE Noiseless'][2] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]

        linear_mse_train = [np.mean([self.results[j][i]['root MSE Train'][0] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        ridge_mse_train = [np.mean([self.results[j][i]['root MSE Train'][1] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        random_forests_mse_train = [np.mean([self.results[j][i]['root MSE Train'][2] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]
        svm_mse_train = [np.mean([self.results[j][i]['root MSE Train'][3] for j in range(self.n_simulations)]) for i in range(len(self.n_values))]



        # Combine the two plots into one
        plt.figure(figsize=(10, 5))

        # Plot the mean MSE values
        plt.subplot(1, 2, 1)
        plt.plot(self.n_values, linear_mse, label='Mean Linear Regression')
        plt.plot(self.n_values, ridge_mse, label='Mean Ridge Regression')
        plt.plot(self.n_values, random_forests_mse, label='Mean Random Forests')
        plt.plot(self.n_values, svm_mse, label='Mean SVM')
        plt.xlabel('Number of Observations')
        plt.ylabel('Mean MSE')
        plt.title('Mean MSE for Different Number of Observations, Noise {}, and Models'.format(self.noise))
        plt.legend()

        # # Plot the mean noiseless MSE values
        # plt.subplot(1, 2, 2)
        # plt.plot(self.n_values, linear_noiseless_mse, label='Mean Linear Regression Noiseless')
        # plt.plot(self.n_values, ridge_noiseless_mse, label='Mean Ridge Regression Noiseless')
        # plt.plot(self.n_values, random_forests_noiseless_mse, label='Mean Random Forests Noiseless')
        # plt.xlabel('Number of Observations')
        # plt.ylabel('Mean Noiseless MSE')
        # plt.title('Mean Noiseless MSE for Different Number of Observations, Noise {}, and Models'.format(self.noise))
        # plt.legend()


        # PLot the mean MSE values for the training set
        plt.subplot(1, 2, 2)
        plt.plot(self.n_values, linear_mse_train, label='Mean Linear Regression Train')
        plt.plot(self.n_values, ridge_mse_train, label='Mean Ridge Regression Train')
        plt.plot(self.n_values, random_forests_mse_train, label='Mean Random Forests Train')
        plt.plot(self.n_values, svm_mse_train, label='Mean SVM Train')
        plt.xlabel('Number of Observations')
        plt.ylabel('Mean MSE Train')
        plt.title('Mean MSE Train for Different Number of Observations, Noise {}, and Models'.format(self.noise))
        plt.legend()


        plt.tight_layout()
        plt.savefig(os.path.join(my_path, 'combined_plots.png'))
        plt.show()


##TODO :look into but of MSE in Test set(canno see ridge and rancom forest in the cahrt)

comparison = ComparisonVaryingObservations(n_values=[30, 100, 150, 200, 300, 500], m_values=[30, 100, 150, 200, 300, 500], noise=5, n_simulations=1, complexity=2)
comparison.run_comparison()
comparison.plot_results()
        

#TODO: Fix hard-coded scale parameter

class Comparison_Counterfactual:
    def __init__(self, n_values, m_values, noise, n_simulations, complexity = 1, scale = True):
        self.n_values = n_values
        self.m_values = m_values
        self.noise = noise
        self.n_simulations = n_simulations
        self.results = []
        self.complexity = complexity

    '''
    The idea is to run a regression trying to estimate the EGFR levels for patients given training data
    and the use that traied model to estimate the EGFR level for this same patients but for dfferent organs.
    '''


    def get_counterfactual_features(self, patient_id, organ_id, df_patients, df_organs):
        patient = df_patients.iloc[patient_id,]
        organ = df_organs.iloc[organ_id,]

        patient_new = pd.concat([df_patients.iloc[patient_id,], df_organs.iloc[organ_id,]], axis = 0)
        patient_new = patient_new.drop(['pat_id', 'org_id'])
        patient_new = pd.DataFrame(patient_new).transpose()
        # patient_new = pd.get_dummies(patient_new, columns=['age', 'weight', 'hla_a', 'hla_b', 'hla_c', 'cold_ischemia_time', 'dsa',
            # 'age_don', 'weight_don', 'hla_a_don', 'hla_b_don', 'hla_c_don'], dummy_na=False)
        patient_new = pd.get_dummies(patient_new, columns=['sex', 'blood_type', 'rh', 'blood_type_don', 'rh_don', 'sex_don'], dummy_na=False)

        

        return patient_new




    def run_comparison(self):
        
        for _ in range(self.n_simulations):
            simulation_results = {}
            generator = SyntheticDataGenerator(self.n_values, self.m_values, self.noise, complexity=self.complexity)
            df_patients, df_organs, df_outcomes, df_outcomes_noiseless = generator.generate_datasets()
            regression  = reg_sklearn.RegressionModel(df_patients, df_organs, df_outcomes, df_outcomes_noiseless, remote=False, split = True, scale= False)
            lin_reg = LinearRegression()
            ridge_reg = Ridge(alpha=1.0)
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

            lin_reg = regression.train_model(lin_reg)
            mse_lr, r2_lr, mse_noiseless_lr, r2_noiseless_lr = regression.evaluate_model(lin_reg)
            simulation_results['Linear Regression'] = mse_lr

            ridge_reg = regression.train_model(ridge_reg)
            mse_rr, r2_rr, mse_noiseless_rr, r2_noiseless_rr = regression.evaluate_model(ridge_reg)
            simulation_results['Ridge Regression'] = mse_rr

            rf_reg = regression.train_model(rf_reg)
            mse_rf, r2_rf, mse_noiseless_rf, r2_noiseless_rf = regression.evaluate_model(rf_reg)
            simulation_results['Random Forest Regression'] = mse_rf

            X_train, counterfactual_outcomes, _, _, _ = regression.data_handler.load_data(factual=False, outcome='eGFR_3', traintest_split=True, scale=True)

            ## TODO : need to dummy enconde and all theshit

            count_predictions = lin_reg.predict(X_train)
            count_rmse = np.sqrt(mean_squared_error(counterfactual_outcomes, count_predictions))
            count_r2 = lin_reg.score(X_train, counterfactual_outcomes)
            simulation_results['Counterfactual Linear Regression'] = count_rmse

            self.results.append(simulation_results)

        # Calculate the mean results
        mean_results = {}
        for model in ['Linear Regression', 'Ridge Regression', 'Random Forest Regression', 'Counterfactual Linear Regression']:
            mse_list = [result[model] for result in self.results]


            mean_mse = np.mean(mse_list)


            mean_results[model] = mean_mse

        # Create a table to display the mean results
        table = pd.DataFrame(mean_results, index=['MSE'])
        print(table)
        




# comparison_counterfactual = Comparison_Counterfactual(n_values=200, m_values=200, noise=5, n_simulations=5, complexity=2)
# comparison_counterfactual.run_comparison()



# n_values = [50, 100, 400, 600]
# mean_results = {'Linear Regression': [], 'Ridge Regression': [], 'Random Forest Regression': [], 'Counterfactual Linear Regression': []}

# for n in n_values:

#     comparison_counterfactual = Comparison_Counterfactual(n_values=n, m_values=n, noise=5, n_simulations=4, complexity=1)
#     comparison_counterfactual.run_comparison()
#     mean_mse_linear = np.mean([comparison_counterfactual.results[j]['Linear Regression'] for j in range(comparison_counterfactual.n_simulations)])
#     mean_mse_ridge = np.mean([comparison_counterfactual.results[j]['Ridge Regression'] for j in range(comparison_counterfactual.n_simulations)])
#     mean_mse_rf = np.mean([comparison_counterfactual.results[j]['Random Forest Regression'] for j in range(comparison_counterfactual.n_simulations)])
#     mean_mse_counterfactual = np.mean([comparison_counterfactual.results[j]['Counterfactual Linear Regression'] for j in range(comparison_counterfactual.n_simulations)])
#     mean_results['Linear Regression'].append(mean_mse_linear)
#     mean_results['Ridge Regression'].append(mean_mse_ridge)
#     mean_results['Random Forest Regression'].append(mean_mse_rf)
#     mean_results['Counterfactual Linear Regression'].append(mean_mse_counterfactual)

# # Extend the table to show the mse for the different combinations of n_values and m_values
# table = pd.DataFrame(mean_results)
# print(table)