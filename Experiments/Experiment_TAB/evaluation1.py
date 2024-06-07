#Script to get the evaluation of the peorfomance of the models when the TAB paramenter varies from 0 to 1.



import configparser
import os
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt



project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
script_path = os.path.dirname(__file__)
config_path = os.path.join(project_path, 'config', 'config1.ini')
sys.path.append(os.path.join(project_path, 'Meta-Learners'))
sys.path.append(os.path.join(project_path, 'DML'))
sys.path.append(os.path.join(project_path, 'synthetic_data_generation'))
sys.path.append(os.path.join(project_path, 'Evaluation'))



# Export the config file path to the config corresponding to this experiment:
os.environ['CONFIG_FILE'] = config_path



from evaluation import Evaluate #must be imported after od.environ --> Must be a child process
from synthetic_data_faster import SyntheticDataGenerator


# Create a config parser
config = configparser.ConfigParser()

# Read the config file
config.read(config_path)

# Record the path of the place where the results are stored and where the data comes from
config['evaluation']['results_path'] = os.path.join(script_path, 'results_experiment_1.h5')
config['data']['path_patients'] = os.path.join(project_path, 'synthetic_data_generation', 'patients.csv')
config['data']['path_organs'] = os.path.join(project_path, 'synthetic_data_generation', 'organs.csv')
config['data']['path_outcomes'] = os.path.join(project_path, 'synthetic_data_generation', 'outcomes.csv')
config['data']['path_outcomes_noiseless'] = os.path.join(project_path, 'synthetic_data_generation', 'outcomes_noiseless.csv')
config['data']['path_effects'] = os.path.join(project_path, 'synthetic_data_generation', 'effects.csv')

with open(config_path, 'w') as configfile:
    config.write(configfile)

# Results dictionary
results = {}

for tab in [0,0.1, 0.2,0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]:

    config['synthetic_data']['TAB'] = str(tab)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    # Generate the data
    generator = SyntheticDataGenerator()
    df_patients, df_organs, df_outcomes, df_outcomes_noiseless, df_effects = generator.generate_datasets()




    try:
        df_patients.to_csv(config['data']['path_patients'], index=False)
        df_organs.to_csv(config['data']['path_organs'], index=False)
        df_outcomes.to_csv(config['data']['path_outcomes'], index=False)
        df_outcomes_noiseless.to_csv(config['data']['path_outcomes_noiseless'], index=False)
        df_effects.to_csv(config['data']['path_effects'], index=False)
    except Exception as e:
        print(f"An error occurred while writing the data to CSV files: {e}")

    for model in ['S_Learner()','T_Learner()', 'DoubleML()', 'DRLearner()']:
        config['evaluation']['parameter'] = 'tab'
        config['evaluation']['parameter_value'] = str(tab)
        config['evaluation']['learner'] = str(model) 

        if model == 'DRLearner()' or model == 'DoubleML()':
            config['evaluation']['learner_type'] = 'DoubleML()'
        else:
            config['evaluation']['learner_type'] = str(model)


        with open(config_path, 'w') as configfile:
            config.write(configfile)

        evaluate = Evaluate()
        if config['evaluation']['metric'] == 'CATE':
            table = evaluate.make_table_cate()
        else:
            table = evaluate.make_table_outcomes()

        # Initialize a new dictionary for the model if it doesn't exist
        if model not in results:
            results[model] = {}
        
        results[model][tab] = table

# Save the results

with open(os.path.join(script_path, 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)


# Plot the results

x_values = [0,0.1, 0.2,0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Define the values for the x-axis
colors = ['red', 'blue', 'green', 'orange']  # Define the colors for each line

for model, color in zip(results.keys(), colors):
    y_values = [results[model][tab].loc['counterfactual','Test'] for tab in x_values]  # Get the error values for each model
    plt.plot(x_values, y_values, color=color, label=model)  # Plot the line for each model

plt.xlabel('TAB')  # Set the label for the x-axis
plt.ylabel('CATE')  # Set the label for the y-axis
plt.legend()  # Show the legend for the lines
plt.savefig(os.path.join(script_path, 'results.pdf'), dpi=300)  # Save the plot as a PDF file with high resolution
plt.show()  # Display the plot







    




        




