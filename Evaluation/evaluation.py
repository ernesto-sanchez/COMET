import sys
import os
import pandas as pd
import configparser
project_path = os.path.dirname(os.path.dirname(__file__))


# Create a config parser
config = configparser.ConfigParser()

config_file = os.getenv('CONFIG_FILE', os.path.join(project_path, 'config', 'config1.ini'))

# Read the config file
config.read(config_file)



sys.path.append(os.path.join(project_path, 'Meta-Learners'))
sys.path.append(os.path.join(project_path, 'DML'))

from SLearner import *
from TLearner_econml import *
from DML import *




class Evaluate:
    def __init__(self):
        self.model = config['evaluation']['model']
        self.metric = config['evaluation']['metric']
        self.learner_type = config['evaluation']['learner_type']
        

    def make_table_cate(self):
        if self.metric != 'CATE':
            raise NotImplementedError("Only CATE metric is implemented")
        
        # Define the table data
        columns = ['Train', 'Test']
        rows = ['factual', 'counterfactual']

        #get the TAB value
        parameter_value = os.getenv('PARAMETER_VALUE', config['synthetic_data'].get('TAB'))
        parameter = os.getenv('PARAMETER', 'tab')





        #get the data
        self.learner_type = eval(self.learner_type)
        
        data = [[self.learner_type.get_pehe_train_factual(), self.learner_type.get_pehe_test_factual()],
                 [self.learner_type.get_pehe_train_count(), self.learner_type.get_pehe()]]
        
       
        

        #create the table using pandas DataFrame
        table = pd.DataFrame(data, index=rows, columns=columns)



        results_file = os.getenv('RESULTS_FILE', os.path.join(project_path, 'Evaluation', 'results_cate'))


        # Save the result to the HDF5 file
        with pd.HDFStore(results_file) as store:
            store[f'{parameter}={parameter_value}'] = table



    def make_table_outcomes(self):
        #define the table data
        columns = ['Train', 'Test']
        rows = ['factual', 'counterfactual']

        #get the TAB value
        tab_value = config['synthetic_data'].get('TAB')

        #get the data
        self.learner_type = eval(self.learner_type)

        data = [[self.learner_type.get_outcome_error_train_factual(), self.learner_type.get_outcome_error_test_factual()],
                 [self.learner_type.get_outcome_error_train_count(), self.learner_type.get_outcome_error_test_count()]]

        #create the table using pandas DataFrame
        table = pd.DataFrame(data, index=rows, columns=columns)


       

        results_file = os.getenv('RESULTS_FILE', os.path.join(project_path, 'Evaluation', 'results_outcome'))

        # Save the result to the HDF5 file
        with pd.HDFStore(results_file) as store:
            store[f'tab_{tab_value}'] = table









if __name__ == "__main__":
    evaluate = Evaluate()
    if config['evaluation']['metric'] == 'CATE':
        evaluate.make_table_cate()
    else:
        evaluate.make_table_outcomes()

    



