import sys
import os
import pandas as pd
config_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config'))
sys.path.append(config_path)
from config import config  # noqa: E402

sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\synthetic_data_generation")
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET")
sys.path.append(r"C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/regressor")
sys.path.append(r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\Meta-Learners")
from SLearner import *




class Evaluate:
    def __init__(self):
        self.model = config['evaluation']['model']
        self.metric = config['evaluation']['metric']
        self.learner = config['evaluation']['learner']
        

    def make_table_cate(self):
        if self.metric != 'CATE':
            raise NotImplementedError("Only CATE metric is implemented")
        
        # Define the table data
        columns = ['Train', 'Test']
        rows = ['factual', 'counterfactual']


        #get the data
        self.learner = eval(self.learner)
        
        data = [[self.learner.get_pehe_train_factual(), self.learner.get_pehe_test_factual()], [self.learner.get_pehe_train_count(), self.learner.get_pehe()]]
        
        # Create the table using pandas DataFrame
        table = pd.DataFrame(data, index=rows, columns=columns)
        
        # Set the title of the table
        title = config['evaluation']['metric']
        
        # Save the table as a file
        table_file =  r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\Evaluation\results_cate.csv" # Replace with your desired file path
        table.to_csv(table_file)
        
        # Append the title to the file
        with open(table_file, 'r+') as f:
            content = f.read()
            f.seek(0)
            f.write(f'{title}' + content)



    def make_table_outcomes(self):
        #define the table data
        columns = ['Train', 'Test']
        rows = ['factual', 'counterfactual']

        #get the data
        self.learner = eval(self.learner)
        data = [[self.learner.get_outcome_error_train_factual(), self.learner.get_outcome_error_test_factual()], [self.learner.get_outcome_error_train_count(), self.learner.get_outcome_error_test_count()]]

        #create the table using pandas DataFrame
        table = pd.DataFrame(data, index=rows, columns=columns)

        #set the title of the table
        title = config['evaluation']['metric']

        #save the table as a file
        table_file =  r"C:\Users\Ernesto\OneDrive - ETH Zurich\Desktop\MT\COMET\Evaluation\results_outcome.csv"
        table.to_csv(table_file)

        #append the title to the file
        with open(table_file, 'r+') as f:
            content = f.read()
            f.seek(0)
            f.write(f'{title}' + content)   








if __name__ == "__main__":
    evaluate = Evaluate()
    #evaluate.make_table_cate()
    evaluate.make_table_outcomes()

    



