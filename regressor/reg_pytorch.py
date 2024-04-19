import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import ast
from sklearn.preprocessing import StandardScaler



class DataHandler:
    def __init__(self, patients:pd.DataFrame, organs:pd.DataFrame, outcomes:pd.DataFrame, outcomes_noiseless:pd.DataFrame, remote:bool = False, onlyfactual = True):
        self.remote = remote
        self.patients = patients
        self.organs = organs
        self.outcomes = outcomes
        self.outcomes_noiseless = outcomes_noiseless
        self.merged = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_noiseless = None
        self.y_test_noiseless = None


    def get_features(self, pat_id, org_id):

        """
        Return the features of a patient and organ in the training set with a specific pat_id and org_id

        """
        patient = self.X_train[pat_id, :]


        return self.X_train, self.y_train, self.X_test, self.y_test, self.y_test_noiseless

    def load_data(self, factual:bool = True, outcome:str = 'eGFR_3', traintest_split:bool = True):
        # if self.remote:
        #     patients = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/patients.csv')
        #     organs = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/organs.csv')
        #     outcomes = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/outcomes.csv')
        #     outcomes_noiseless = pd.read_csv('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/synthetic_data_generation/outcomes_noiseless.csv')
        # else: 
        #     patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
        #     organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
        #     outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
        #     outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')

        # self.outcomes = pd.DataFrame(self.outcomes)
        # self.outcomes_noiseless = pd.DataFrame(self.outcomes_noiseless)

        self.outcomes = self.outcomes.dropna()
        self.outcomes_noiseless = self.outcomes_noiseless.dropna()

        # CAUTION: Uncomment this line if loading data from a csv file
        # self.outcomes = self.outcomes.map(ast.literal_eval)
        # self.outcomes_noiseless = self.outcomes_noiseless.map(ast.literal_eval)


        # if outcome == 'eGFR_3':
        #     outcomes = self.outcomes.map(lambda x: x['eGFR'][2] if x and 'eGFR' in x else None)
        #     outcomes_noiseless = self.outcomes_noiseless.map(lambda x: x['eGFR'][2] if x and 'eGFR' in x else None)

        # else:
        #     raise ValueError('Outcome not supported')
        


        patients = pd.get_dummies(self.patients)
        organs = pd.get_dummies(self.organs) 

        


        if factual:
            #outcomes = np.diag(outcomes.values)
            #outcomes_noiseless = np.diag(outcomes_noiseless.values)
            merged = pd.concat([patients, organs], axis=1)
            merged = merged.drop('pat_id', axis = 1)
            merged = merged.drop('org_id', axis = 1)
        else:

            """ If not factual, we match the first organ with all the patients
                reminder outcomes[i][j] is the outcome of the i-th patient with the j-th organ
            """
            # outcomes_temp = outcomes.values[:, 0]
            # outcomes_noiseless_temp = outcomes_noiseless.values[:, 0]

            # # outcomes_temp[0] = outcomes.values[0,1]
            # # outcomes_noiseless_temp[0] = outcomes_noiseless.values[0,1]

            # # Create the second DataFrame
            # df2 = pd.concat([organs.iloc[0, :]] * len(patients), axis=1).transpose()

            # # Reset the index of 'df2' and drop the old index
            # df2 = df2.reset_index(drop=True)

            # # Concatenate 'patients' and 'df2'
            # merged = pd.concat([patients, df2], axis=1)

            # merged = merged.drop('pat_id', axis = 1)
            # merged = merged.drop('org_id', axis = 1)

            # outcomes = outcomes_temp
            # outcomes_noiseless = outcomes_noiseless_temp
        




        X = merged
        y = outcomes.values
        y_noiseless = outcomes_noiseless.values

    #     Coumns of Merged:  ['age', 'weight', 'hla_a', 'hla_b', 'hla_c', 'cold_ischemia_time', 'dsa',
    #    'age_don', 'weight_don', 'hla_a_don', 'hla_b_don', 'hla_c_don',
    #    'sex_female', 'sex_male', 'blood_type_0', 'blood_type_A',
    #    'blood_type_AB', 'blood_type_B', 'rh_+', 'rh_-', 'blood_type_don_0',
    #    'blood_type_don_A', 'blood_type_don_AB', 'blood_type_don_B', 'rh_don_+',
    #    'rh_don_-', 'sex_don_female', 'sex_don_male'],
    #  



        if traintest_split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_2, _, y_train_noiseless,  y_test_noiseless = train_test_split(X, y_noiseless, test_size=0.2, random_state=42)
            return X_train, y_train, X_test, y_test, y_train_noiseless, y_test_noiseless
        else:
            X_train, y_train, X_test, y_test, y_train_noiseless, y_test_noiseless = X, y, X, y, y_noiseless, y_noiseless


 
        return X_train, y_train, X_test, y_test, y_train_noiseless, y_test_noiseless

class MLPRegressor(nn.Module):


    def __init__(self, split, scale, patients, organs, outcomes, outcomes_noiseless, remote=False):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(28, 25)##TODO change later, hardcoded now
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(25, 25)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(25,1)
        self.split = split
        self.scale = scale
        self.data_handler = DataHandler(patients, organs, outcomes, outcomes_noiseless, remote=False)
        self.X_train, self.y_train, self.X_test, self.y_test, self.y_train_noiseless, self.y_test_noiseless = self.data_handler.load_data(factual=True, outcome='eGFR_3', traintest_split=split)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

    def train_model(model, train_loader, criterion, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                #targets = targets.unsqueeze(1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}')
                    running_loss = 0.0

    def evaluate_model_train(model, train_loader, criterion, scalerx = None, scalery = None):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                #targets = targets.unsqueeze(1)
                if scalery is not None: 
                    outputs = scalery.inverse_transform(outputs)
                    targets = scalery.inverse_transform(targets)
                    outputs = torch.Tensor(outputs)
                    targets = torch.Tensor(targets)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        mse = total_loss / len(train_loader)
        return mse
    
    def evaluate_model_test(model, test_loader, criterion, scalerx = None, scalery = None):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                #targets = targets.unsqueeze(1)
                if scalery is not None: 
                    outputs = scalery.inverse_transform(outputs)
                    targets = scalery.inverse_transform(targets)
                    outputs = torch.Tensor(outputs)
                    targets = torch.Tensor(targets)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        mse = total_loss / len(test_loader)
        return mse




def run_regression(model, scale=True):
    # Define the input and output sizes
    input_size = model.X_train.shape[1]
    output_size = model.y_train.shape[0]


    if scale:
        # Normalize the data
        scalerx = StandardScaler()
        scalery = StandardScaler()
        model.X_train = scalerx.fit_transform(model.X_train)
        #model.y_train = scalery.fit_transform(model.y_train.reshape(-1, 1))
        model.y_train = scalery.fit_transform(model.y_train)

        model.X_test = scalerx.transform(model.X_test)
        model.y_test = scalery.transform(model.y_test) #reshape if not using onlydiagonal

    # Create DataLoader for training and test datasets
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(model.X_train.astype(float)), torch.Tensor(model.y_train))
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(model.X_test.astype(float)), torch.Tensor(model.y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Train the model
    num_epochs = 50
    model.train_model(train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model

    train_mse = np.sqrt(model.evaluate_model_train(train_loader, criterion, scalerx = scalerx, scalery = scalery))
    test_mse = np.sqrt(model.evaluate_model_test(test_loader, criterion, scalerx = scalerx, scalery =scalery))
    print("Train Mean Squared Error:", train_mse)
    print("Test Mean Squared Error:", test_mse)








if __name__ == '__main__':
    patients = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/patients.csv')
    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes.csv')
    outcomes_noiseless = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_noiseless.csv')
    #outcomes_diagonal = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/outcomes_diagonal.csv')
    regression_model = MLPRegressor(patients= patients, organs= organs,outcomes= outcomes, outcomes_noiseless= outcomes, remote=False, split=True, scale=True)

    print(run_regression(regression_model, scale=True))



