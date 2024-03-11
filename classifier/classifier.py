import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import TensorDataset
import ast


"""
This Script takes in the input data and returns the predicted class of the input data.
For simplicity, we are only predicting the outcome eGFR in the in the first/second/3rd year.

We willl trian an MLP for the task, taking in the patient and organ features as input and outputting the predicted eGFR.
We will explain here the hyperparams of the model (n layers/architecture, learning rate, etc) and the input/output of the model.

"""

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def train_model(model, criterion, optimizer, dataloader, epochs=100):
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            #inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



def main():
    # Load the synthetic data
    patients = pd.read_csv('./synthetic_data_generation/patients.csv')
    organs = pd.read_csv('./synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('./synthetic_data_generation/outcomes.csv')

    outcomes = outcomes.applymap(ast.literal_eval)

    

    #Delete the other outcomes
    outcomes = outcomes.applymap(lambda x: x['eGFR'][0] if x and 'eGFR' in x else None)

    
    

    # One-hot encode the categorical columns
    patients_encoded = pd.get_dummies(patients)
    organs_encoded = pd.get_dummies(organs)

    # Convert boolean values to integers
    patients_encoded = patients_encoded.astype(int)
    organs_encoded = organs_encoded.astype(int)
    outcomes = outcomes.astype(int)


    #### Errror here cannot convert to tensro
    # Convert the DataFrames to tensors
    # Remove non-numeric columns
    patients_tensor = torch.tensor(patients_encoded.values, dtype=torch.float32)
    organs_tensor = torch.tensor(organs_encoded.values, dtype=torch.float32)
    outcomes_tensor = torch.tensor(outcomes.values, dtype=torch.float32)

    # Concatenate the patients and organs tensors along the second dimension
    data = torch.cat((patients_tensor, organs_tensor), dim=1)

    # Create a dataset from the data and outcomes
    dataset = TensorDataset(data, outcomes_tensor)
    # Create a dataloader for the synthetic data
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create the model
    input_size = len(patients.columns) + len(organs.columns)
    model = MLP(input_size)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, loss, optimizer, dataloader)

    # Save the model
    torch.save(model.state_dict(), './classifier/model.pth')


if __name__ == '__main__':
    main()