import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import TensorDataset
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm



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
            nn.Linear(64, 11)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def train_model(model, criterion, optimizer, dataloader, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = model.to(device) 
    loss_history = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(dataloader)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.long()
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            progress_bar.set_description(f"Loss: {loss.item()}")

        loss_history.append(loss.item())
        print("epoch done")

    return loss_history


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the synthetic data
    patients = pd.read_csv('./synthetic_data_generation/patients.csv')
    organs = pd.read_csv('./synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('./synthetic_data_generation/outcomes.csv')

    outcomes = outcomes.applymap(ast.literal_eval)

    

    #Delete the other outcomes. TODO: all outcomes
    outcomes = outcomes.applymap(lambda x: x['eGFR'][3] if x and 'eGFR' in x else None)

    


    #merge the dataframes
    # Add a temporary key for merging
    patients['key'] = 0
    organs['key'] = 0

    # Perform the cross join
    merged = pd.merge(patients, organs, on='key', suffixes=('_patient', '_organ'))

    # Drop the temporary key
    merged.drop('key', axis=1, inplace=True)

    # Create the id column
    merged['id'] = merged['pat_id'].astype(str) + '_' + merged['org_id'].astype(str)

    # Move the id column to the first position
    cols = ['id'] + [col for col in merged.columns if col != 'id']
    merged = merged[cols]
    merged = merged.drop(['pat_id', 'org_id'], axis=1)


    # One-hot encode the categorical columns
    # patients_encoded = pd.get_dummies(patients)
    # organs_encoded = pd.get_dummies(organs)
    merged_encoded = pd.get_dummies(merged)

    # Convert boolean values to integers
    # patients_encoded = patients_encoded.astype(int)
    # organs_encoded = organs_encoded.astype(int)
    outcomes = outcomes.astype(int)
    merged_encoded = merged_encoded.astype(int)


    


    #### Errror here cannot convert to tensro
    # Convert the DataFrames to tensors
    # Remove non-numeric columns
    # patients_tensor = torch.tensor(patients_encoded.values, dtype=torch.float32)
    # organs_tensor = torch.tensor(organs_encoded.values, dtype=torch.float32)
    outcomes_tensor = torch.tensor(outcomes.values, dtype=torch.float32)
    merged_tensor = torch.tensor(merged_encoded.values, dtype=torch.float32)

    #change to values between 1 and 10
    outcomes_tensor = outcomes_tensor/10
    outcomes_tensor = outcomes_tensor.round()

    #get only diagonal entries: TODO: all entries
    #outcomes_tensor = torch.diag(outcomes_tensor)
    outcomes_tensor = outcomes_tensor.transpose(0,1)
    outcomes_tensor = torch.reshape(outcomes_tensor, (1,-1))
    outcomes_tensor = outcomes_tensor.transpose(0,1)


    # OLD:Concatenate the patients and organs tensors along the second dimension
    #data = torch.cat((patients_tensor, organs_tensor), dim=1)


    #OLD: no test/train split
    # # Create a dataset from the data and outcomes
    # dataset = TensorDataset(merged_tensor, outcomes_tensor)
    # # Create a dataloader for the synthetic data
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


    # Create the model
    #OLD: input_size = patients_tensor.shape[1] + organs_tensor.shape[1]
    input_size = merged_tensor.shape[1]
    model = MLP(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(merged_tensor, outcomes_tensor, test_size=0.2, random_state=42)

    # Create DataLoaders for the training set and the test set
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=10)

    # Train the model
    loss_history = train_model(model, criterion, optimizer, train_dataloader)


    # Print the final loss
    print('train loss:', loss_history[-1])

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig('./classifier/loss_history.png')

    #sSave the model
    torch.save(model.state_dict(), './classifier/model.pth')



    # Load the state dict previously saved
    state_dict = torch.load('./classifier/model.pth')

    # Update the model's state dict
    model.load_state_dict(state_dict)
    model = model.to(device)

     # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.long()
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_dataloader)

    print('Test loss:', test_loss)


if __name__ == '__main__':
    main()