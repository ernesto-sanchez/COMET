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
from sklearn.model_selection import KFold




"""
This Script takes in the input data and returns the predicted class of the input data.
For simplicity, we are only predicting the outcome eGFR in the in the first/second/3rd year.

We willl trian an MLP for the task, taking in the patient and organ features as input and outputting the predicted eGFR.
We will explain here the hyperparams of the model (n layers/architecture, learning rate, etc) and the input/output of the model.

"""

class MLP_class(nn.Module):
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
    
class MLP_reg(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 11), 
            nn.ReLU(),
            nn.Linear(11, 1)
        )

    def forward(self, x):
        output = self.layers(x)
        return output



def train_model(model, criterion, optimizer, dataloader, epochs=30, l1_lambda=0.000):
    l1_lambda = l1_lambda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = model.to(device) 
    loss_history = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(dataloader)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.long()
            labels = labels.view(-1)
            loss = criterion(outputs, labels)

            # Add L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

            loss.backward()
            optimizer.step()


            progress_bar.set_description(f"Loss: {loss.item()}")

        loss_history.append(loss.item())
        print("epoch done")

    return loss_history


def calculate_test_loss(model, criterion, testloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.long()
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(testloader)  # Return the average loss



def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the synthetic data
    patients = pd.read_csv('./synthetic_data_generation/patients.csv')
    organs = pd.read_csv('./synthetic_data_generation/organs.csv')
    outcomes = pd.read_csv('./synthetic_data_generation/outcomes.csv')
    outcomes = outcomes.dropna()

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
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=1)

    # Train the model
    #loss_history = train_model(model, criterion, optimizer, train_dataloader, l1_lambda=0.0005)

    # ## K-Fold Cross Validation
    # k_folds = 5
    # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # for fold, (train_ids, test_ids) in enumerate(kfold.split(merged_tensor)):
    #     # Print
    #     print(f'FOLD {fold}')
    #     print('--------------------------------')
        
    #     # Sample elements randomly from a given list of ids, no replacement.
    #     train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    #     test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
    #     # Define data loaders for training and testing data in this fold
    #     trainloader = torch.utils.data.DataLoader(
    #                     TensorDataset(merged_tensor, outcomes_tensor), 
    #                     batch_size=10, sampler=train_subsampler)
    #     testloader = torch.utils.data.DataLoader(
    #                     TensorDataset(merged_tensor, outcomes_tensor),
    #                     batch_size=1, sampler=test_subsampler)

    #     # Init the neural network
    #     model = MLP(input_size)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #     # Train the model
    #     loss_history = train_model(model, criterion, optimizer, trainloader)
        

    #     # Print the final loss
    #     print('train loss:', loss_history[-1])

    #     test_loss = calculate_test_loss(model, criterion, testloader)
    #     print(f'Test loss: {test_loss}')
    


    # # Print the final loss
    # print('train loss:', loss_history[-1])

    # # Plot the loss history
    # plt.plot(loss_history)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # #plt.show()
    # plt.savefig('./classifier/loss_history.png')

    # #sSave the model
    # torch.save(model.state_dict(), './classifier/model.pth')



    # Load the state dict previously saved
    state_dict = torch.load('./classifier/model.pth')

    # Update the model's state dict
    model.load_state_dict(state_dict)
    model = model.to(device)

     # Evaluate the model on the test set
    model.eval()
    correct_predictions = 0
    with torch.no_grad():
        test_loss = 0
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.long()
            labels = labels.view(-1)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct_predictions += (predicted == labels).sum().item()


            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_dataloader)

    

    print('Test loss:', test_loss)
    accuracy = correct_predictions / len(test_dataloader.dataset)
    print('Test accuracy:', accuracy)


if __name__ == '__main__':
    main()