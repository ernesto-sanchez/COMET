import numpy as np
import torch
from torch import nn

from typing import Optional
from typing import Any, Optional

import numpy as np

import wandb

# default model architectures
DEFAULT_LAYERS_OUT = 2
DEFAULT_LAYERS_OUT_T = 2
DEFAULT_LAYERS_R = 3
DEFAULT_LAYERS_R_T = 3

DEFAULT_UNITS_OUT = 100
DEFAULT_UNITS_R = 200
DEFAULT_UNITS_OUT_T = 100
DEFAULT_UNITS_R_T = 200

DEFAULT_NONLIN = "elu"

# other default hyperparameters
DEFAULT_STEP_SIZE = 0.0001
DEFAULT_STEP_SIZE_T = 0.0001
DEFAULT_N_ITER = 10000
DEFAULT_BATCH_SIZE = 20

DEFAULT_PENALTY_L2 = 1e-2
DEFAULT_PENALTY_DISC = 0
DEFAULT_PENALTY_ORTHOGONAL = 1 / 100
DEFAULT_AVG_OBJECTIVE = True

# defaults for early stopping
DEFAULT_VAL_SPLIT = 0.3
DEFAULT_N_ITER_MIN = 15
DEFAULT_PATIENCE = 5

# Defaults for crossfitting
DEFAULT_CF_FOLDS = 2

# other defaults
DEFAULT_SEED = 42
DEFAULT_N_ITER_PRINT = 10
LARGE_VAL = np.iinfo(np.int32).max

DEFAULT_UNITS_R_BIG_S = 100
DEFAULT_UNITS_R_SMALL_S = 50

DEFAULT_UNITS_R_BIG_S3 = 150
DEFAULT_UNITS_R_SMALL_S3 = 50

N_SUBSPACES = 3
DEFAULT_DIM_S_OUT = 50
DEFAULT_DIM_S_R = 100
DEFAULT_DIM_P_OUT = 50
DEFAULT_DIM_P_R = 100

TRAIN_STRING = "training"
VALIDATION_STRING = "validation"

EPS = 1e-8

DEVICE='cpu'

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
}

wandb.init(project="COMET")

def make_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, X, y, TRAIN_STRING

        return X, y, w, X, y, w, TRAIN_STRING

    X = X.cpu()
    y = y.cpu()
    # make actual split
    if w is None:
        X_t, X_val, y_t, y_val = train_test_split(
            X, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return (
            X_t.to(DEVICE),
            y_t.to(DEVICE),
            X_val.to(DEVICE),
            y_val.to(DEVICE),
            VALIDATION_STRING,
        )

    w = w.cpu()
    if stratify_w:
        # split to stratify by group
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X,
            y,
            w,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X, y, w, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return (
        X_t.to(DEVICE),
        y_t.to(DEVICE),
        w_t.to(DEVICE),
        X_val.to(DEVICE),
        y_val.to(DEVICE),
        w_val.to(DEVICE),
        VALIDATION_STRING,
    )



class PropensityNet(nn.Module):

    def __init__(
        self,
        name: str,
        n_unit_in: int,
        n_unit_out: int,
        weighting_strategy: str,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        nonlin: str = DEFAULT_NONLIN,
        lr: float = DEFAULT_STEP_SIZE,
        weight_decay: float = DEFAULT_PENALTY_L2,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        clipping_value: int = 1,
        batch_norm: bool = True,
        early_stopping: bool = False,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ) -> None:
        super(PropensityNet, self).__init__()
        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]

        if batch_norm:
            layers = [
                nn.Linear(in_features=n_unit_in, out_features=n_units_out_prop),
                nn.BatchNorm1d(n_units_out_prop),
                NL(),
            ]
        else:
            layers = [
                nn.Linear(in_features=n_unit_in, out_features=n_units_out_prop),
                NL(),
            ]

        for i in range(n_layers_out_prop - 1):
            if dropout:
                layers.extend([nn.Dropout(dropout_prob)])
            if batch_norm:
                layers.extend(
                    [
                        nn.Linear(
                            in_features=n_units_out_prop, out_features=n_units_out_prop
                        ),
                        nn.BatchNorm1d(n_units_out_prop),
                        NL(),
                    ]
                )
            else:
                layers.extend(
                    [
                        nn.Linear(
                            in_features=n_units_out_prop, out_features=n_units_out_prop
                        ),
                        NL(),
                    ]
                )
        layers.extend(
            [
                nn.Linear(in_features=n_units_out_prop, out_features=n_unit_out),
                nn.Softmax(dim=-1),
            ]
        )

        self.model = nn.Sequential(*layers).to(DEVICE)
        self.name = name
        self.weighting_strategy = weighting_strategy
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
#         print(self.model)
        return self.model(X)

    def get_importance_weights(
        self, X: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        p_pred = self.forward(X).squeeze()[:, 1]
        return compute_importance_weights(p_pred, w, self.weighting_strategy, {})

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:



        return nn.NLLLoss()(torch.log(y_pred + EPS), y_target)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "PropensityNet":
        self.train()

        X = self._check_tensor(X)
        y = self._check_tensor(y).long()

        # get validation split (can be none)
        X, y, X_val, y_val, val_string = make_val_split(
            X, y, val_split_prop=self.val_split_prop, seed=self.seed
        )
        y_val = y_val.squeeze()
        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        # do train        
        val_loss_best = LARGE_VAL
        patience = 0
        for i in range(1000):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                self.optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next]
#                 print(X_next.shape)
                
                y_next = y[idx_next].squeeze()

                preds = self.forward(X_next.float()).squeeze()

                batch_loss = self.loss(preds, y_next)

                wandb.log({"batch_loss": torch.mean(batch_loss)})

                batch_loss.backward()

                #new code

                # Convert predicted probabilities to class labels
                preds_labels = torch.argmax(preds, dim=1)

                # Calculate accuracy
                accuracy = (preds_labels == y_next).float().mean()

                wandb.log({"accuracy": accuracy})

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                self.optimizer.step()
                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    preds = self.forward(X_val).squeeze()
                    val_loss = self.loss(preds, y_val)

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience > self.patience and (
                            (i + 1) * n_batches > self.n_iter_min
                        ):
                            break
                    if i % self.n_iter_print == 0:
                        print(
                            f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                        )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)



import argparse
import pandas as pd
import torch

import os
# import pyro
import numpy as np
print(os.getcwd())

import pandas as pd

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split



def load_data(dataset_name = 'acic', current_id='0'):
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.dirname(project_path)


    
    # data path
    if dataset_name == 'acic2016':
            dataset_path = "./data_acic2016/acic2016_norm_data/" + current_id + ".csv"
            print('dataset_path', dataset_path)

    if dataset_name == 'acic2018':
            # dataset_path = "./data_acic2018/acic2018_norm_data/" + current_id + ".csv"
            dataset_path = os.path.join(data_path, "ACIC2018", "merged", current_id + "_merged" + ".csv")

            print('dataset_path', dataset_path)

    if dataset_name == 'synthetic':
        dataset_path = os.path.join(data_path, "ACIC2018", "merged", "synthetic_merged.csv")
        print('dataset_path', dataset_path)
    
    # load data
    load_csv = pd.read_csv(dataset_path, sep = ',', decimal = ',')
    load_table = load_csv.values.astype("float32")

    # get x and t from load table
    if dataset_name == 'acic2016':
        x_dim = 82
    if dataset_name == 'acic2018':
        # x_dim = 177
        #new code: --> Need to take sapmle_id out of the network!!!
        x_dim = 176

    if dataset_name == 'synthetic':
        x_dim = 19
    



    x = load_table[:, 5:] # 0-4 collum is not x
    t = load_table[:, 0].reshape(-1, 1)

    # initialize
    pi = PropensityNet(
                "slearner_prop_estimator",
                x_dim,
                10,  # number of treatments
                "ipw",
                n_units_out_prop=DEFAULT_UNITS_OUT,
                n_layers_out_prop=0,
                weight_decay=DEFAULT_PENALTY_L2,
                lr=DEFAULT_STEP_SIZE,
                n_iter=1000,
                batch_size=DEFAULT_BATCH_SIZE,
                n_iter_print=DEFAULT_N_ITER_PRINT,
                seed=DEFAULT_SEED,
                nonlin=DEFAULT_NONLIN,
                val_split_prop=DEFAULT_VAL_SPLIT,
                batch_norm=False,
                early_stopping=True,
                dropout=False,
                dropout_prob=0.2,
            )
    # train with early stop
    print(type(x))
    if isinstance(x, np.ndarray):
        pi.fit(torch.from_numpy(x).float(), torch.from_numpy(np.squeeze(t)).long())

        p_pred = pi.forward(torch.from_numpy(x).float())

    else:
        x = x.float().to(device)
        t = t.long().to(device)

        pi.fit(x, t)
    
    pred_t = torch.argmax(p_pred, dim = 1)

    print('============== Finish training propnet on this dataset') 
    return pi

    




#new code

if __name__ == '__main__':
    load_data(dataset_name = 'synthetic')