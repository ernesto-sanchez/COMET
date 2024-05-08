import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import TensorDataset
import ast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn import metrics
from reg_sklearn import DataHandler
from econml.dml import LinearDML


import matplotlib.pyplot as plt




est = LinearDML()
est.fit(y, T, X=X, W=W)
est.const_marginal_effect(X)
