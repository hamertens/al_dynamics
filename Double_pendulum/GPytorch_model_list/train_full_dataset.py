import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from models_functions import train_gp, gp_eval, check_settling_time
from setup import DATAFRAME_COLUMNS, DATA_FILEPATH, FOLDER_FILEPATH
import sys
import gpytorch
import torch

file_path_train_inputs = DATA_FILEPATH + 'train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = DATA_FILEPATH + 'train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

file_path_test_inputs = DATA_FILEPATH + 'test_inputs.csv'
df_test_inputs = pd.read_csv(file_path_test_inputs)
test_inputs = df_test_inputs.values

file_path_test_outputs = DATA_FILEPATH + 'test_outputs.csv'
df_test_outputs = pd.read_csv(file_path_test_outputs)
test_outputs = df_test_outputs.values

# Set the number of rows you want to choose
num_rows_to_choose = 1000

# Choose 2000 random indices
random_indices = np.random.choice(train_inputs.shape[0], size=num_rows_to_choose, replace=False)

# Select the corresponding rows from each array
train_inputs = train_inputs[random_indices]
train_outputs = train_outputs[random_indices]

train_inputs = torch.tensor(train_inputs, dtype=torch.float32)

gp_likelihood, gp_model = train_gp(train_inputs, train_outputs, 100)

prediction, var = gp_eval(test_inputs, gp_model, gp_likelihood)

RMSE = np.sqrt(mean_squared_error(test_outputs, prediction))
print(RMSE)