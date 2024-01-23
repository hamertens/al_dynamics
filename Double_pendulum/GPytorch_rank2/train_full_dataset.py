import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from models_functions import train_gp, gp_eval, check_settling_time
from setup import DATAFRAME_COLUMNS, DATA_FILEPATH, FOLDER_FILEPATH
import sys
import gpytorch

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
num_rows_to_choose = 3000

# Choose 2000 random indices
random_indices = np.random.choice(train_inputs.shape[0], size=2000, replace=False)

# Select the corresponding rows from each array
train_inputs = train_inputs[random_indices]
train_outputs = train_outputs[random_indices]


# Set the number of iterations
iterations = 800

# Train on entire dataset
likelihood, model = train_gp(train_inputs, train_outputs, iterations)

# Predict on test dataset
prediction, _ = gp_eval(test_inputs, model, likelihood)
# Calculate RMSE between the ground truth values and the predictions
RMSE = np.sqrt(mean_squared_error(test_outputs, prediction))

metric_df = pd.DataFrame({'RMSE': [RMSE]})
metric_df.to_csv("output_data/metrics_full_training.csv", index=False)