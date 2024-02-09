import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error
import pandas as pd
from setup import DATAFRAME_COLUMNS, DATA_FILEPATH, FOLDER_FILEPATH
from models_functions import check_settling_time
import sys
from setup import TRAINING_TYPE, KERNEL

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


file_path_training_inputs = FOLDER_FILEPATH + 'output_data/training_inputs_active.csv'
df_training_inputs = pd.read_csv(file_path_training_inputs)
training_inputs = df_training_inputs.values

file_path_training_outputs = FOLDER_FILEPATH + 'output_data/training_outputs_active.csv'
df_training_outputs = pd.read_csv(file_path_training_outputs)
training_outputs = df_training_outputs.values

file_path_predictions = FOLDER_FILEPATH + 'output_data/predictions.csv'
df_predictions = pd.read_csv(file_path_predictions)
predictions = df_predictions.values

if TRAINING_TYPE == 'continuous':
    if KERNEL == 'matern':
        file_path_parameters = FOLDER_FILEPATH + 'output_data/parameters.csv'
        parameters_df = pd.read_csv(file_path_parameters)
        length_scale = parameters_df.iloc[-1]['length_scale']
        nu = parameters_df.iloc[-1]['nu']
    if KERNEL == 'rbf':
        file_path_parameters = FOLDER_FILEPATH + 'output_data/parameters.csv'
        parameters_df = pd.read_csv(file_path_parameters)
        length_scale = parameters_df.iloc[-1]['length_scale']

# initialize the kernel
if TRAINING_TYPE == 'continuous':
    if KERNEL == 'matern':
        kernel = Matern(length_scale=length_scale, nu=nu)
    if KERNEL == 'rbf':
        kernel = RBF(length_scale=length_scale)
else:
    if KERNEL == 'matern':
        kernel = Matern(length_scale=1.0, nu=1.5)
    if KERNEL == 'rbf':
        kernel = RBF(length_scale=1.0)

# Create the Gaussian Process Regressor
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(training_inputs, training_outputs)

if TRAINING_TYPE == 'continuous':
    if KERNEL == 'matern':
        learned_length_scale = gpr.kernel_.length_scale
        learned_nu = gpr.kernel_.nu
        new_row_parameters_df = pd.DataFrame({'length_scale': [learned_length_scale], 'nu': [learned_nu]})
    if KERNEL == 'rbf':
        learned_length_scale = gpr.kernel_.length_scale
        new_row_parameters_df = pd.DataFrame({'length_scale': [learned_length_scale]})
    
    parameters_df = pd.concat([parameters_df, new_row_parameters_df], ignore_index=True)
    parameters_df.to_csv('output_data/parameters.csv', index=False)


# Predict for all possible input pairs
Y_pred, Y_std = gpr.predict(train_inputs, return_std=True)
# Find max uncertainty
sum_values = Y_std.sum(axis=1)
max_index = np.argmax(sum_values)

# Append the new array to training inputs
training_inputs = np.vstack((training_inputs, train_inputs[max_index]))
# Append the new array to training outputs
training_outputs = np.vstack((training_outputs, train_outputs[max_index]))
# Calculate if convergence is met
predictions = np.vstack((predictions, Y_pred[max_index]))
if len(predictions) >= 20:
    convergence = check_settling_time(predictions, training_outputs)
    if convergence == True:
        sys.exit(1)
#Append RMSE and variance
prediction = gpr.predict(test_inputs, return_std=False)
# Calculate RMSE between the ground truth values and the predictions
RMSE = np.sqrt(mean_squared_error(test_outputs, prediction))
variance= np.sum(Y_std)


# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=DATAFRAME_COLUMNS)
training_outputs_active_df = pd.DataFrame(training_outputs, columns=DATAFRAME_COLUMNS)


# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('output_data/training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('output_data/training_outputs_active.csv', index=False)


metrics_df = pd.read_csv('output_data/metrics.csv')
new_row_metrics = pd.DataFrame({'RMSE': [RMSE], 'Variance': [variance]})
metrics_df = pd.concat([metrics_df, new_row_metrics], ignore_index=True)
# Save the updated DataFrame back to the same CSV file
metrics_df.to_csv('output_data/metrics.csv', index=False)

predictions_df = pd.DataFrame(predictions, columns=DATAFRAME_COLUMNS)
predictions_df.to_csv('output_data/predictions.csv', index=False)