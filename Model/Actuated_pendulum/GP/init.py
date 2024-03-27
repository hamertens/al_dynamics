import pandas as pd
import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from setup import DATA_FILEPATH, DATAFRAME_COLUMNS_INPUT, DATAFRAME_COLUMNS_OUTPUT
from setup import TRAINING_TYPE, KERNEL

file_path_train_inputs = DATA_FILEPATH + 'train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = DATA_FILEPATH + 'train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

# Use random.randint() to generate a random index
# Generate the first random index
random_index1 = random.randint(0, len(train_inputs) - 1)

# Initialize the second random index to be the same as the first one
random_index2 = random_index1

# Keep generating a new random index until it's different from the first one
while random_index2 == random_index1:
    random_index2 = random.randint(0, len(train_inputs) - 1)

training_inputs = np.array([train_inputs[random_index1], train_inputs[random_index2]])
training_outputs = np.array([train_outputs[random_index1], train_outputs[random_index2]])

# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=DATAFRAME_COLUMNS_INPUT)
training_outputs_active_df = pd.DataFrame(training_outputs, columns=DATAFRAME_COLUMNS_OUTPUT)

# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('output_data/training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('output_data/training_outputs_active.csv', index=False)

metric_df = pd.DataFrame({'RMSE': [], 'Variance': []})
metric_df.to_csv("output_data/metrics.csv", index=False)


# initialize the kernel
if KERNEL == 'rbf':
    kernel = RBF(length_scale=1.0)
if KERNEL == 'matern':
    kernel = Matern(length_scale=1.0, nu=1.5)
    
# Create the Gaussian Process Regressor
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(training_inputs, training_outputs)
# Add initial predictions
Y_pred, _ = gpr.predict(training_inputs, return_std=True)
predictions = Y_pred
predictions_df = pd.DataFrame(predictions, columns=DATAFRAME_COLUMNS_OUTPUT)
predictions_df.to_csv('output_data/predictions.csv', index=False)

if TRAINING_TYPE == 'continuous':
    if KERNEL == 'matern':
        learned_length_scale = gpr.kernel_.length_scale
        learned_nu = gpr.kernel_.nu
        parameters_df = pd.DataFrame({'length_scale': [learned_length_scale], 'nu': [learned_nu]})
        parameters_df.to_csv('output_data/parameters.csv', index=False)
    if KERNEL == 'rbf':
        learned_length_scale = gpr.kernel_.length_scale
        parameters_df = pd.DataFrame({'length_scale': [learned_length_scale]})
        parameters_df.to_csv('output_data/parameters.csv', index=False)