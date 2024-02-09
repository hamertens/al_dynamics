import numpy as np

from sklearn.metrics import mean_squared_error
import pandas as pd
from models_functions import check_settling_time, train_dropout_model, check_divergence
from setup import DATA_FILEPATH, FOLDER_FILEPATH, DATAFRAME_COLUMNS_INPUT, DATAFRAME_COLUMNS_OUTPUT
import sys


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


# Predict for all possible input pairs
train_state, model = train_dropout_model(training_inputs, training_outputs, train_inputs, train_outputs)
Y_std = train_state.y_std_test
# Find max uncertainty
sum_values = Y_std.sum(axis=1)
max_index = np.argmax(sum_values)

# Append the new array to training inputs
training_inputs = np.vstack((training_inputs, train_inputs[max_index]))
# Append the new array to training outputs
training_outputs = np.vstack((training_outputs, train_outputs[max_index]))
# Calculate if convergence is met
Y_pred = train_state.y_pred_test
predictions = np.vstack((predictions, Y_pred[max_index]))
if len(predictions) >= 20:
    convergence = check_settling_time(predictions, training_outputs)
    if convergence == True:
        print("converged")
        sys.exit(1)
#Append RMSE and variance
prediction = model.predict(test_inputs)
# Calculate RMSE between the ground truth values and the predictions
RMSE = np.sqrt(mean_squared_error(test_outputs, prediction))
variance = np.sum(Y_std)


# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=DATAFRAME_COLUMNS_INPUT)
training_outputs_active_df = pd.DataFrame(training_outputs, columns=DATAFRAME_COLUMNS_OUTPUT)


# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('output_data/training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('output_data/training_outputs_active.csv', index=False)

metrics_df = pd.read_csv('output_data/metrics.csv')
new_row_metrics = pd.DataFrame({'RMSE': [RMSE], 'Variance': [variance]})
metrics_df = pd.concat([metrics_df, new_row_metrics], ignore_index=True)
# Save the updated DataFrame back to the same CSV file
metrics_df.to_csv('output_data/metrics.csv', index=False)

predictions_df = pd.DataFrame(predictions, columns=DATAFRAME_COLUMNS_OUTPUT)
predictions_df.to_csv('output_data/predictions.csv', index=False)

divergence = check_divergence(metrics_df)
if divergence == True:
    print("diverged")
    sys.exit(1)