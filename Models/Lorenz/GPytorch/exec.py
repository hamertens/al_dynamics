import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from models_functions import train_gp, gp_eval
import sys


file_path_train_inputs = '/home/hansm/active_learning/Lorenz/GPytorch/data/lorenz_train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = '/home/hansm/active_learning/Lorenz/GPytorch/data/lorenz_train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

file_path_test_inputs = '/home/hansm/active_learning/Lorenz/GPytorch/data/lorenz_test_inputs.csv'
df_test_inputs = pd.read_csv(file_path_test_inputs)
test_inputs = df_test_inputs.values

file_path_test_outputs = '/home/hansm/active_learning/Lorenz/GPytorch/data/lorenz_test_outputs.csv'
df_test_outputs = pd.read_csv(file_path_test_outputs)
test_outputs = df_test_outputs.values


file_path_training_inputs = '/home/hansm/active_learning/Lorenz/GPytorch/training_inputs_active.csv'
df_training_inputs = pd.read_csv("training_inputs_active.csv")
training_inputs = df_training_inputs.values

file_path_training_outputs = '/home/hansm/active_learning/Lorenz/GPytorch/training_outputs_active.csv'
df_training_outputs = pd.read_csv("training_outputs_active.csv")
training_outputs = df_training_outputs.values

file_path_predictions = '/home/hansm/active_learning/Lorenz/GPytorch/predictions.csv'
df_predictions = pd.read_csv(file_path_predictions)
model_predictions = df_predictions.values


def check_settling_time(prediction, goal):
    # Ensure both arrays have at least 20 entries
    if len(prediction) < 20 or len(goal) < 20:
        raise ValueError("Both arrays should have at least 20 entries.")

    # Take the last 20 entries from both arrays
    last_20_prediction = prediction[-30:-10]
    last_20_goal = goal[-30:-10]

    # Calculate the maximum allowed average error (2%)
    max_average_error = 0.02

    # Check if the average error across all dimensions is within the 2% error band
    for val1, val2 in zip(last_20_prediction, last_20_goal):

        errors = np.abs((val1 - val2) / val2)  # Calculate the percentage error for each dimension
        avg_error = np.mean(errors)  # Calculate the average error across all dimensions
        if avg_error > max_average_error:
            return False  # Average error exceeds the 2% limit

    return True  # Average error is within the 2% limit for all 20 entries

iterations = 500
# Train on current inputs and outputs
likelihood, model = train_gp(training_inputs, training_outputs, iterations)

# Predict for all possible input pairs
Y_pred, Y_std = gp_eval(train_inputs, model, likelihood)
# Find max uncertainty
sum_values = Y_std[:, 0] + Y_std[:, 1] + Y_std[:, 2]
max_index = np.argmax(sum_values)
# Append the new array to training inputs
training_inputs = np.vstack((training_inputs, train_inputs[max_index]))
# Append the new array to training outputs
training_outputs = np.vstack((training_outputs, train_outputs[max_index]))
# Calculate if convergence is met
model_predictions = np.vstack((model_predictions, Y_pred[max_index]))

if len(model_predictions) >= 20:
    convergence = check_settling_time(model_predictions, training_outputs)
    if convergence == True:
        sys.exit(1)
#Append RMSE and variance
prediction, _ = gp_eval(test_inputs, model, likelihood)
# Calculate RMSE between the ground truth values and the predictions
RMSE = np.sqrt(mean_squared_error(test_outputs, prediction))

variance= np.sum(Y_std)

# Calculate training rmse
train_rmse = np.sqrt(mean_squared_error(train_outputs, Y_pred))

# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=['x', 'y', 'z'])
training_outputs_active_df = pd.DataFrame(training_outputs, columns=['x', 'y', 'z'])


# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('training_outputs_active.csv', index=False)

metrics_df = pd.read_csv('metrics.csv')
new_row_metrics = pd.DataFrame({'RMSE': [RMSE], 'Variance': [variance], "Train_RMSE": [train_rmse]})
metrics_df = pd.concat([metrics_df, new_row_metrics], ignore_index=True)
# Save the updated DataFrame back to the same CSV file
metrics_df.to_csv('metrics.csv', index=False)

predictions_df = pd.DataFrame(model_predictions, columns=['x', 'y', 'z'])
predictions_df.to_csv('predictions.csv', index=False)