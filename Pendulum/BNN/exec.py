import numpy as np
import deepxde as dde
from sklearn.metrics import mean_squared_error
import pandas as pd

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


file_path_train_inputs = '/home/hansm/active_learning/Pendulum/BNN/data/train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = '/home/hansm/active_learning/Pendulum/BNN/data/train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

file_path_test_inputs = '/home/hansm/active_learning/Pendulum/BNN/data/test_inputs.csv'
df_test_inputs = pd.read_csv(file_path_test_inputs)
test_inputs = df_test_inputs.values

file_path_test_outputs = '/home/hansm/active_learning/Pendulum/BNN/data/test_outputs.csv'
df_test_outputs = pd.read_csv(file_path_test_outputs)
test_outputs = df_test_outputs.values


file_path_training_inputs = '/home/hansm/active_learning/Pendulum/BNN/training_inputs_active.csv'
df_training_inputs = pd.read_csv("training_inputs_active.csv")
training_inputs = df_training_inputs.values

file_path_training_outputs = '/home/hansm/active_learning/Pendulum/BNN/training_outputs_active.csv'
df_training_outputs = pd.read_csv("training_outputs_active.csv")
training_outputs = df_training_outputs.values

file_path_predictions = '/home/hansm/active_learning/Pendulum/BNN/predictions.csv'
df_predictions = pd.read_csv(file_path_predictions)
predictions = df_predictions.values


def check_settling_time(prediction, goal):
    # Ensure both arrays have at least 20 entries
    if len(prediction) < 20 or len(goal) < 20:
        raise ValueError("Both arrays should have at least 20 entries.")

    # Take the last 20 entries from both arrays
    last_20_prediction = prediction[-20:]
    last_20_goal = goal[-20:]

    # Calculate the maximum allowed average error (2%)
    max_average_error = 0.02

    # Check if the average error across all dimensions is within the 2% error band
    for val1, val2 in zip(last_20_prediction, last_20_goal):

        errors = np.abs((val1 - val2) / val2)  # Calculate the percentage error for each dimension
        avg_error = np.mean(errors)  # Calculate the average error across all dimensions
        if avg_error > max_average_error:
            return False  # Average error exceeds the 2% limit

    return True  # Average error is within the 2% limit for all 20 entries

# Define the BNN
def train_dropout_model(x_train, y_train, x_test, y_test):

  layer_size = [2] + [50] * 3 + [2]
  activation = "sigmoid"
  initializer = "Glorot uniform"
  regularization = ["l2", 1e-5]
  dropout_rate = 0.01
  net = dde.nn.FNN(
      layer_size,
      activation,
      initializer,
      regularization,
      dropout_rate
  )
  data = dde.data.DataSet(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
  BNN_model = dde.Model(data, net)
  BNN_uncertainty = dde.callbacks.DropoutUncertainty(period=5000)
  BNN_model.compile("adam", lr=0.001, metrics=["l2 relative error"])

  losshistory, train_state = BNN_model.train(iterations=10000, callbacks= [BNN_uncertainty])
  del losshistory
  del data
  del net
  return train_state, BNN_model

# Predict for all possible input pairs
train_state, model = train_dropout_model(training_inputs, training_outputs, train_inputs, train_outputs)
Y_std = train_state.y_std_test
# Find max uncertainty
sum_values = Y_std[:, 0] + Y_std[:, 1]
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
    print(convergence)
#Append RMSE and variance
prediction = model.predict(test_inputs)
# Calculate RMSE between the ground truth values and the predictions
RMSE = np.sqrt(mean_squared_error(test_outputs, prediction))
variance = np.sum(Y_std)


# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=['theta', 'omega'])
training_outputs_active_df = pd.DataFrame(training_outputs, columns=['theta', 'omega'])


# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('training_outputs_active.csv', index=False)

metrics_df = pd.read_csv('metrics.csv')
new_row_metrics = pd.DataFrame({'RMSE': [RMSE], 'Variance': [variance]})
metrics_df = pd.concat([metrics_df, new_row_metrics], ignore_index=True)
# Save the updated DataFrame back to the same CSV file
metrics_df.to_csv('metrics.csv', index=False)

predictions_df = pd.DataFrame(predictions, columns=['theta', 'omega'])
predictions_df.to_csv('predictions.csv', index=False)