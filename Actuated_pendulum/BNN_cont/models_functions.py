import numpy as np
import os
os.environ['DDE_BACKEND'] = 'tensorflow'
import deepxde as dde
from setup import FOLDER_FILEPATH, INPUT_DIMENSIONALITY, OUTPUT_DIMENSIONALITY

# Define the BNN
def train_dropout_model(x_train, y_train, x_test, y_test, init_script=False, input_dim = INPUT_DIMENSIONALITY, output_dim = OUTPUT_DIMENSIONALITY):

  layer_size = [input_dim] + [64] * 2 + [output_dim]
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
  save_path = FOLDER_FILEPATH + 'output_data/model/'
  if init_script == True:
      load_path = None
  else:
      load_path = FOLDER_FILEPATH + 'output_data/model/-5000.ckpt.index'    
  losshistory, train_state = BNN_model.train(iterations=3000, callbacks= [BNN_uncertainty], model_save_path=save_path 
                                             ,model_restore_path=load_path)
  del losshistory
  del data
  del net
  return train_state, BNN_model


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
        # Skip the current entry if any of the values in val2 are 0
        if np.any(np.isclose(val2, 0)):
            continue
        errors = np.abs((val1 - val2) / val2)  # Calculate the percentage error for each dimension
        avg_error = np.mean(errors)  # Calculate the average error across all dimensions
        if avg_error > max_average_error:
            return False  # Average error exceeds the 2% limit

    return True  # Average error is within the 2% limit for all 20 entries

def check_divergence(metrics_df):
    length = len(metrics_df)
    if length < 500:
        return False
    rmse_array = metrics_df["RMSE"].values
    # compute the average rmse of the last 10 values in rmse_array
    average_rmse_current = np.mean(rmse_array[-10:])
    average_rmse_old = np.mean(rmse_array[-210:-200])
    if average_rmse_current*1.05 > average_rmse_old:
        return True
    return False