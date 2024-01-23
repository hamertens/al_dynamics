import pandas as pd
import random
import numpy as np
import deepxde as dde

file_path_train_inputs = '/home/hansm/active_learning/Actuated_pendulum/BNN/data/train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = '/home/hansm/active_learning/Actuated_pendulum/BNN/data/train_outputs.csv'
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
training_inputs_active_df = pd.DataFrame(training_inputs, columns=['theta', 'omega', 'torque'])
training_outputs_active_df = pd.DataFrame(training_outputs, columns=['theta', 'omega'])

# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('training_outputs_active.csv', index=False)

metric_df = pd.DataFrame({'RMSE': [], 'Variance': []})
metric_df.to_csv("metrics.csv", index=False)

# Define the BNN
def train_dropout_model(x_train, y_train, x_test, y_test):

  layer_size = [3] + [50] * 3 + [2]
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
  BNN_uncertainty = dde.callbacks.DropoutUncertainty(period=1000)
  BNN_model.compile("adam", lr=0.001, metrics=["l2 relative error"])

  losshistory, train_state = BNN_model.train(iterations=1000, callbacks= [BNN_uncertainty])
  del losshistory
  del data
  del net
  return train_state, BNN_model


train_state, model = train_dropout_model(training_inputs, training_outputs, training_inputs, training_outputs)
# Add initial predictions
predictions = train_state.y_pred_test
predictions_df = pd.DataFrame(predictions, columns=['theta', 'omega'])
predictions_df.to_csv('predictions.csv', index=False)