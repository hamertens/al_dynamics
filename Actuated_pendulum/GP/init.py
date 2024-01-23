import pandas as pd
import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

file_path_train_inputs = '/home/hansm/active_learning/Actuated_pendulum/GP/data/train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = '/home/hansm/active_learning/Actuated_pendulum/GP/data/train_outputs.csv'
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

metric_df = pd.DataFrame({'RMSE': [], 'Variance': [], "Train_RMSE": []})
metric_df.to_csv("metrics.csv", index=False)


# Create the Gaussian Process Regressor
kernel = RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(training_inputs, training_outputs)
# Add initial predictions
Y_pred, _ = gpr.predict(training_inputs, return_std=True)
predictions = Y_pred
predictions_df = pd.DataFrame(predictions, columns=['theta', 'omega'])
predictions_df.to_csv('predictions.csv', index=False)