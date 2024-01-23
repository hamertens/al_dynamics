import numpy as np
import deepxde as dde
from sklearn.metrics import mean_squared_error
import pandas as pd

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


file_path_train_inputs = '/home/hansm/active_learning/Double_pendulum/BNN/data/train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = '/home/hansm/active_learning/Double_pendulum/BNN/data/train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

file_path_test_inputs = '/home/hansm/active_learning/Double_pendulum/BNN/data/test_inputs.csv'
df_test_inputs = pd.read_csv(file_path_test_inputs)
test_inputs = df_test_inputs.values

file_path_test_outputs = '/home/hansm/active_learning/Double_pendulum/BNN/data/test_outputs.csv'
df_test_outputs = pd.read_csv(file_path_test_outputs)
test_outputs = df_test_outputs.values

# Define the BNN
def train_dropout_model(x_train, y_train, x_test, y_test):

  layer_size = [4] + [50] * 3 + [4]
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
  BNN_model.compile("adam", lr=0.003, metrics=["l2 relative error"]
                    #, decay = ("cosine", 1000, 0.1)
                    )

  losshistory, train_state = BNN_model.train(iterations=5000, callbacks= [BNN_uncertainty])
  del losshistory
  del data
  del net
  return train_state, BNN_model

train_state, model = train_dropout_model(train_inputs, train_outputs, test_inputs, test_outputs)

pred = model.predict(test_inputs)
rmse_predict = np.sqrt(mean_squared_error(pred, test_outputs))
print(f"rmse using model.predict = {rmse_predict}")

train_state_pred = train_state.y_pred_test
rmse_train_state = np.sqrt(mean_squared_error(train_state_pred, test_outputs))
print(f"rmse using train state = {rmse_train_state}")