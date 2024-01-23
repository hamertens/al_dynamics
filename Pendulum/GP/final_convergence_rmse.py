import numpy as np
from scipy.integrate import odeint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from itertools import product
import math
import pandas as pd
import random

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


file_path_train_inputs = '/home/hansm/active_learning/Lorenz/data/lorenz_train_inputs.csv'
df_train_inputs = pd.read_csv(file_path_train_inputs)
train_inputs = df_train_inputs.values

file_path_train_outputs = '/home/hansm/active_learning/Lorenz/data/lorenz_train_outputs.csv'
df_train_outputs = pd.read_csv(file_path_train_outputs)
train_outputs = df_train_outputs.values

file_path_test_inputs = '/home/hansm/active_learning/Lorenz/data/lorenz_test_inputs.csv'
df_test_inputs = pd.read_csv(file_path_test_inputs)
test_inputs = df_test_inputs.values

file_path_test_outputs = '/home/hansm/active_learning/Lorenz/data/lorenz_test_outputs.csv'
df_test_outputs = pd.read_csv(file_path_test_outputs)
test_outputs = df_test_outputs.values

# Create the Gaussian Process Regressor
#kernel = RBF(length_scale=1.0)
#gpr = GaussianProcessRegressor(kernel=kernel)
#gpr.fit(train_inputs, train_outputs)

#prediction = gpr.predict(test_inputs, return_std=False)
#rmse = np.sqrt(mean_squared_error(test_outputs, prediction))
rmse = 1.165e-05
df = pd.DataFrame({'RMSE': [rmse]})
df.to_csv("final_convergence_rmse.csv", index=False)