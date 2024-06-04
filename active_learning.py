import numpy as np
from systems import systems
from utils import functions
import sys 
import time

start_time = time.time()

# Load the System
system = systems.Lorenz()
sample_inputs = system.load_sample_inputs()
sample_outputs = system.load_sample_outputs()
test_inputs = system.load_test_inputs()
test_outputs = system.load_test_outputs()

# Read the active learning training inputs, training outputs and predictions from the CSV files
training_inputs, training_outputs, predictions = functions.read_csv_files()

# Initialize the model
training_type = "continuous"
#model_type = "mcdropout"
model_type = "ensemble"
#model_type = "gp"

if model_type == "mcdropout":
    from models import mcdropout
    hps = {'epochs': 6000, 'batch_size': 32, 'lr': 0.001, 'num_models': 5, 'num_neurons': 60, 'num_layers': 2}
    model = mcdropout.MCDropout(training_type, sample_inputs, sample_outputs, test_inputs, test_outputs, training_inputs, training_outputs, hps)
elif model_type == "ensemble":
    from models import ensemble
    hps = {'epochs': 500, 'batch_size': 32, 'lr': 0.001, 'num_models': 5, 'num_neurons': 60, 'num_layers': 2}
    model = ensemble.Ensemble(training_type, sample_inputs, sample_outputs, test_inputs, test_outputs, training_inputs, training_outputs, hps)
elif model_type == "gp":
    from models import gp
    kernel_type = "rbf"
    #kernel_type = "matern"
    model = gp.GP(training_type, kernel_type, sample_inputs, sample_outputs, test_inputs, test_outputs, training_inputs, training_outputs)

max_index, new_prediction, variance, RMSE = model.active_learning()


# Append new prediction, training input and training output
predictions = np.vstack((predictions, new_prediction))
training_inputs = np.vstack((training_inputs, sample_inputs[max_index]))
training_outputs = np.vstack((training_outputs, sample_outputs[max_index]))

end_time = time.time()
elapsed_time = end_time - start_time

# Write the updated training inputs, training outputs, predictions and metrics to CSV files
functions.write_csv_files(training_inputs, training_outputs, system.dataframe_columns_input, system.dataframe_columns_output, RMSE, variance, predictions, elapsed_time)

# check convergence
if len(predictions) >= 20:
    convergence = functions.check_settling_time(predictions, training_outputs)
    if convergence == True:
        sys.exit(1)