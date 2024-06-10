import pandas as pd
import random
import numpy as np
from systems import systems
import os
import argparse

parser = argparse.ArgumentParser(description="Import dynamical system and model specifications")
parser.add_argument("-m", "--model", type=str, required=True, help="Model type (required)")
parser.add_argument("-t", "--training", type=str, required=True, help="Type (required)")
parser.add_argument("-s", "--system", type=str, required=True, help="Type (required)")
parser.add_argument("-k", "--kernel", type=str, help="Kernel (optional)")

args = parser.parse_args()
model_type = args.model
training_type = args.training
system_type = args.system
kernel_type = args.kernel


# Load the System
if system_type == "lorenz":
    system = systems.Lorenz()
elif system_type == "pendulum":
    system = systems.Pendulum()
elif system_type == "double_pendulum":
    system = systems.DoublePendulum()
elif system_type == "two_tank_system":
    system = systems.TwoTankSystem()
elif system_type == "actuated_pendulum":
    system = systems.ActuatedPendulum()

# Load the System
sample_inputs = system.load_sample_inputs()
sample_outputs = system.load_sample_outputs()
test_inputs = system.load_test_inputs()
test_outputs = system.load_test_outputs()

# Use random.randint() to generate a random index
# Generate the first random index
random_index1 = random.randint(0, len(sample_inputs) - 1)
# Initialize the second random index to be the same as the first one
random_index2 = random_index1
# Keep generating a new random index until it's different from the first one
while random_index2 == random_index1:
    random_index2 = random.randint(0, len(sample_inputs) - 1)

training_inputs = np.array([sample_inputs[random_index1], sample_inputs[random_index2]])
training_outputs = np.array([sample_outputs[random_index1], sample_outputs[random_index2]])

# Creating DataFrames for training inputs and outputs
training_inputs_active_df = pd.DataFrame(training_inputs, columns=system.dataframe_columns_input)
training_outputs_active_df = pd.DataFrame(training_outputs, columns=system.dataframe_columns_output)

# create output_data folder if necessary
output_data_filepath = os.path.join(os.path.dirname(__file__), 'output_data')
if not os.path.exists(output_data_filepath):
    os.makedirs(output_data_filepath)


# Saving DataFrames to CSV files
training_inputs_active_df.to_csv('output_data/training_inputs_active.csv', index=False)
training_outputs_active_df.to_csv('output_data/training_outputs_active.csv', index=False)

metric_df = pd.DataFrame({'RMSE': [], 'Variance': [], 'Time': []})
metric_df.to_csv("output_data/metrics.csv", index=False)



if training_type == "continuous" and model_type == "mcdropout":
    if not os.path.exists(output_data_filepath + '/model'):
        os.makedirs(output_data_filepath + '/model')

if model_type == "mcdropout":
    from models import mcdropout
    hps = {'epochs': 6000, 'batch_size': 32, 'lr': 0.001, 'forward_passes': 5000, 'num_neurons': 60, 'num_layers': 2}
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


predictions = model.first_run()
predictions_df = pd.DataFrame(predictions, columns=system.dataframe_columns_output)
predictions_df.to_csv('output_data/predictions.csv', index=False)
