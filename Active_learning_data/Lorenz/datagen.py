import sys
sys.path.append('../../') 
from Dynamical_System.dynamical_systems import LorenzSystem
import numpy as np

# Initialize the LorenzSystem with default parameters
lorenz = LorenzSystem()

# Generate and save the training datasets
lorenz.save_data_to_csv([1.0, 0.0, 0.0], np.linspace(0, 1000, 10000), prefix='train_')

# Generate and save the test datasets
lorenz.save_data_to_csv([10, 20.0, 5.0], np.linspace(0, 1000, 10000), prefix='test_')