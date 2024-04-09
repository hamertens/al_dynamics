import sys
sys.path.append('../../') 
from Dynamical_System.dynamical_systems import Pendulum

# Initialize the Pendulum with default parameters
pendulum = Pendulum()

# Generate and save the training datasets
pendulum.save_data_to_csv(100, 70, train_prefix='train_', test_prefix='test_')