import sys
sys.path.append('../../') 
from Dynamical_System.dynamical_systems import ActuatedPendulum

# Initialize the ActuatedPendulum with default parameters
actuated_pendulum = ActuatedPendulum()

# Generate and save the test datasets
actuated_pendulum.save_data_to_csv(30, 25, train_prefix='train_', test_prefix='test_')