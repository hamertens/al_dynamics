import sys
sys.path.append('../../') 
from Dynamical_System.dynamical_systems import TwoTankSystem

# Initialize the TwoTankSystem with default parameters
tts = TwoTankSystem()

# Generate and save the training and test datasets
tts.save_data_to_csv(30, 25, train_prefix='train_', test_prefix='test_')
