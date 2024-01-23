import numpy as np
from scipy.integrate import odeint
import pandas as pd

# Define the ODEs
def damped_pendulum(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b * omega - c * np.sin(theta)]
    return dydt

# Define the parameters
b = 0.1  # damping coefficient
c = 1.0  # gravitational constant

# Training data
t_stepsize = 0.1 # stepsize = 0.1 seconds
t_range = np.array([0, t_stepsize])

def datagen(disc):
  # Create theta_range and omega_range arrays
  theta_range = np.linspace(-np.pi, np.pi, disc)
  omega_range = np.linspace(-1, 1, disc)

  # Create a grid of coordinates
  theta_grid, omega_grid = np.meshgrid(theta_range, omega_range)

  # Flatten the grid to obtain the list of all possible combinations
  points = np.column_stack((theta_grid.flatten(), omega_grid.flatten()))

  solutions = np.zeros((disc**2, 2))
  for i in range(len(points)):
    sol = odeint(damped_pendulum, points[i], t_range, args=(b, c))[1]
    solutions[i] = sol

  input_df = pd.DataFrame(points, columns=['theta', 'omega'])
  output_df = pd.DataFrame(solutions, columns=['theta', 'omega'])
  return input_df, output_df

train_input_df, train_output_df = datagen(100)

# Save the DataFrame to a CSV file
train_input_df.to_csv('train_inputs.csv', index=False)
train_output_df.to_csv('train_outputs.csv', index=False)

# Test data
test_input_df, test_output_df = datagen(70)
# Save the DataFrame to a CSV file
test_input_df.to_csv('test_inputs.csv', index=False)
test_output_df.to_csv('test_outputs.csv', index=False)