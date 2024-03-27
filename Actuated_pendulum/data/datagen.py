import numpy as np
from scipy.integrate import odeint
import pandas as pd
from itertools import product

def actuated_pendulum(state, t, torque):
    theta, omega = state
    g = 9.81  # acceleration due to gravity
    L = 10.0   # length of the pendulum
    m = 1.0   # mass of the pendulum
    b = 0.11  # damping coefficient

    # Compute the derivatives of theta and omega
    dtheta_dt = omega
    domega_dt = (-g / L) * np.sin(theta) + (torque / (m * L**2)) - b * omega

    return [dtheta_dt, domega_dt]

# Training data
t_stepsize = 0.1 # stepsize = 0.1 seconds
t_range = np.array([0, t_stepsize])

def datagen(disc):
  # Generate test dataset
  # Generate theta, omega, and torque values using np.linspace
  theta_range = np.linspace(-np.pi/2, np.pi/2, num=disc)
  omega_range = np.linspace(-1, 1, num=disc)
  torque_range = np.linspace(-1, 1, num=disc)

  # Generate all possible combinations of theta, omega, and torque
  X_test = np.array(list(product(theta_range, omega_range, torque_range)))

  # Calculate the ground truth values for the test dataset
  ground_truth = []
  for i in range(len(X_test)):
      t = np.linspace(0, 0.1, num=2)
      y0 = [X_test[i, 0], X_test[i, 1]]
      sol = odeint(actuated_pendulum, y0, t, args=(X_test[i, 2],))
      ground_truth.append([sol[1, 0], sol[1, 1]])

  # Convert the ground truth values to NumPy array
  ground_truth = np.array(ground_truth)

  
  input_df = pd.DataFrame(X_test, columns=['theta', 'omega', "torque"])
  output_df = pd.DataFrame(ground_truth, columns=['theta', 'omega'])
  return input_df, output_df

train_input_df, train_output_df = datagen(30)

# Save the DataFrame to a CSV file
train_input_df.to_csv('train_inputs.csv', index=False)
train_output_df.to_csv('train_outputs.csv', index=False)

# Test data
test_input_df, test_output_df = datagen(25)
# Save the DataFrame to a CSV file
test_input_df.to_csv('test_inputs.csv', index=False)
test_output_df.to_csv('test_outputs.csv', index=False)