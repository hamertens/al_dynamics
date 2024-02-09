import numpy as np
from scipy.integrate import odeint
import pandas as pd
from itertools import product


# Define the ODE system
def tank_system(state, t, q):

    A1 = 1
    A2 = 1
    g = 9.81
    h1, h2 = state
    q1 = A1 * np.sqrt(2 * g * (h1-h2))
    q2 = A2 * np.sqrt(2 * g * h2)

    dh1_dt = (q - q1) / A1
    dh2_dt = (q1 - q2) / A2

    return [dh1_dt, dh2_dt]

# Training data
t_stepsize = 0.1 # stepsize = 0.1 seconds
t_range = np.array([0, t_stepsize])

def datagen(disc):
  # Create theta_range and omega_range arrays
  h1_range = np.linspace(20, 40, disc)
  h2_range = np.linspace(0, 20, disc)
  q_range = np.linspace(1, 20, int(disc/2))

  # Create a grid of coordinates
  X_test = np.array(list(product(h1_range, h2_range, q_range)))
  ground_truth = []
  for i in range(len(X_test)):
    y0 = [X_test[i, 0], X_test[i, 1]]
    sol = odeint(tank_system, y0, t_range, args=(X_test[i, 2],))
    ground_truth.append([sol[1, 0], sol[1, 1]])

  input_df = pd.DataFrame(X_test, columns=['h1', 'h2', "q"])
  output_df = pd.DataFrame(ground_truth, columns=['h1', 'h2'])
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