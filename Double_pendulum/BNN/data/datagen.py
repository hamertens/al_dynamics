import numpy as np
from scipy.integrate import odeint
from itertools import product
import pandas as pd

# Define the double pendulum equations of motion
def double_pendulum(y, t):
    theta1, omega1, theta2, omega2 = y
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    g = 9.81

    # Equations of motion
    theta1_dot = omega1
    theta2_dot = omega2

    delta = theta2 - theta1
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) * np.cos(delta)
    omega1_dot = ((m2 * l1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
                   m2 * g * np.sin(theta2) * np.cos(delta) +
                   m2 * l2 * omega2 * omega2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta1)) / den1)

    den2 = (l2 / l1) * den1
    omega2_dot = ((-m2 * l2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
                   (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                   (m1 + m2) * l1 * omega1 * omega1 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta2)) / den2)

    return [theta1_dot, omega1_dot, theta2_dot, omega2_dot]

def datagen(discretize):

  theta1_range = np.linspace(0, 2*np.pi, discretize)
  theta2_range = np.linspace(0, 2*np.pi, discretize)
  omega1_range = np.linspace(-1, 1, discretize)
  omega2_range = np.linspace(-1, 1, discretize)
  combinations = np.array(list(product(theta1_range, theta2_range, omega1_range,
              omega2_range)))

  input_lst = []
  output_lst = []
  t = [0, 0.1]

  for row in combinations:
    theta1, theta2, omega1, omega2 = row
    sol = odeint(double_pendulum, row, t)
    input_lst.append(row)
    output_lst.append(sol[1])
  inputs = np.array(input_lst)
  outputs = np.array(output_lst)

  return inputs, outputs

train_inputs, train_outputs = datagen(11)
test_inputs, test_outputs = datagen(9)

# Define the column names
column_names = ["theta1", "theta2", "omega1", "omega2"]

# Create a Pandas DataFrame from the NumPy array
df = pd.DataFrame(train_inputs, columns=column_names)
# Define the file path where you want to save the CSV file
file_path = 'train_inputs.csv'  # Change the file name as needed
# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

# Create a Pandas DataFrame from the NumPy array
df = pd.DataFrame(test_inputs, columns=column_names)
# Define the file path where you want to save the CSV file
file_path = 'test_inputs.csv'  # Change the file name as needed
# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

# Create a Pandas DataFrame from the NumPy array
df = pd.DataFrame(train_outputs, columns=column_names)
# Define the file path where you want to save the CSV file
file_path = 'train_outputs.csv'  # Change the file name as needed
# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

# Create a Pandas DataFrame from the NumPy array
df = pd.DataFrame(test_outputs, columns=column_names)
# Define the file path where you want to save the CSV file
file_path = 'test_outputs.csv'  # Change the file name as needed
# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)