import numpy as np
from scipy.integrate import odeint
import pandas as pd
from itertools import product

class ActuatedPendulum:
    def __init__(self, g=9.81, L=10.0, m=1.0, b=0.11, t_stepsize=0.1):
        self.g = g  # acceleration due to gravity
        self.L = L  # length of the pendulum
        self.m = m  # mass of the pendulum
        self.b = b  # damping coefficient
        self.t_stepsize = t_stepsize

    def actuated_pendulum(self, state, t, torque):
        theta, omega = state
        dtheta_dt = omega
        domega_dt = (-self.g / self.L) * np.sin(theta) + (torque / (self.m * self.L**2)) - self.b * omega
        return [dtheta_dt, domega_dt]

    def datagen(self, disc):
        theta_range = np.linspace(-np.pi/2, np.pi/2, num=disc)
        omega_range = np.linspace(-1, 1, num=disc)
        torque_range = np.linspace(-1, 1, num=disc)
        X_test = np.array(list(product(theta_range, omega_range, torque_range)))
        ground_truth = []
        for i in range(len(X_test)):
            t = np.linspace(0, self.t_stepsize, num=2)
            y0 = X_test[i, :2]
            sol = odeint(self.actuated_pendulum, y0, t, args=(X_test[i, 2],))
            ground_truth.append(sol[-1])

        ground_truth = np.array(ground_truth)
        input_df = pd.DataFrame(X_test, columns=['theta', 'omega', "torque"])
        output_df = pd.DataFrame(ground_truth, columns=['theta', 'omega'])
        return input_df, output_df

    def save_data_to_csv(self, disc_train, disc_test, train_prefix='', test_prefix=''):
        # Generate and save training data
        train_input_df, train_output_df = self.datagen(disc_train)
        train_input_df.to_csv(train_prefix + 'train_inputs.csv', index=False)
        train_output_df.to_csv(train_prefix + 'train_outputs.csv', index=False)

        # Generate and save test data
        test_input_df, test_output_df = self.datagen(disc_test)
        test_input_df.to_csv(test_prefix + 'test_inputs.csv', index=False)
        test_output_df.to_csv(test_prefix + 'test_outputs.csv', index=False)
        
    def simulate(self, initial_conditions, time):
        # Simulate the actuated pendulum dynamics over the specified time
        # This function returns the system state at each time step
        return states
        

        
class LorenzSystem:
    def __init__(self, sigma=10, rho=28, beta=8/3):
        # Initialize the Lorenz system with its parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def lorenz_system(self, state, t):
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]

    def simulate(self, initial_conditions, time):
        # Simulate the Lorenz system dynamics over the specified time
        # This function returns the system state at each time step
        return states

    def dataset_gen(self, initial_conditions, t):
        solution = self.simulate(initial_conditions, t)
        x_trajectory, y_trajectory, z_trajectory = solution[:, 0], solution[:, 1], solution[:, 2]
        input_df = pd.DataFrame({'x': x_trajectory[:-1], 'y': y_trajectory[:-1], 'z': z_trajectory[:-1]})
        output_df = pd.DataFrame({'x': x_trajectory[1:], 'y': y_trajectory[1:], 'z': z_trajectory[1:]})
        return input_df, output_df

    def save_data_to_csv(self, initial_conditions, t, prefix=''):
        input_df, output_df = self.dataset_gen(initial_conditions, t)
        input_df.to_csv(prefix + 'inputs.csv', index=False)
        output_df.to_csv(prefix + 'outputs.csv', index=False)
        
class Pendulum:
    def __init__(self, length=1.0, mass=1.0, damping=0.1, gravity=1.0):
        self.length = length
        self.mass = mass
        self.damping = damping
        self.gravity = gravity

    def damped_pendulum(self, state, t):
        theta, omega = state
        dydt = [omega, -self.damping * omega - self.gravity * np.sin(theta)]
        return dydt

    def datagen(self, disc, t_stepsize=0.1):
        theta_range = np.linspace(-np.pi, np.pi, disc)
        omega_range = np.linspace(-1, 1, disc)
        theta_grid, omega_grid = np.meshgrid(theta_range, omega_range)
        points = np.column_stack((theta_grid.flatten(), omega_grid.flatten()))

        t_range = np.array([0, t_stepsize])
        solutions = np.zeros((disc**2, 2))
        for i in range(len(points)):
            sol = odeint(self.damped_pendulum, points[i], t_range)[-1]
            solutions[i] = sol

        input_df = pd.DataFrame(points, columns=['theta', 'omega'])
        output_df = pd.DataFrame(solutions, columns=['theta', 'omega'])
        return input_df, output_df

    def save_data_to_csv(self, disc_train, disc_test, train_prefix='', test_prefix=''):
        train_input_df, train_output_df = self.datagen(disc_train)
        train_input_df.to_csv(train_prefix + 'inputs.csv', index=False)
        train_output_df.to_csv(train_prefix + 'outputs.csv', index=False)

        test_input_df, test_output_df = self.datagen(disc_test)
        test_input_df.to_csv(test_prefix + 'inputs.csv', index=False)
        test_output_df.to_csv(test_prefix + 'outputs.csv', index=False)
        
    def simulate(self, initial_conditions, time):
        # Simulate the pendulum dynamics over the specified time
        # This function returns the pendulum state at each time step
        return states
    
class TwoTankSystem:
    def __init__(self, A1=1, A2=1, g=9.81):
        self.A1 = A1
        self.A2 = A2
        self.g = g

    def tank_system(self, state, t, q):
        h1, h2 = state
        q1 = self.A1 * np.sqrt(2 * self.g * (h1-h2))
        q2 = self.A2 * np.sqrt(2 * self.g * h2)

        dh1_dt = (q - q1) / self.A1
        dh2_dt = (q1 - q2) / self.A2

        return [dh1_dt, dh2_dt]

    def datagen(self, disc, t_stepsize=0.1):
        h1_range = np.linspace(20, 40, disc)
        h2_range = np.linspace(0, 20, disc)
        q_range = np.linspace(1, 20, int(disc/2))

        X_test = np.array(list(product(h1_range, h2_range, q_range)))
        ground_truth = []
        t_range = np.array([0, t_stepsize])

        for i in range(len(X_test)):
            y0 = [X_test[i, 0], X_test[i, 1]]
            sol = odeint(self.tank_system, y0, t_range, args=(X_test[i, 2],))
            ground_truth.append(sol[-1])

        input_df = pd.DataFrame(X_test, columns=['h1', 'h2', "q"])
        output_df = pd.DataFrame(ground_truth, columns=['h1', 'h2'])
        return input_df, output_df

    def save_data_to_csv(self, disc_train, disc_test, train_prefix='', test_prefix=''):
        train_input_df, train_output_df = self.datagen(disc_train)
        train_input_df.to_csv(train_prefix + 'inputs.csv', index=False)
        train_output_df.to_csv(train_prefix + 'outputs.csv', index=False)

        test_input_df, test_output_df = self.datagen(disc_test)
        test_input_df.to_csv(test_prefix + 'inputs.csv', index=False)
        test_output_df.to_csv(test_prefix + 'outputs.csv', index=False)
        
    def simulate(self, initial_conditions, time):
        # Simulate the two-tank system dynamics over the specified time
        # This function returns the tank levels at each time step
        return states



