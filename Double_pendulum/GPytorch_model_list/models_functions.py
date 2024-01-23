import numpy as np
import torch
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def train_gp(training_inputs, training_outputs, training_iterations):
    training_inputs = torch.tensor(training_inputs, dtype=torch.float32)
    gaussian_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model_list = []
    for col_index in range(training_outputs.shape[1]):
        training_outputs_column = torch.tensor(training_outputs[:, col_index], dtype=torch.float32)
        exact_model = ExactGPModel(training_inputs, training_outputs_column, gaussian_likelihood)
        model_list.append(exact_model)
    likelihoods = [model.likelihood for model in model_list]
    likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)
    model = gpytorch.models.IndependentModelList(*model_list)

    mll = SumMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    return likelihood, model

def gp_eval(test_inputs, model, likelihood):
  test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
  # Set into eval mode
  model.eval()
  likelihood.eval()


  # Make predictions (use the same test points)
  with torch.no_grad(), gpytorch.settings.fast_pred_var():
      
      # This contains predictions for both outcomes as a list
      predictions = likelihood(*model(test_inputs, test_inputs, test_inputs, test_inputs))
      predictions_lst = [prediction.mean.numpy() for prediction in predictions]
      final_prediction = np.column_stack((predictions_lst))
      variance_lst = []
      for prediction in predictions:
        lower, upper = prediction.confidence_region()
        variance = upper.numpy() - lower.numpy()
        variance_lst.append(variance)
  return final_prediction, np.column_stack(variance_lst)


def check_settling_time(prediction, goal):
    # Ensure both arrays have at least 20 entries
    if len(prediction) < 20 or len(goal) < 20:
        raise ValueError("Both arrays should have at least 20 entries.")

    # Take the last 20 entries from both arrays
    last_20_prediction = prediction[-20:]
    last_20_goal = goal[-20:]

    # Calculate the maximum allowed average error (2%)
    max_average_error = 0.02

    # Check if the average error across all dimensions is within the 2% error band
    for val1, val2 in zip(last_20_prediction, last_20_goal):

        errors = np.abs((val1 - val2) / val2)  # Calculate the percentage error for each dimension
        avg_error = np.mean(errors)  # Calculate the average error across all dimensions
        if avg_error > max_average_error:
            return False  # Average error exceeds the 2% limit

    return True  # Average error is within the 2% limit for all 20 entries