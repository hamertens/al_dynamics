import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
        )
# rank = 1 means that you are approximating the covariance matrix with a single principal component (rank-1 approximation)
# a higher rank is more computationally expensive but might be more accurate
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
def train_gp(training_inputs, training_outputs, training_iterations):

  train_x = torch.tensor(training_inputs, dtype=torch.float32)
  train_y = torch.tensor(training_outputs, dtype=torch.float32)

  likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
  model = MultitaskGPModel(train_x, train_y, likelihood)
  # Find optimal model hyperparameters
  model.train()
  likelihood.train()

  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

  # "Loss" for GPs - the marginal log likelihood
  mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

  for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
  
  return likelihood, model#, optimizer, output, loss

def gp_eval(eval_inputs, model, likelihood):
  test_x = torch.tensor(eval_inputs, dtype=torch.float32)
  # Set into eval mode
  model.eval()
  likelihood.eval()

  # Make predictions
  with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    variance = upper.numpy() - lower.numpy()
  return mean.numpy(), variance