Implementation for Paper "Comparative Analysis of Uncertainty Quantification Models in Active
Learning for Efficient System Identification of Dynamical Systems"

You can recreate our experiments by executing the bash script "al_dynamics.py" which runs the active learning loop.

The bash script executes an init.py script that creates a metrics.csv, predictions.csv, training_inputs_active.csv and training_outputs_active.csv
The bash script then executes active_learning.py for a certain number of iterations. In each iteration the prediction of the model is saved to predictions.csv, in metrics.csv the current error (RMSE), run-time and variance of the model is saved. 
In training_inputs_active.csv and training_outputs_active.csv the training datapoints are saved that are chosen from the active learning algorithm.

The ML models and dynamical systems are implemented as classes and can therefore be substituted by uncommenting one line in init.py and active_learning.py.

3 different anaconda environments are used for the different ML models. You can find the yml files in this repo to install the environments and run the code.





