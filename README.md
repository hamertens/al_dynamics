Active Learning for dynamical systems repository

Execute bash scripts in Dynamical_System/ML_model/bash_script.sh

The bash script executes an init.py script that creates a metrics.csv, predictions.csv, training_inputs_active.csv and training_outputs_active.csv
The bash script then executes exec.py for a certain number of iterations. In each iteration the prediction of the model is saved to predictions.csv, in metrics.csv the current error (RMSE) and variance of the model is saved. 
In training_inputs_active.csv and training_outputs_active.csv the training datapoints are saved that are chosen from the active learning algorithm.

When final_iterations are reached track_time.py is executed to track the total execution time.
The models_functions.py file contains the classes for the ML models, function to train and evaluate them and check_settling_time functon to check if the model has converged.

In the setup.py file, the user can specify all the different parameters that are used by the other python files. Once all specifications are added, the setup.py file needs to executed once to create the filepaths.txt file.

Currently, I'm using 4 different anaconda environments for the individual ML models. You can find the yml files in this repo to install the environments and run the code.





