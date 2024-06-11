Implementation for Paper "Comparative Analysis of Uncertainty Quantification Models in Active
Learning for Efficient System Identification of Dynamical Systems"

All systems and models are implemented as modular classes. You can change the variables in al_dynamics.sh to run the code with the specific dynamical system, ml model, training type (and kernel type). al_dynamics.sh will execute init.py once and loop over active_learning.py. You can also run the python scripts like this:

'''
python active_learning.py --model gp --system lorenz --training continuous --kernel rbf
'''

The bash script executes an init.py script that creates a metrics.csv, predictions.csv, training_inputs_active.csv and training_outputs_active.csv
The bash script then executes active_learning.py for a certain number of iterations. In each iteration the prediction of the model is saved to predictions.csv, in metrics.csv the current error (RMSE), run-time and variance of the model is saved. 
In training_inputs_active.csv and training_outputs_active.csv the training datapoints are saved that are chosen from the active learning algorithm.

3 different anaconda environments are used for the different ML models. You can find the yml files in this repo to install the environments and run the code.





