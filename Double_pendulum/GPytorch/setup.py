# Define specifications for the model and dynamic system
FOLDER_FILEPATH = "/home/hansm/active_learning/Double_pendulum/GPytorch/"
DATA_FILEPATH = "/home/hansm/active_learning/Double_pendulum/data/"
MAX_DIMENSIONALITY = 4
DATAFRAME_COLUMNS  = ["theta1", "theta2", "omega1", "omega2"]


#name of anaconda environment
env_name = "gpytorch" 

gpytorch_bash_filepath = "/home/hansm/active_learning/Double_pendulum/GPytorch/gpytorch.sh"

# Path to the init Python script
python_script_init="/home/hansm/active_learning/Double_pendulum/GPytorch/init.py"

# Path to the loop Python script
python_script_exec="/home/hansm/active_learning/Double_pendulum/GPytorch/exec.py"

# List of file paths
filepaths = [gpytorch_bash_filepath, python_script_init, python_script_exec, env_name]

if __name__ == "__main__":
    # Write the file paths to a temporary file, each on a new line
    with open("filepaths.txt", "w") as f:
        for path in filepaths:
            f.write(path + "\n")
