# Define specifications for the model and dynamic system
FOLDER_FILEPATH = "/home/hansm/active_learning/Lorenz/GP/"
DATA_FILEPATH = "/home/hansm/active_learning/Lorenz/data/"
MAX_DIMENSIONALITY = 3
DATAFRAME_COLUMNS  = ["x", "y", "z"]

#TRAINING_TYPE = 'not continuous'
#KERNEL = 'rbf'

model_system_type = 4


if model_system_type == 1:
    TRAINING_TYPE = 'not continuous'
    KERNEL = 'rbf'
elif model_system_type == 2:
    TRAINING_TYPE = 'continuous'
    KERNEL = 'rbf'
elif model_system_type == 3:
    TRAINING_TYPE = 'not continuous'
    KERNEL = 'matern'
elif model_system_type == 4:
    TRAINING_TYPE = 'continuous'
    KERNEL = 'matern'


#name of anaconda environment
env_name = "gp" 

gpytorch_bash_filepath = "/home/hansm/active_learning/Lorenz/GP/gp.sh"

# Path to the init Python script
python_script_init="/home/hansm/active_learning/Lorenz/GP/init.py"

# Path to the loop Python script
python_script_exec="/home/hansm/active_learning/Lorenz/GP/exec.py"

# List of file paths
filepaths = [gpytorch_bash_filepath, python_script_init, python_script_exec, env_name]

if __name__ == "__main__":
    # Write the file paths to a temporary file, each on a new line
    with open("filepaths.txt", "w") as f:
        for path in filepaths:
            f.write(path + "\n")
