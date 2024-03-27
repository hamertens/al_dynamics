import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
gp_dir = os.path.join(base_dir, 'GP')
data_dir = os.path.join(base_dir, 'data')

# Define specifications for the model and dynamic system
FOLDER_FILEPATH = gp_dir + os.sep
DATA_FILEPATH = data_dir + os.sep
MAX_DIMENSIONALITY = 3
DATAFRAME_COLUMNS_INPUT  = ['h1', 'h2', "q"]
DATAFRAME_COLUMNS_OUTPUT  = ['h1', 'h2']

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

gpytorch_bash_filepath = os.path.join(gp_dir, "gp.sh")

# Path to the init Python script
python_script_init = os.path.join(gp_dir, "init.py")

# Path to the loop Python script
python_script_exec = os.path.join(gp_dir, "exec.py")

# List of file paths
filepaths = [gpytorch_bash_filepath, python_script_init, python_script_exec, env_name]

if __name__ == "__main__":
    # Write the file paths to a temporary file, each on a new line
    with open("filepaths.txt", "w") as f:
        for path in filepaths:
            f.write(path + "\n")
