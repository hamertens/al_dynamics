import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
ensemble_cont_dir = os.path.join(base_dir, 'Ensemble_continuous')
data_dir = os.path.join(base_dir, 'data')
# Define specifications for the model and dynamic system
FOLDER_FILEPATH = ensemble_cont_dir + os.sep
DATA_FILEPATH = data_dir + os.sep
INPUT_DIMENSIONALITY = 3
OUTPUT_DIMENSIONALITY = 2
DATAFRAME_COLUMNS_INPUT  = ['h1', 'h2', "q"]
DATAFRAME_COLUMNS_OUTPUT  = ['h1', 'h2']

#name of anaconda environment
env_name = "ensemble_cont" 

ensemble_bash_filepath = os.path.join(ensemble_cont_dir, "ensemble.sh")

# Path to the init Python script
python_script_init = os.path.join(ensemble_cont_dir, "init.py")

# Path to the loop Python script
python_script_exec = os.path.join(ensemble_cont_dir, "exec.py")

# List of file paths
filepaths = [ensemble_bash_filepath, python_script_init, python_script_exec, env_name]

if __name__ == "__main__":
    # Write the file paths to a temporary file, each on a new line
    with open("filepaths.txt", "w") as f:
        for path in filepaths:
            f.write(path + "\n")