# Define specifications for the model and dynamic system
FOLDER_FILEPATH = "/home/hansm/active_learning/Lorenz/Ensemble/"
DATA_FILEPATH = "/home/hansm/active_learning/Lorenz/data/"
INPUT_DIMENSIONALITY = 3
OUTPUT_DIMENSIONALITY = 3
DATAFRAME_COLUMNS  = ["x", "y", "z"]

#name of anaconda environment
env_name = "ensemble" 

ensemble_bash_filepath = "/home/hansm/active_learning/Lorenz/Ensemble/ensemble.sh"

# Path to the init Python script
python_script_init="/home/hansm/active_learning/Lorenz/Ensemble/init.py"

# Path to the loop Python script
python_script_exec="/home/hansm/active_learning/Lorenz/Ensemble/exec.py"

# List of file paths
filepaths = [ensemble_bash_filepath, python_script_init, python_script_exec, env_name]

if __name__ == "__main__":
    # Write the file paths to a temporary file, each on a new line
    with open("filepaths.txt", "w") as f:
        for path in filepaths:
            f.write(path + "\n")