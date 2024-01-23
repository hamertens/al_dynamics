# Define specifications for the model and dynamic system
FOLDER_FILEPATH = "/home/hansm/active_learning/Pendulum/Ensemble/"
DATA_FILEPATH = "/home/hansm/active_learning/Pendulum/data/"
INPUT_DIMENSIONALITY = 2
OUTPUT_DIMENSIONALITY = 2
DATAFRAME_COLUMNS  = ["theta", "omega"]


#name of anaconda environment
env_name = "ensemble" 

ensemble_bash_filepath = "/home/hansm/active_learning/Pendulum/Ensemble/ensemble.sh"

# Path to the init Python script
python_script_init="/home/hansm/active_learning/Pendulum/Ensemble/init.py"

# Path to the loop Python script
python_script_exec="/home/hansm/active_learning/Pendulum/Ensemble/exec.py"

# List of file paths
filepaths = [ensemble_bash_filepath, python_script_init, python_script_exec, env_name]

if __name__ == "__main__":
    # Write the file paths to a temporary file, each on a new line
    with open("filepaths.txt", "w") as f:
        for path in filepaths:
            f.write(path + "\n")