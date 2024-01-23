#!/bin/bash


source activate bnn


# Start tracking time
start_time=$(date +%s)

# Path to the init Python script
python_script_init="/home/hansm/active_learning/Lorenz/BNN/init.py"
python3 $python_script_init

# Path to the loop Python script
python_script_exec="/home/hansm/active_learning/Lorenz/BNN/exec.py"

# Loop x times

for ((i=1; i<=2000; i++))
do
    output=$(python $python_script_exec)

    
    if [ "$output" == "True" ]; then
        break
    fi
done

# Calculate the execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Pass the execution time to the Python script
python3 track_time.py "$execution_time"
