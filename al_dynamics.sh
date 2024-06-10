#!/bin/bash

env_name="gp"

source activate $env_name


# Start tracking time
start_time=$(date +%s)

# specify variable for kernel_type, model_type, training_type and system and then use them as arguments for the python scripts
kernel_type="rbf"
model_type="gp"
training_type="continuous"
system="lorenz"
# Execute the init Python script
python init.py --kernel $kernel_type --model $model_type --training $training_type --system $system


# Loop x times

for ((i=1; i<=100; i++))
do
    # Execute the exec Python script
    python3 active_learning.py --kernel $kernel_type --model $model_type --training $training_type --system $system

    if [ $? -ne 0 ]; then
      break
    fi
done

# Calculate the execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Pass the execution time to the Python script
python3 utils/track_time.py "$execution_time"
