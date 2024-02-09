#!/bin/bash


# Read the file paths from the temporary file into an array
mapfile -t filepaths < filepaths.txt

source activate ${filepaths[3]}


# Start tracking time
start_time=$(date +%s)

# Execute the init Python script
python ${filepaths[1]}



# Loop x times

for ((i=1; i<=1000; i++))
do
    python ${filepaths[2]}

    if [ $? -ne 0 ]; then
      break
    fi
done

# Calculate the execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Pass the execution time to the Python script
python3 track_time.py "$execution_time"
