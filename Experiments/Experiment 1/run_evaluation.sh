#!/bin/bash

# Enable debugging
set -x



# Get the directory of the current script
script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# Define project path (two levels up from the script directory)
project_path=$(realpath "$script_dir/../../")

# Define config file path
config_path="$project_path/config/config1.ini"

# Define results file path
results_file="$script_dir/results_experiment_1.h5"


# Export stuff
export CONFIG_FILE=$config_path
export RESULTS_FILE="$results_file"


# Print out the paths for debugging
echo "Script directory: $script_dir"
echo "Project path: $project_path"
echo "Config path: $config_path"

# Check if config file exists
if [ ! -f "$config_path" ]; then
    echo "Config file not found: $config_path"
    exit 1
fi




# Loop through TAB values and update config file, then run the Python script
for tab in 0 0.2 0.4 0.6 0.8 1; do
    echo "Updating TAB to $tab"

    
    # Update the TAB value in the config file
    crudini --set "$config_path" synthetic_data TAB "$tab"
    
    if [ $? -ne 0 ]; then
        echo "Failed to update config file"
        exit 1
    fi

    #generate data with the new tab parameter
    python "$project_path/synthetic_data_generation/synthetic_data_faster.py"

    if [ $? -ne 0 ]; then
        echo "Failed to generate synthetic data"
        exit 1
    fi

    sleep 1

    # Define the evaluation script path
    file="$project_path/evaluation/evaluation.py"

    # Check if the Python script exists
    if [ ! -f "$file" ]; then
        echo "Python script not found: $file"
        exit 1
    fi

    echo "Running Python script: $file with CONFIG_FILE: $CONFIG_FILE and RESULTS_FILE: $RESULTS_FILE"
    python "$file"

    if [ $? -ne 0 ]; then
        echo "Python script failed"
        exit 1
    fi
done



# Disable debugging
set +x
