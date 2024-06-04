#!/bin/bash

# Enable debugging
set -x



# Get the directory of the current script
script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# Define project path (two levels up from the script directory)
project_path=$(realpath "$script_dir/../../")

# Define config file path
config_path="$project_path/config/config2.ini"

# Define results file path
results_file="$script_dir/results_experiment_2.h5"

#Define the datasets paths
patients_path="$project_path/synthetic_data_generation/patients.csv"
organs_path="$project_path/synthetic_data_generation/organs.csv"
outcomes_path="$project_path/synthetic_data_generation/outcomes.csv"
outcomes_noiseless_path="$project_path/synthetic_data_generation/outcomes_noiseless.csv"
effects_path="$project_path/synthetic_data_generation/effects.csv"



# Export stuff
export CONFIG_FILE=$config_path
export RESULTS_FILE="$results_file"
export PARAMETER="n_clusters"


# Print out the paths for debugging
echo "Script directory: $script_dir"
echo "Project path: $project_path"
echo "Config path: $config_path"

# Check if config file exists
if [ ! -f "$config_path" ]; then
    echo "Config file not found: $config_path"
    exit 1
fi




# Loop through number of clusters and update config file, then run the Python script
for n_clust in 2 5 10 15 20 25; do
    echo "Updating n_clusters to $n_clust"

    # Update the n_clusters value in the config file
    crudini --set "$config_path" evaluation clustering_n_clusters "$n_clust"
    
    if [ $? -ne 0 ]; then
        echo "Failed to update config file"
        exit 1
    fi

    # Generate data with the new n_clusters parameter
    python "$project_path/synthetic_data_generation/synthetic_data_faster.py"

    if [ $? -ne 0 ]; then
        echo "Failed to generate synthetic data"
        exit 1
    fi

    sleep 1

    export PARAMETER_VALUE=$n_clust

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
