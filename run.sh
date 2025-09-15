#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if at least one model name is provided as an argument.
if [ "$#" -lt 1 ]; then
    echo "Error: Please provide at least one model name."
    echo "Usage: $0 <model_name1> [model_name2] [...]"
    echo "Example: $0 flux-schnell sdxl"
    exit 1
fi

# Initialize variables to store the list of input files for each command.
input_files_cmd1=""
input_files_cmd2=""
input_files_cmd3=""

# Loop through all model names passed as command-line arguments.
for model_name in "$@"
do
    echo "Preparing file paths for model '$model_name'..."

    # Construct the required file paths for the three commands based on the model name.
    # File path for command 1
    file1="output/${model_name}/merged_result_${model_name}.jsonl"
    # File path for command 2 (assumes it's generated based on command 1's output)
    file2="output/${model_name}/subject_detection_${model_name}.jsonl"
    # File path for command 3 (assumes it's generated based on command 2's output)
    file3="output/${model_name}/merged_result_${model_name}_vllm.jsonl"

    # Append the constructed file path to the corresponding variable, separated by a space.
    input_files_cmd1+=" ${file1}"
    input_files_cmd2+=" ${file2}"
    input_files_cmd3+=" ${file3}"
done

# --- Execute the three Python commands in sequence ---

echo "----------------------------------------"
echo "Step 1: Executing vllm_subject_mp.py"
# Use xargs to trim leading/trailing whitespace from the variable.
cmd1_args=$(echo "$input_files_cmd1" | xargs)
echo "Executing command: python vllm_subject_mp.py --input_files ${cmd1_args}"
python vllm_subject_mp.py --input_files ${cmd1_args}
echo "Step 1 complete."
echo ""

echo "----------------------------------------"
echo "Step 2: Executing vllm_mp.py"
cmd2_args=$(echo "$input_files_cmd2" | xargs)
echo "Executing command: python vllm_mp.py --input_files ${cmd2_args}"
python vllm_mp.py --input_files ${cmd2_args}
echo "Step 2 complete."
echo ""

echo "----------------------------------------"
echo "Step 3: Executing gather_response_score100.py"
cmd3_args=$(echo "$input_files_cmd3" | xargs)
echo "Executing command: python gather_response_score100.py --input_files ${cmd3_args}"
python gather_response_score100.py --input_files ${cmd3_args}
echo "Step 3 complete."
echo "----------------------------------------"

echo "All tasks completed successfully!"