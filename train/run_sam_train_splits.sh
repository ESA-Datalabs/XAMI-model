#!/bin/bash

# This script is designed to run a Python training script, `train_sam.py`, across multiple GPUs.

skf_iterations=(0 1 2 3)  # List of iteration indices
device_ids=(0 1 2 3)      # List of GPU device IDs (assumes 4 GPUs)

gpu=0  # Initial GPU ID

for iter in "${skf_iterations[@]}"; do
    log_file="train_sam_${iter}.log"  # Log file for output redirection
    echo "Running train_sam.py with iter=${iter}..." 
    # Run the training script in the background
    nohup python train_sam.py ${iter} ${gpu} > $log_file 2>&1 &
    gpu=$((gpu + 1))  

    # Reset GPU ID to 0 if it exceeds the number of available GPUs
    if [ "$gpu" -ge 4 ]; then  
        gpu=0
    fi
done
