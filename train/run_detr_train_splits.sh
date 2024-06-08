#!/bin/bash

# This script is designed to run a Python training script, `train_detector.py`.

skf_iterations=(0) # List of iteration indices from Stratified K-Fold

for iter in "${skf_iterations[@]}"; do
	log_file="train_yolo_${iter}.log" # Log file for output redirection
	echo "Running train_detector.py with iter=${iter}..."
	# Run the training script in the background
	nohup python train_detector.py ${iter} > $log_file 2>&1 &
done
