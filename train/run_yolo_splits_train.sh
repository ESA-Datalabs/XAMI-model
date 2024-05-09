#!/bin/bash

iterations=(0 1 2 3)

for iter in "${iterations[@]}"; do
	log_file="train_detector_${iter}.log"
	echo "Running train_detector.py with iter=${iter}..."
	nohup python train_detector.py ${iter} > $log_file 2>&1 &
done
