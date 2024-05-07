#!/bin/bash

iterations=(1 2 3 4)
device_ids=(0 1 2 3)

gpu=1
for iter in "${iterations[@]}"; do
	log_file="train_log_sam_${iter}.log"
	echo "Running train_sam.py with iter=${iter}..."
	nohup python train_sam.py ${iter} ${gpu} > $log_file 2>&1 &
	gpu=$((gpu + 1))
    if [ "$gpu" -ge 4 ]; then  
      gpu=1
    fi
done