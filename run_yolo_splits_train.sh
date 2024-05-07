#!/bin/bash

iterations=(1 2 3 4)

for iter in "${iterations[@]}"; do
	log_file="train_log_yolo_${iter}.log"
	echo "Running ft_YOLO_V8_segm.py with iter=${iter}..."
	nohup python ft_YOLO_V8_segm.py ${iter} > $log_file 2>&1 &
done
