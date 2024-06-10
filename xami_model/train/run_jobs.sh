#!/bin/bash

# Number of GPUs
num_gpus=4


# Run the first set of jobs
for ((i=0; i<$num_gpus; i++))
do
    gpu_id=$i
    echo "Running job $i on GPU $gpu_id"
    nohup python train_yolo_sam.py $gpu_id > output_$i.log 2>&1 &
done