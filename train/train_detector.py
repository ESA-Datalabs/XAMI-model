from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import yaml
from ultralytics import RTDETR
import sys

iter = sys.argv[1]

model_checkpoint = 'yolov8n-seg.pt'
model = YOLO(model_checkpoint) 
data_yaml_path = f"../../AstroArtefactToolkit_XMMoptical/mskf_YOLO_{iter}/data.yaml"

with open(data_yaml_path, 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

project = f"yolov8-segm-{iter}" 
name = model_checkpoint.replace('.pt', '') 

# Train YOLOv8 model
results = model.train(data=data_yaml_path,
                      project=project,
                      name=name,
					  task='segm',
                      epochs=300,
                      patience=0, # patience=0 disables early stopping
                      batch=16,
                      imgsz=512,
                      device=1,
					  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
					  lr0=0.0006,
					  dropout=0.2,
					  mask_ratio = 1,
					  mosaic=0,
					  # cos_lr=True,
					  box=0.9,
					  cls=0.8,
					  # label_smoothing=0.1,
                      # augment=True, 
                      # freeze layers
                     )

# # Train RT-DETR
# # Load a COCO-pretrained RT-DETR-l model
# model_checkpoint = 'rtdetr-l.pt'
# model = RTDETR(model_checkpoint)

# project = f"rt-detr-ft_no_stars-n-iter{iter}" 
# name = model_checkpoint.replace('.pt', '') 

# # Train the model
# results = model.train(data=data_yaml_path,
#                       project=project,
#                       name=name,
# 					  task='detect',
#                       epochs=300,
#                       patience=0, # patience=0 disables early stopping
#                       batch=16,
#                       imgsz=512,
#                       device=0,
# 					  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
# 					  lr0=0.0006/2,
# 					  dropout=0.2,
# 					  mask_ratio = 1,
# 					  mosaic=0,
# 					  # cos_lr=True,
# 					  box=0.9,
# 					  cls=0.8,
# 					  # label_smoothing=0.1,
#                       # augment=True, 
#                       # freeze layers
#                      )