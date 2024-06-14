from ultralytics import YOLO, RTDETR
import sys

if len(sys.argv) ==1:
    iter = 0
elif len(sys.argv) == 2:
    iter = sys.argv[1]
else:
    print("Usage: python train_detector.py [iter]")
    sys.exit(1)

yolo_dataset_path = f"./xami_dataset_zip/xami_dataset_yolov8/" # replace with path to YOLO dataset
data_yaml_path = yolo_dataset_path+'data.yaml'
device = 2

model_type = 'yolov8'

if model_type == 'yolov8':
        
    # Train YOLOv8-segm
    model_checkpoint = 'yolov8l-seg.pt'
    model = YOLO(model_checkpoint) 

    project = f"yolov8-segm-{iter}" 
    name = model_checkpoint.replace('.pt', '') 

    # Train YOLOv8 model
    results = model.train(data=data_yaml_path,
                          project=project,
                          name=name,
    					  task='detect',
                          epochs=300,
                          patience=0, # patience=0 disables early stopping
                          batch=16,
                          imgsz=512,
                          device=device,
    					  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    					  lr0=0.0006,
    					  dropout=0.2,
    					  mask_ratio = 1,
    					  mosaic=0,
    					  box=0.9,
    					  cls=0.8,
                         )
    
if model_type=='rt-detr':
        
    # Train RT-DETR
    model_checkpoint = 'rtdetr-l.pt'
    model = RTDETR(model_checkpoint)

    project = f"rt-detr-{iter}" 
    name = model_checkpoint.replace('.pt', '') 

    # Train the model
    results = model.train(data=data_yaml_path,
                        project=project,
                        name=name,
                        task='detect',
                        epochs=300,
                        patience=0, # patience=0 disables early stopping
                        batch=16,
                        imgsz=512,
                        device=0,
                        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                        lr0=0.0006/2,
                        dropout=0.2,
                        mask_ratio = 1,
                        mosaic=0,
                        box=0.9,
                        cls=0.8,
                        deterministic=False, # to silence some warnings about cuda's non-deterministic behavior
                        )

# # Train the model
# results = model.train(data=data_yaml_path,
#                       project=project,
#                       name=name,
# 					  task='detect',
#                       epochs=300,
#                       patience=0, # patience=0 disables early stopping
#                       batch=64,
#                       imgsz=512,
#                       device=device,
# 					  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
# 					  lr0=0.003,
# 					  dropout=0.2,
# 					  mask_ratio = 1,
# 					  mosaic=0.7,
# 					  cos_lr=True,
# 					  box=7.5,
# 					  cls=1.0,
#                       dfl=1.0,
#                       weight_decay=0.0007,
#                       agnostic_nms=True,
#                       iou=0.6,
#                       nms=True,
# 					  label_smoothing=0.1,
#                       augment=True, 
#                       mixup=0.7,
#                       flipud=0.6,
#                       # freeze layers
#                      )   
                        
# # Train RT-DETR
# # Load a COCO-pretrained RT-DETR-l model
# model_checkpoint = 'rtdetr-l.pt'
# model = RTDETR(model_checkpoint)

# project = f"rt-detr-iter{iter}" 
# name = model_checkpoint.replace('.pt', '') 

# # Train the model
# results = model.train(data=data_yaml_path,
#                       project=project,
#                       name=name,
# 					  task='detect',
#                       epochs=300,
#                       patience=0, # patience=0 disables early stopping
#                       batch=32,
#                       imgsz=512,
#                       device=device,
# 					  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
# 					  lr0=0.0006/2,
# 					  dropout=0.2,
# 					  mask_ratio = 1,
# 					  mosaic=0,
# 					  cos_lr=True,
# 					  box=1.0,
# 					  cls=0.8,
# 					  label_smoothing=0.1,
#                       augment=True, 
#                      )