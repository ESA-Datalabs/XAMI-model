from ultralytics import YOLO, RTDETR
import yaml
import sys
import json
import os

# Append project path when running in CLI
# Otherwise, the project path is already in the sys.path
# relative_project_path = os.path.join(sys.path[0], '../')
relative_project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(relative_project_path)
print('Project path:', relative_project_path)
from dataset import coco_to_yolo_converter

if len(sys.argv) ==1:
    iter = 0
elif len(sys.argv) == 2:
    iter = sys.argv[1]
else:
    print("Usage: python train_detector.py [iter]")
    sys.exit(1)

yolo_dataset_path = f"../data/xami_dataset_YOLO/" # # replace with path to YOLO output dataset
 
#  Convert the dataset form COCO IS format into YOLOv8
convert = False

if convert:
    dir_absolute_path = relative_project_path
    dataset_path = '../xami_dataset/' 
    json_file_path = dataset_path+'train/'+'_annotations.coco.json' # need only training example
    # YOLO yaml files works with absolute paths. Replace this with the actual absolute path to the dataset.
    dataset_absolute_path = dir_absolute_path+'xami_dataset/'
    
    with open(json_file_path) as f:
        data_in = json.load(f)
        
    classes = [str(cat['name']) for cat in data_in['categories']]
    
    for mode in ['train', 'valid']:
        input_path = f"{dataset_path}{mode}"

        # adding '/' at the end will give an error for the parent directory
        output_path = f"{yolo_dataset_path}/{mode}"
        input_json_train = f"_annotations.coco.json"
        converter = coco_to_yolo_converter.COCOToYOLOConverter(input_path, output_path, input_json_train, plot_yolo_masks=False)
        converter.convert()
        
        # generate data.yaml
        yaml_path = os.path.dirname(output_path)+f'/data.yaml'

        if mode =='valid': # train and valid folder successfully created
            yolo_data = {
                'names': classes,
                'nc': len(classes),
                'train': f'{os.path.join(dir_absolute_path, os.path.dirname(output_path)).replace(".", "").replace("//", "/")}/train/images',
                'val': f'{os.path.join(dir_absolute_path, os.path.dirname(output_path)).replace(".", "").replace("//", "/")}/valid/images'
            }
            
            # Write the data to a YAML file
            with open(yaml_path, 'w') as file:
                yaml.dump(yolo_data, file, default_flow_style=False)
            
            print(f"YAML file {yaml_path} created and saved.")

model_checkpoint = 'yolov8n-seg.pt'
model = YOLO(model_checkpoint) 
data_yaml_path = yolo_dataset_path+'data.yaml' #f"../../XAMI-dataset/notebooks/mskf_YOLO_{iter}/data.yaml"

with open(data_yaml_path, 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

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