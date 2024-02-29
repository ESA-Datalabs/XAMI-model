from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

def show_masks(masks, ax, random_color=False):
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


#Instance
model = YOLO('yolov8n-seg.yaml') 
model = YOLO('yolov8n-seg.pt') 

import yaml

data_yaml_path = "./xmm_om_images_512_SG_SR_CR_only-even_fewer/data.yaml"
with open(data_yaml_path, 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

project = "yolov8-segm-ft_no_stars-n" 
name = "200_epochs-no_stars-n-even_fewer_obj" 

# Train the model
results = model.train(data=data_yaml_path,
                      project=project,
                      name=name,
					  task='detect',
                      epochs=200,
                      patience=0, # patience=0 disables early stopping
                      batch=8,
                      imgsz=512,
                      device=0,
					  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
					  lr0=0.001,
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
