# %% [markdown]
# **check current model prediction**

# %%
# import sys
# from YoloSamPipeline import YoloSam

# %%
# yolo_path = './yolov8-segm-ft_no_stars-n/rtdetr-l/weights/last.pt' 

# yolo_sam_pipeline = YoloSam(
#     device='cuda:0', 
#     yolo_checkpoint=yolo_path, 
#     sam_checkpoint='./output_sam/ft_mobile_sam_final_2024-04-05 05:45:16.547865.pth', # the checkpoint and model_type (vit_h, vit_t, etc..) must be compatible
#     model_type='vit_t')

# %%
# yolo_sam_pipeline.run_predict('../XMM_OM_dataset/zscaled_512_stretched/S0412991401_U.png') 

# %%
# yolo_sam_pipe.run_predict('../XMM_OM_dataset/zscaled_512_stretched/S0412991401_U.png') 

# %% [markdown]
# # Import the YOLOv8 pretrained model
# 
# - The model is pretrained (in another notebook)  using a Roboflow dataset version on OM images. 

# %%
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
PYTORCH_NO_CUDA_MEMORY_CACHING=1

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import torch
from torch import cuda
import os
import numpy as np
import random
from PIL import Image
import matplotlib.colors as mcolors
import numpy.ma as ma
import json
np.set_printoptions(precision=15)

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from typing import Any, Dict, Generator, List
import matplotlib.pyplot as plt
import numpy as np

from dataset import dataset_utils, voc_annotate_and_Roboflow_export
from sam_predictor import predictor_utils, astro_sam
from OM_DeepLearning.XMM_OM_code_git.losses import loss_utils
from yolo_predictor import yolo_predictor_utils
# import torch.autograd.profiler as profiler

# %%
device_id = 2
batch_size = 12
lr=6e-5
wd=0.0005
torch.cuda.set_device(device_id) # ❗️❗️❗️

# %%
wandb_track = False

if wandb_track:
    from datetime import datetime
    # !pip install wandb
    # !wandb login --relogin
    import wandb
    wandb.login()
    run = wandb.init(project="yolo-sam", name=f"yolo-sam {datetime.now()}")

# %% [markdown]
# ## Dataset (YOLOv8 format)

# %% [markdown]
# **hyperparameters docs: https://docs.ultralytics.com/usage/cfg/#train**

# %%
yolo_dataset_path = './roboflow_datasets/xmm_om_artefacts_512-21-YOLO/'

# %%
import yaml
with open(yolo_dataset_path+"data.yaml", 'r') as stream:
    yam_data = yaml.safe_load(stream) # dict with keys 'names', 'nc', 'roboflow', 'test', 'train', 'val'
yam_data['names']

classes = {i:name for i, name in enumerate(yam_data['names'])}
train_path = yam_data['train']
val_path = yam_data['val']
print(classes)

# %% [markdown]
# # Couple YOLO bboxes with SAM

# %% [markdown]
# **load SAM model**

# %%
import sys
import PIL
from PIL import Image

sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/')
from ft_mobile_sam import sam_model_registry, SamPredictor

# mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/Mobil_eSAM-fine-tuning/weights/mobile_sam.pt"
mobile_sam_checkpoint = "./output_sam/ft_mobile_sam_final.pth"

yolov8_pretrained_model = YOLO('./yolov8-segm-ft_no_stars-n/yolov8n-seg2/weights/best.pt');
yolov8_pretrained_model.to(f'cuda:{device_id}');

device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
print("device:", device)

mobile_sam_model = sam_model_registry["vit_t"](checkpoint=mobile_sam_checkpoint)
mobile_sam_model.to(device)
# %%
astrosam_model = astro_sam.AstroSAM(mobile_sam_model, device, None)
astrosam_model.predictor = SamPredictor(astrosam_model.model)

# %%
train_dir = yolo_dataset_path+'train/images/'
valid_dir = yolo_dataset_path+'valid/images/'
train_image_files = os.listdir(train_dir)
valid_image_files = os.listdir(valid_dir)

# %%
for name, param in astrosam_model.model.named_parameters():
    params_to_train = ['mask_tokens', 'output_upscaling', 'output_hypernetworks_mlps', 'iou_prediction_head']
    if 'mask_decoder' in name: # and any(s in name for s in params_to_train):
        param.requires_grad = True
    else:
        param.requires_grad = False
        
print(f"🚀 The model has {sum(p.numel() for p in astrosam_model.model.parameters() if p.requires_grad)} trainable parameters.\n")
predictor_utils.check_requires_grad(astrosam_model.model)

# %%
import time
import torch.nn.functional as F
import tqdm
from tqdm import tqdm

train_num_batches = len(train_image_files) // batch_size
valid_num_batches = len(valid_image_files) // batch_size
parameters_to_optimize = [param for param in astrosam_model.model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(parameters_to_optimize, lr=lr, weight_decay=wd) #betas=(0.9, 0.999))

# %%
len(train_image_files), len(valid_image_files)

# %%
best_valid_loss = float('inf')
num_epochs = 40
n_epochs_stop = 5 + num_epochs//10
epoch_sam_loss_train_list, epoch_sam_loss_val_list, epoch_yolo_loss_train_list, epoch_yolo_loss_val_list = [], [], [], []

for epoch in range(num_epochs):

    # train
    # astrosam_model.model.train()
    epoch_sam_loss_train, epoch_yolo_loss_train, preds, gts, gt_classes, pred_classes, all_iou_scores = astrosam_model.run_yolo_sam_epoch(
                                                                                                                yolov8_pretrained_model,
                                                                                                                phase='train',
                                                                                                                batch_size=batch_size, 
                                                                                                                image_files=train_image_files, 
                                                                                                                images_dir=train_dir, 
                                                                                                                num_batches=train_num_batches,
                                                                                                                optimizer=optimizer)
    
    # validate
    # astrosam_model.model.eval()
    with torch.no_grad():
        epoch_sam_loss_val, epoch_yolo_loss_val, preds, gts, gt_classes, pred_classes, all_iou_scores = astrosam_model.run_yolo_sam_epoch(
                                                                                                                    yolov8_pretrained_model,
                                                                                                                    phase='val',
                                                                                                                    batch_size=batch_size, 
                                                                                                                    image_files=valid_image_files, 
                                                                                                                    images_dir=valid_dir, 
                                                                                                                    num_batches=valid_num_batches)
    epoch_sam_loss_train_list.append(epoch_sam_loss_train)
    epoch_sam_loss_val_list.append(epoch_sam_loss_val)
    epoch_yolo_loss_train_list.append(epoch_yolo_loss_train)
    epoch_yolo_loss_val_list.append(epoch_yolo_loss_val)

    print(f"epoch train SAM loss: {epoch_sam_loss_train}, epoch valid SAM loss: {epoch_sam_loss_val}")
    print(f"epoch train YOLO loss: {epoch_yolo_loss_train}, epoch valid YOLO loss: {epoch_yolo_loss_val}")

    if wandb_track:
        wandb.log({'epoch train SAM loss': epoch_sam_loss_train, 'epoch valid SAM loss': epoch_sam_loss_val})
        wandb.log({'epoch train YOLO loss': epoch_yolo_loss_train, 'epoch valid YOLO loss': epoch_yolo_loss_val})
 
    if epoch_sam_loss_val < best_valid_loss:
        best_valid_loss = epoch_sam_loss_val
        best_model = astrosam_model.model
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print("Early stopping initiated.")
            early_stop = True
            break

torch.save(best_model.state_dict(), f'yolo_sam_final.pth')

if wandb_track:
    wandb.run.summary["batch_size"] = batch_size
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["best_valid_loss"] = best_valid_loss
    wandb.run.summary["num_epochs"] = num_epochs
    wandb.run.summary["learning rate"] = lr
    wandb.run.summary["weight_decay"] = wd
    wandb.run.summary["# train images"] = len(train_image_files)
    wandb.run.summary["# validation images"] = len(valid_image_files)
    wandb.run.summary["checkpoint"] = mobile_sam_checkpoint
    run.finish()


# %% [markdown]
# ## Compute mean average precision

# %%
# epoch_sam_loss_val, epoch_yolo_loss_val, preds, gts, gt_classes, pred_classes, all_iou_scores = astrosam_model.run_yolo_sam_epoch(
#     yolov8_pretrained_model,
#     phase='val',
#     batch_size=batch_size, 
#     image_files=valid_image_files, 
#     images_dir=valid_dir, 
#     num_batches=valid_num_batches)

# %%
len(valid_image_files)

# %%
len(preds), len(gts), len(gt_classes), len(pred_classes), len(all_iou_scores)

# %%
all_iou_scores

# %%
preds[1][0].shape

# %%
# all_ious = [iou_score.detach().cpu().numpy() for iou_score in all_iou_scores]
# all_ious

# %%
# pred_masks_all = preds 
# gt_masks_all = gts  

# %%
all_ious_flatten = []
for i in range(len(pred_classes)):
    for j in range(len(pred_classes[i])):
        all_ious_flatten.append(all_iou_scores[i][j].detach().cpu().numpy()[0])

# %%
# all_ious

# %%
# pred_flatten = []
# pred_classes_flatten = []
# gt_flatten = []
# gt_classes_flatten = []

# for i in range(len(pred_masks_all)):
#     for j in range(len(pred_masks_all[i])):
#         pred_flatten.append((pred_masks_all[i][j][0].detach().cpu().numpy()>0.5).astype(int))
#         pred_classes_flatten.append(pred_classes[i][j])

# for i in range(len(gt_masks_all)):
#     for j in range(len(gt_masks_all[i])):
#         gt_flatten.append((gt_masks_all[i][j][0].detach().cpu().numpy()>0.5).astype(int))
#         gt_classes_flatten.append(gt_classes[i][j])

# %%
# pred_flatten = np.array(pred_flatten)
# gt_flatten = np.array(gt_flatten)
# all_ious = np.array(all_ious)
# gt_classes_flatten = np.array(gt_classes_flatten)
# pred_classes_flatten = np.array(pred_classes_flatten)

# %%
# pred_flatten.shape, gt_flatten.shape, gt_classes_flatten.shape, pred_classes_flatten.shape, all_ious.shape

# %%
import numpy as np
import matplotlib.pyplot as plt

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(all_ious_flatten, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Segmentation IoUs')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# %%
all_preds, all_gts, all_ious = [], [], []
for i in range(len(preds)):
    gt_i = np.array([gts[i][j][0].detach().cpu().numpy() for j in range(len(gts[i]))])
    pred_i = np.array([preds[i][j] for j in range(len(preds[i]))])
    ious_i = np.array([all_iou_scores[i][j][0].detach().cpu().numpy() for j in range(len(all_iou_scores[i]))])
    all_gts.append(gt_i)    
    all_preds.append(pred_i)  
    all_ious.append(ious_i)
    
# all_ious = [iou_score.detach().cpu().numpy() for iou_score in all_iou_scores]
all_gt_classes = [np.array(gt_classes[i], dtype=np.int8) for i in range(len(gt_classes))]
all_pred_classes = [np.array(pred_classes[i], dtype=np.int8) for i in range(len(pred_classes))]

len(all_preds), len(all_gts), len(all_ious), len(all_gt_classes), len(all_pred_classes)

# %%
all_ious[1]

# %%
all_preds[1].shape, all_gts[1].shape

# %%
from torch import tensor

preds_per_image = []
gts_per_image = []

for img_i in range(len(all_preds)):
    img_preds_dict = dict(
        masks=tensor(all_preds[img_i], dtype=torch.bool),
        scores=tensor(all_ious[img_i]),
        labels=tensor(all_pred_classes[img_i], dtype=torch.int16),
      )
    img_gts_dict = dict(
        masks=tensor(all_gts[img_i], dtype=torch.bool),
        labels=tensor(all_gt_classes[img_i], dtype=torch.int16),
      )
    print(img_preds_dict['masks'].shape)
    print(img_gts_dict['masks'].shape)
    
    preds_per_image.append(img_preds_dict)
    gts_per_image.append(img_gts_dict)
    

# %%
preds_per_image[1]['masks'].shape, preds_per_image[1]['scores'].shape, preds_per_image[1]['labels'].shape,

# %%
preds_per_image

# %%
gts_per_image

# %%
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(
    iou_type = "segm", 
    iou_thresholds = [0.75], 
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True,
    extended_summary=False)

# '''
# extended_summary:
# - ``ious``: a dictionary containing the IoU values for every image/class combination e.g.
#                   ``ious[(0,0)]`` would contain the IoU for image 0 and class 0. Each value is a tensor with shape
#                   ``(n,m)`` where ``n`` is the number of detections and ``m`` is the number of ground truth boxes for
#                   that image/class combination.
#                 - ``precision``: a tensor of shape ``(TxRxKxAxM)`` containing the precision values. Here ``T`` is the
#                   number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
#                   ``A`` is the number of areas and ``M`` is the number of max detections per image.
#                 - ``recall``: a tensor of shape ``(TxKxAxM)`` containing the recall values. Here ``T`` is the number of
#                   IoU thresholds, ``K`` is the number of classes, ``A`` is the number of areas and ``M`` is the number
#                   of max detections per image.
#                 - ``scores``: a tensor of shape ``(TxRxKxAxM)`` containing the confidence scores.  Here ``T`` is the
#                   number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
#                   ``A`` is the number of areas and ``M`` is the number of max detections per image.
# '''

# %%
metric.update(preds_per_image, gts_per_image)
from pprint import pprint
pprint(metric.compute())

# %%
import matplotlib.pyplot as plt
import numpy as np

num_instances = len(preds_per_image)
k=0
for i in range(num_instances):
    for j in range(len(preds_per_image[i]['masks'])):
        k+=1
        if k>20:
            break
        # if preds_per_image[i]["scores"][j] >0.9:
        if True:
            plt.figure(figsize=(20, 10))
            # Plot predicted mask
            plt.subplot(1, 2, 1)
            plt.imshow(preds_per_image[i]['masks'][j].numpy(), cmap='gray')
            plt.title(f'Predicted Label: {preds_per_image[i]["labels"][j].item()}, Score: {preds_per_image[i]["scores"][j].item():.2f}')
            plt.axis('off')
            
            # Plot target (ground truth) mask
            plt.subplot(1, 2, 2)
            plt.imshow(gts_per_image[i]['masks'][j].numpy(), cmap='gray')
            plt.title(f'True Label: {gts_per_image[i]["labels"][j].item()}')
            plt.axis('off')
        
            plt.show()
            plt.close()

# %%
a = 0
all=0
for i in range(num_instances):
    for j in range(len(preds_per_image[i]['masks'])):
        pred_mask = preds_per_image[i]['masks'][j].numpy()
        pred_label = preds_per_image[i]["labels"][j].item()
        score = preds_per_image[i]["scores"][j].item()
        gt_mask = gts_per_image[i]['masks'][j].numpy()
        gt_label = gts_per_image[i]["labels"][j].item()
        if pred_label == gt_label:
            a+=1
        all+=1
        
a, all, a/all

# %%
import glob
from roboflow import Roboflow

def export_image_det_to_Roboflow(input_dir, filename, masks, obj_results):
    class_names = obj_results[0].names
    class_labels = obj_results[0].boxes.data[:, -1].int().tolist()
    
    objects = []
    for i in range(len(masks)):
        # masks[i]: [ 1, H, W]
        mask_np = masks[i].detach().cpu().numpy()
        polygon = voc_annotate_and_Roboflow_export.binary_image_to_polygon(mask_np[0])
        bbox = dataset_utils.mask_to_bbox(mask_np)
        if class_names[class_labels[i]] != 'star' and class_names[class_labels[i]] != 'other': # ignore stars and 'other' label
            objects.append({
                'name': class_names[class_labels[i]],
                'bbox': bbox,
                'segmentations': polygon[0]
            })
    if len(objects)>0:
        voc_annotate_and_Roboflow_export.create_annotation_SAM(
            filename=filename, 
            width=512, 
            height=512, 
            depth=3, 
            objects=objects, 
            offset=1.2) # generating xml file for VOC format
        image_path = input_dir+filename
        annotation_filename = filename.replace(".png", ".xml")
        upload_project.upload(image_path, annotation_filename, overwrite=False)
        os.remove(annotation_filename)
    else:
        print("No objects after label filtering.")

# %%
import os
import torch
import cv2 

# Optional Roboflow export in VOC format given filenames
export_to_Roboflow = False
import time

# best_model = mobile_sam_model.cpu()
if export_to_Roboflow:
    # Initialize Roboflow client
    rf = Roboflow(api_key="EBeK30tpU3HW2VGGl0xa")
    upload_project = rf.workspace("iuliaelisa").project("xmm_om_artefacts_512") # error if the project doesn't exist

new_images_dir = '../XMM_OM_dataset/zscaled_512_stretched/'
new_image_files =  os.listdir(new_images_dir)
# best_model.eval()
    
with torch.no_grad(): 
    # eg_img = 'S0018141301_M.png'
    for image_name in new_image_files[3022:3030]:
        print('Image', new_images_dir+image_name)
        image = cv2.imread(new_images_dir + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        try:
            sam_mask_pre, obj_results = yolo_sam_pipeline.run_predict(new_images_dir + image_name) 
            if export_to_Roboflow:
                export_image_det_to_Roboflow(new_images_dir, image_name, sam_mask_pre, obj_results)
                
        except Exception as e: # most likely the image had no annotations
            print(e)
            if export_to_Roboflow:
                upload_project.upload(new_images_dir+image_name)
            continue

# %%
# mobile_sam_state_dict = best_model.state_dict()

# %%
# mobile_sam_state_dict.keys()

# %%
# torch.save(yolov8_pretrained_model.state_dict(), 'yolo_model.bin')
# torch.save(best_model.state_dict(), 'pytorch_model.bin')

# %%
# import json

# config_dict = {
#     "hidden_size": model.hidden_size,
#     "num_attention_heads": model.num_attention_heads,
#     "num_hidden_layers": model.num_hidden_layers,
# }

# with open("config.json", "w") as f:
#     json.dump(config_dict, f, indent=4)

# %%
# yolov8_pretrained_model.export(format='onnx', imgsz=[512,512])

# %%
# import torch
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/')

# from importlib import reload
# from ft_mobile_sam import sam_model_registry, SamPredictor
# from ft_mobile_sam.utils.onnx import SamOnnxModel

# import onnxruntime
# from onnxruntime.quantization import QuantType
# from onnxruntime.quantization.quantize import quantize_dynamic

# import warnings
# onnx_model_path = None  # Set to use an already exported model, then skip to the next section.

# onnx_model_path = "sam_onnx_example.onnx"

# sam = mobile_sam_model.to('cpu') # the model must be set on CP
# onnx_model = SamOnnxModel(sam, return_single_mask=True)

# dynamic_axes = {
#     "point_coords": {1: "num_points"},
#     "point_labels": {1: "num_points"},
# }

# embed_dim = sam.prompt_encoder.embed_dim
# embed_size = sam.prompt_encoder.image_embedding_size
# mask_input_size = [4 * x for x in embed_size]
# dummy_inputs = {
#     "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
#     "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
#     "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
#     "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
#     "has_mask_input": torch.tensor([1], dtype=torch.float),
#     # "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
# }
# output_names = ["masks", "iou_predictions", "low_res_masks"]

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
#     warnings.filterwarnings("ignore", category=UserWarning)
#     with open(onnx_model_path, "wb") as f:
#         torch.onnx.export(
#             onnx_model,
#             tuple(dummy_inputs.values()),
#             f,
#             export_params=True,
#             verbose=False,
#             opset_version=16,
#             do_constant_folding=True,
#             input_names=list(dummy_inputs.keys()),
#             output_names=output_names,
#             dynamic_axes=dynamic_axes,
#         )    

# %%
# optional: export to ONNX

# !python scripts/export_onnx_model.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./mobile_sam.onnx


