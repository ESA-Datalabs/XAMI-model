#!/usr/bin/env python
# coding: utf-8

# ## Set up

# In[1]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"

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

# Ensure deterministic behavior (cannot control everything though)
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.set_device(6) # ❗️❗️❗️

from importlib import reload
import dataset_utils
reload(dataset_utils)
from dataset_utils import *

import predictor_utils
reload(predictor_utils)
from predictor_utils import *


# In[2]:


# input_dir = '/workspace/raid/OM_DeepLearning/XMM_OM_code_git/xmm_om_images-contrast-512-v5-3/train/'
# json_file_path = '/workspace/raid/OM_DeepLearning/XMM_OM_code_git/xmm_om_images-contrast-512-v5-3/train/_annotations.coco.json'

input_dir = '/workspace/raid/OM_DeepLearning/XMM_OM_code_git/-xmm_om_images_v4-contrast-512-5-2/train/'
json_file_path = '/workspace/raid/OM_DeepLearning/XMM_OM_code_git/-xmm_om_images_v4-contrast-512-5-2/train/_annotations.coco.json'

# COCO segmentation bboxes are in XYWH format
with open(json_file_path) as f:
    data = json.load(f)
    
ground_truth_masks, bbox_coords, classes, class_categories = get_coords_and_masks_from_json(input_dir, data) # type: ignore


# In[3]:


raw_images = [input_dir+img_data['file_name'] for img_data in data['images']]

image_paths_no_augm = []

for im_path in raw_images:
    has_annots = 0
    for k, v in bbox_coords.items():
        if im_path.split('/')[-1] in k:
            has_annots = 1
            
    if has_annots==0:
                print("Img doesn't have annotations after filtering.")
    else:
        image_paths_no_augm.append(im_path)

len(image_paths_no_augm)


# In[4]:


'before', len(data['annotations']), 'after', len(ground_truth_masks.values()), len(bbox_coords.values()), len(classes.values())


# In[5]:


# for i in range(len(data['annotations'])):
#     points = data['annotations'][i]['segmentation'][0]
#     binary_m = create_mask(points, (512, 512)) # COCO segmentations are polygon points, and must be converted to masks
#     bbox = mask_to_bbox(binary_m)  #XYXY
#     if bbox[2] - bbox[0]<4 and bbox[3] - bbox[1] <4:
#         print(bbox[2],bbox[3])
#         print(data['annotations'][i]['image_id'])
#         plt.imshow(binary_m)
#         plt.show()
#         plt.close()


# ## Augmentation
# 
# This algorithm performs augmentations and updates the negative masks in the case of a geometric transformations. Otherwise, it masks the result of a noise/blur transformation given the mask of the initial image. 
# 
# **!! For augmentation, the bboxes are expected to be in the XYWH format, not XYXY format (used by SAM). However, the SAM AMG generated results are in the XYWH format (converted from XYXY).**
# 

# In[6]:


import os
import glob

files = glob.glob(f'{input_dir}/*augm*')

for file in files:
    os.remove(file)


# In[7]:


# https://albumentations.ai/docs/examples/showcase/
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import albumentations as A
import random
import math

# load utils
from importlib import reload
import dataset_utils
reload(dataset_utils)
from dataset_utils import *


def enlarge_bbox(bbox, delta, image_size):
    x_min, y_min, w, h = bbox
    x_min = max(0, math.floor(x_min))
    y_min = max(0,  math.floor(y_min))
    
    w = min(image_size[1], math.ceil(w))
    h = min(image_size[0], math.ceil(h))
    return np.array([x_min, y_min, w, h])

geometrical_augmentations = A.Compose([
    A.Flip(),
    A.RandomRotate90(),
    A.RandomSizedCrop((512 - 50, 512 - 50), 512, 512),
], bbox_params={'format':'coco', 'min_area': 0.1, 'min_visibility': 0.3, 'label_fields': ['category_id']}, p=1)

noise_blur_augmentations = A.Compose([
    A.GaussianBlur(blur_limit=(3, 3), p=1),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
    A.ISONoise(p=0.8),
], bbox_params={'format':'coco', 'min_area': 0.1, 'min_visibility': 0.3, 'label_fields': ['category_id']}, p=1)

    
image_paths = []
for image_path in image_paths_no_augm:
    image_paths.append(image_path)
    image_ = cv2.imread(image_path)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    image_size = (image_.shape[0], image_.shape[1])
    masks = [value_i for key_i, value_i in ground_truth_masks.items() if image_path.split('/')[-1] in key_i]
    bboxes_ = [enlarge_bbox(np.array([value_i[0], math.floor(value_i[1]),  math.floor(value_i[2] - value_i[0]), \
                                    math.floor(value_i[3] - value_i[1])]), 2, image_size) \
                                    for key_i, value_i in bbox_coords.items() if image_path.split('/')[-1] in key_i]   
    label_ids = [classes[key_i] for key_i, value_i in bbox_coords.items() if image_path.split('/')[-1] in key_i]

    image_size = (image_.shape[0], image_.shape[1])
    for bbox in bboxes_:
        if bbox[2] <= 1 or bbox[3] <= 1:
            print("Invalid bbox detected:", bbox)
        
    if len(bboxes_) != len(masks):
        print('len(bboxes_) != len(masks)', len(bboxes_), len(masks))
        continue

    if len(bboxes_) != len(label_ids):
        print('len(bboxes_) != len(label_ids)', len(bboxes_), len(label_ids))
        continue

    if len(bboxes_) == 0:
        print(image_path)
        
    img_negative_mask = (image_>0).astype(int)
    
    # the geometrical augm doesn't change the shape of the image
    augmented1 = augment_and_show(geometrical_augmentations, image_, masks, bboxes_, label_ids, class_categories, show_=False)
    new_image_negative_mask = (augmented1['image']>0).astype(int) # to mask the transform which is derived from the geometric transform
    augmented3 = augment_and_show(noise_blur_augmentations, image_, masks, bboxes_, label_ids, class_categories, show_=False)

    # mask the transform using the image negative mask
    augmented3['image'] = augmented3['image'] * img_negative_mask
        
    new_filename1 = image_path.replace('.'+image_path.split('.')[-1], '_augm1.jpg')
    new_filename3 = image_path.replace('.'+image_path.split('.')[-1], '_augm3.jpg')

    # print('aug masks1:', len(augmented1['masks']), 'bboxes:', len(augmented1['bboxes']))
    # print('aug masks3:',len(augmented3['masks']), 'bboxes:', len(augmented3['bboxes']))

    update_dataset_with_augms(augmented1, new_filename1, bbox_coords, ground_truth_masks, image_paths, classes)
    update_dataset_with_augms(augmented3, new_filename3, bbox_coords, ground_truth_masks, image_paths, classes)


# In[8]:


for im_path in image_paths:
    has_annots = 0
    for k, v in bbox_coords.items():
        if im_path.split('/')[-1] in k:
            has_annots = 1
            
    if has_annots==0:
           image_paths.remove(im_path)

len(image_paths)


# In[9]:


for key_i, value_i in bbox_coords.items():
     if image_path.split('/')[-1] in key_i:
        x1, y1, x2, y2 = value_i
        if x2 - x1<2 or y2-y1 <2:
            print(value_i)


# In[10]:


for image_path in image_paths:
    image_ = cv2.imread(image_path)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    if image_.shape [1]< 512:
        print(image_.shape)


# In[11]:


len(image_paths)


# In[12]:


# augmented3['image'].shape, len(augmented3['bboxes']), len(augmented3['masks']) 


# In[13]:


from PIL import Image
import numpy as np

means = []
stds = []

for image_path in image_paths:
    image = Image.open(image_path)

    image_array = np.array(image) / 255.0

    mean = image_array.mean()
    std = image_array.std()

    means.append(mean)
    stds.append(std)

means = np.array(means)
stds = np.array(stds)


# In[14]:


np.mean(means), np.std(means)


# In[15]:


np.mean(stds),np.std(stds)


# ## Dataset split

# In[16]:


import json 
import cv2
import numpy as np
from matplotlib.path import Path
import math

def split_list(input_list, percentages):
    size = len(input_list)
    idx = 0
    output = []
    for percentage in percentages:
        chunk_size = round(percentage * size)
        chunk = input_list[idx : idx + chunk_size]
        output.append(chunk)
        idx += chunk_size
    return output

def create_dataset(image_paths, ground_truth_masks, bbox_coords):
        
    d_gt_masks, d_bboxes = {}, {}
    for img_path in image_paths:
        id = img_path.split('/')[-1]
        d_gt_masks.update({mask_id:mask_array for mask_id, mask_array in ground_truth_masks.items() if mask_id.startswith(id)})
        d_bboxes.update({bbox_id:bbox for bbox_id, bbox in bbox_coords.items() if bbox_id.startswith(id)}) 

    return d_gt_masks, d_bboxes
    
training_size, val_size, test_size = (0.7, 0.2, 0.1)
splits = split_list(image_paths, [training_size, val_size, test_size])
training_image_paths, val_image_paths, test_image_paths = splits[0], splits[1], splits[2]

train_gt_masks, train_bboxes = create_dataset(training_image_paths, ground_truth_masks, bbox_coords)
test_gt_masks, test_bboxes = create_dataset(test_image_paths, ground_truth_masks, bbox_coords)
val_gt_masks, val_bboxes = create_dataset(val_image_paths, ground_truth_masks, bbox_coords)

del ground_truth_masks, bbox_coords, image_paths


# In[17]:


print('train:', len(training_image_paths), 'test:', len(test_image_paths), 'val:', len(val_image_paths))


# In[18]:


for one_path in training_image_paths[:4]:
    image_id2 = one_path.split('/')[-1]
    print(image_id2)
    image_masks_ids = [key for key in train_gt_masks.keys() if key.startswith(image_id2)]
    image_ = cv2.imread(one_path)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(image_)
    for name in image_masks_ids:
            show_box(train_bboxes[name], plt.gca())
            show_mask(train_gt_masks[name], plt.gca())
    plt.axis('off')
    # plt.savefig(f'example{image_id2}.png')
    plt.show()
    plt.close()


# ## 🚀 Prepare Mobile SAM Fine Tuning

# In[47]:


import sys
import PIL
from PIL import Image

sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/')
import ft_mobile_sam
from ft_mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/weights/mobile_sam.pt"
mobile_sam_checkpoint = "./ft_mobile_sam_final.pth"
device = "cuda:6" if torch.cuda.is_available() else "cpu"
print("device:", device)

mobile_sam_model = sam_model_registry["vit_t"](checkpoint=mobile_sam_checkpoint)
mobile_sam_model.to(device);


# In[20]:


use_wandb = True

if use_wandb:
    from datetime import datetime
    # !pip install wandb
    # !wandb login --relogin
    import wandb
    wandb.login()
    run = wandb.init(project="OM_AI_v1", name=f"ft_MobileSAM {datetime.now()}")

    wandb.watch(mobile_sam_model, log='all', log_graph=True)


# ## Convert the input images into a format SAM's internal functions expect.

# In[21]:


# Preprocess the images
import os
from collections import defaultdict
import torch
import segment_anything
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize

from importlib import reload
import astronomy_utils
reload(astronomy_utils)
from astronomy_utils import *

transform = ResizeLongestSide(mobile_sam_model.image_encoder.img_size)

def transform_image(image, k):
       
        # sets a specific mean for each image
        image_T = np.transpose(image, (2, 1, 0))
        mean_ = np.mean(image_T[image_T>0])
        std_ = np.std(image_T[image_T>0]) 
        pixel_mean = torch.as_tensor([mean_, mean_, mean_], dtype=torch.float, device=device)
        pixel_std = torch.as_tensor([std_, std_, std_], dtype=torch.float, device=device)

        mobile_sam_model.register_buffer("pixel_mean", torch.Tensor(pixel_mean).unsqueeze(-1).unsqueeze(-1), False) # not in SAM
        mobile_sam_model.register_buffer("pixel_std", torch.Tensor(pixel_std).unsqueeze(-1).unsqueeze(-1), False) # not in SAM

        # mobile_sam_model.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        # mobile_sam_model.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        transformed_data = {}
        negative_mask = np.where(image > 0, True, False)
        negative_mask = torch.from_numpy(negative_mask)  
        negative_mask = negative_mask.permute(2, 0, 1)
        negative_mask = resize(negative_mask, [1024, 1024], antialias=True) 
        negative_mask = negative_mask.unsqueeze(0)
        # scales the image to 1024x1024 by longest side 
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        # normalization and padding
        input_image = mobile_sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        input_image[~negative_mask] = 0
        transformed_data['image'] = input_image.clone() 
        transformed_data['input_size'] = input_size 
        transformed_data['image_id'] = k
        transformed_data['original_image_size'] = original_image_size
    
        return transformed_data


# In[22]:


from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths, bbox_coords, gt_masks, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_id = self.image_paths[idx].split("/")[-1]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        if self.transform is not None:
            return self.transform(image, img_id)
        else:
            print("❗️No transform❗️")
            return image
        
batch_size = 10

train_set = ImageDataset(training_image_paths, train_bboxes, train_gt_masks, transform_image) 
val_set = ImageDataset(val_image_paths, val_bboxes, val_gt_masks, transform_image) 
test_set = ImageDataset(test_image_paths, test_bboxes, test_gt_masks, transform_image) 

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# In[23]:


import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import torch.nn as nn

lr=6e-3
wd=5e-3

parameters_to_optimize = [param for param in mobile_sam_model.mask_decoder.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(parameters_to_optimize, lr=lr, weight_decay=wd) #betas=(0.9, 0.999))
ce_loss_fn = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1.0) # not very helpful

def dice_loss(pred, target, negative_mask = None, smooth = 1): 
    pred = pred.contiguous()
    target = target.contiguous()    
    
    if negative_mask is not None: # masked loss
        negative_mask = negative_mask.bool()
        pred = pred * negative_mask
        target = target * negative_mask 
        
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # axs[0].imshow(pred[0, 0].cpu().detach().numpy(), cmap='gray')
    # axs[0].set_title('Pred')
    # axs[1].imshow(target[0, 0].cpu().detach().numpy(), cmap='gray')
    # axs[1].set_title('Target')
    # axs[2].imshow(negative_mask[0, 0].cpu().detach().numpy(), cmap='gray')
    # axs[2].set_title('Negative Mask')
    # plt.show()

    intersection = (pred * target).sum()
    loss = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
    return 1 - loss

def focal_loss(inputs, targets, alpha = 0.8, gamma = 2):

        # inputs = inputs.flatten(0,1)
        targets = targets.flatten(0,2)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss
	
# ## Model weights

# In[24]:


for name, param in mobile_sam_model.named_parameters():
    if "mask_decoder" in name:
    # if False:
    # layers_to_fine_tune = ['mask_decoder.output_hypernetworks_mlps.1', 'mask_decoder.output_hypernetworks_mlps.2', 'mask_decoder.output_hypernetworks_mlps.3', \
    #                       'mask_decoder.iou_prediction_head.layers', 'mask_decoder.output_hypernetworks_mlps.0.layers', 'output_upscaling']
    # if any(s in name for s in layers_to_fine_tune):
        param.requires_grad = True
    else:
        param.requires_grad = False
        # if 'norm' in name:
        #     param.eval() 


# In[25]:


def check_requires_grad(model, show=True):
    for name, param in model.named_parameters():
        if param.requires_grad and show:
            print("✅ Param", name, " requires grad.")
        elif param.requires_grad == False:
            print("❌ Param", name, " doesn't require grad.")


# In[26]:


print(f"🚀 The model has {sum(p.numel() for p in mobile_sam_model.parameters() if p.requires_grad)} trainable parameters.\n")
check_requires_grad(mobile_sam_model)


# ## The classifier

# ## Run fine tuning

# In[27]:


len(train_gt_masks.keys())


# In[28]:


# **idea** can use mixed-precision training:
# This is a technique that involves using a mix of float16 and float32 tensors 
# to make the model use less memory and run faster. 
# PyTorch provides the torch.cuda.amp module for automatic mixed-precision training.
# from torch.cuda.amp import autocast, GradScaler


# In[29]:


import predictor_utils
reload(predictor_utils)
from predictor_utils import *

def validate_model_AMG(ft_mobile_sam_model, mobile_sam_model_orig):
    validation_loss = []
    ft_validation_loss = []
    for val_img_path in val_image_paths[:5]:
        with torch.no_grad():

            print('Original MobileSAM')
            annotated_image_orig_mobile_sam, annotated_image_orig_mobile_sam_loss = amg_predict(mobile_sam_model_orig, orig_mobile_SamAutomaticMaskGenerator, \
                                                       val_gt_masks, 'ft_MobileSAM', val_img_path, mask_on_negative=False, show_plot=True)

            print('Fine-tuned MobileSAM')
            annotated_img, loss = amg_predict(mobile_sam_model, SamAutomaticMaskGenerator, val_gt_masks, 'ft_MobileSAM', \
                                              val_img_path, mask_on_negative=True, show_plot=True)
            
            ft_validation_loss.append(annotated_image_orig_mobile_sam_loss)
            validation_loss.append(loss)

    return np.mean(ft_validation_loss), np.mean(validation_loss)


# In[57]:


def one_image_predict(image_masks, gt_masks, gt_bboxes, image_embedding, original_image_size, input_size, negative_mask, image, model):
    image_loss=[]
    mask_result = []
    total_area = 0.0
    ce_loss = 0.0
    for k in image_masks:   

                # process bboxes
                prompt_box = np.array(gt_bboxes[k])
                box = predictor.transform.apply_boxes(prompt_box, original_image_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                box_torch = box_torch[None, :]

                # process masks
                mask_input_torch = torch.as_tensor(gt_masks[k], dtype=torch.float, device=predictor.device).unsqueeze(0).unsqueeze(0)
                mask_input_torch = F.interpolate(mask_input_torch, size=(256, 256), mode='bilinear', align_corners=False)

                # '''
                # this is not in SAM. I added this, because the values in the mask are not only 0/1
                # not sure if it's correct ❗️❗️❗️
                # '''
                # mask_input_torch = (mask_input_torch>0) * 1.0

                # process coords and labels
                x_min, y_min, x_max, y_max = gt_bboxes[k]
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        
                point_coords = np.array([(gt_bboxes[k][2]+gt_bboxes[k][0])/2.0, (gt_bboxes[k][3]+gt_bboxes[k][1])/2.0])
                point_labels = np.array([1])
                point_coords = predictor.transform.apply_coords(point_coords, original_image_size)
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=predictor.device).unsqueeze(0)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

                # mask_copy = gt_masks[k].copy()
                # mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2BGR)
                # mask_copy = mask_copy * 255
                # print(x_min, y_min, x_max, y_max)
                # cv2.rectangle(mask_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # plt.imshow(mask_copy)
                # plt.plot((gt_bboxes[k][2]+gt_bboxes[k][0])/2.0, (gt_bboxes[k][3]+gt_bboxes[k][1])/2.0, 'ro')
                # plt.show()

                if epoch > 25:
                  box_torch=None
                    
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                  points=(coords_torch, labels_torch),
                  boxes=box_torch, #None, 
                  masks=None, #mask_input_torch,
                )

                del box_torch, coords_torch, labels_torch
                torch.cuda.empty_cache()

                # print(image_embedding.shape, model.prompt_encoder.get_dense_pe().shape, sparse_embeddings.shape, dense_embeddings.shape)
        
                low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, # True value works better for ambiguous prompts (single points)
                    # Also, multimask_output=False will set some gradients on 0
                )

                # low_res_masks_numpy = low_res_masks.detach().cpu().numpy()
                # iou_numpy = iou_predictions.detach().cpu().numpy()[0]
                # fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                # axs[0].imshow(low_res_masks_numpy[0][0])
                # axs[0].set_title(f'{iou_numpy[0]}', fontsize=12)
            
                # axs[1].imshow(low_res_masks_numpy[0][1])
                # axs[1].set_title(f'{iou_numpy[1]}', fontsize=12)

                # axs[2].imshow(low_res_masks_numpy[0][2])
                # axs[2].set_title(f'{iou_numpy[1]}', fontsize=12)
            
                # plt.show()
                # plt.close()
        
                downscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                # binary_mask = normalize(threshold(downscaled_masks, 0.0, 0))
                binary_mask = torch.sigmoid(downscaled_masks - model.mask_threshold)
                gt_mask_resized = torch.from_numpy(np.resize(gt_masks[k], (1, 1, gt_masks[k].shape[0], gt_masks[k].shape[1]))).to(device)
                gt_binary_mask = torch.as_tensor(gt_mask_resized>0, dtype=torch.float32) 
        
                numpy_gt_binary_mask = gt_binary_mask.contiguous().detach().cpu().numpy()

                # # mask = np.expand_dims(gt_binary_mask, axis=-1)
                # # mask_input_torch = torch.as_tensor(mask, dtype=torch.float, device=device)
                # # mask_input_torch = mask_input_torch / mask_input_torch.max()
                # input_image = torch.as_tensor(image, dtype=torch.float, device=device)
                # input_image = input_image / input_image.max()

                # # CE loss over mask
                # class_output = extended_model(mask_input_torch[0].permute(1,2,0) * input_image).unsqueeze(0) # also take the image pixels into account

                # # class_output = extended_model(binary_mask, iou_predictions).unsqueeze(0)
                # # print('class_output', class_output)
                # gt_class = torch.as_tensor([classes[k]], dtype=torch.long, device=device)
                # # find the index where the max value is in the class_output
                # # max_index = torch.argmax(class_output)
                # # print('class_output shape:', class_output, 'gt class', gt_class)
                # ce_loss += ce_loss_fn(class_output, gt_class)
                # # print('CE loss:', ce_loss)

                # print(iou_predictions)
                mask_result.append(binary_mask[0][0].detach().cpu())

                # plt.imshow(numpy_gt_binary_mask[0][0])
                # plt.show()
        
                # compute weighted dice loss (smaller weights on smaller objects)
                mask_area = np.sum(gt_masks[k])

                if mask_area > 0:
                  focal = focal_loss(downscaled_masks[0][0], gt_binary_mask)
                  dice = dice_loss(binary_mask, gt_binary_mask, negative_mask) 
                    
                  image_loss.append(mask_area * (20 * focal + dice )) # used in SAM paper
                  total_area += mask_area

                del binary_mask 
                del gt_mask_resized, numpy_gt_binary_mask 
                del low_res_masks, iou_predictions 
                torch.cuda.empty_cache()

    image_loss = torch.stack(image_loss)
    image_loss /= total_area
    image_loss = torch.mean(image_loss) * 100
    # if len(image_masks) > 0:
    #     ce_loss /= len(image_masks)
    # image_loss = (image_loss + ce_loss)/2.0
    
    # fig, axs = plt.subplots(1, 3, figsize=(40, 20))

    # axs[0].imshow(image)
    # # show_masks(mask_result, axs[0], random_color=False)
    # # axs[0].set_title(f'{idd.split(".")[0]}', fontsize=10)
    # axs[0].set_title(f'{k.split(".")[0]}', fontsize=40)
    # axs[0].axis('off')

    # axs[1].imshow(image)
    # show_masks(mask_result, axs[1], random_color=False)
    # axs[1].set_title('Predicted masks', fontsize=40)
    # axs[1].axis('off')
    
    # axs[2].imshow(image)
    # show_masks([gt_masks[k] for k in image_masks], axs[2], random_color=False)
    # axs[2].set_title('Ground truth masks', fontsize=40)
    # axs[2].axis('off')
    # # plt.savefig(f'./plots/pred_{k}.png', dpi=300)

    # plt.show()
    # plt.close()

    return image_loss


# In[31]:


import torch
import numpy as np
from tqdm import tqdm

def train_validate_step(model, dataloader, gt_masks, gt_bboxes, device, optimizer=None, mode='train'):
    assert mode in ['train', 'validate'], "Mode must be 'train' or 'validate'"
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    losses = []
    for inputs in tqdm(dataloader):
        batch_loss = 0.0
        batch_size = len(inputs['image'])
        for i in range(batch_size):
            image_masks = {k for k in gt_masks.keys() if k.startswith(inputs['image_id'][i])}
            idd = inputs['image_id'][i]
            input_image = inputs['image'][i].to(device)
            # print(input_image.mean(), input_image.min(), input_image.max())
            
            image = cv2.imread(input_dir+inputs['image_id'][i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            original_image_size = image.shape[:-1]
            input_size = (1024, 1024)
            # np_image = np.transpose(input_image[0].detach().cpu().numpy(), (1, 2, 0))
            # print(np.min(np_image), np.max(np_image))
            # plt.imshow(input_image[0][0].detach().cpu(), cmap='viridis')
            # plt.title(f'{inputs["image_id"][i]}.png')
            # plt.show()
            
            # IMAGE ENCODER
            image_embedding = model.image_encoder(input_image)
               
            # negative_mask has the size of the image
            negative_mask = np.where(image>0, True, False)
            negative_mask = torch.from_numpy(negative_mask)  
            negative_mask = negative_mask.permute(2, 0, 1)
            negative_mask = negative_mask[0]
            negative_mask = negative_mask.unsqueeze(0).unsqueeze(0)
            negative_mask = negative_mask.to(device)
            
            # RUN PREDICTION ON IMAGE
            if len(image_masks)>0:
                batch_loss += (one_image_predict(image_masks, gt_masks, gt_bboxes, image_embedding, original_image_size, input_size, negative_mask, image, model)/len(image_masks)) 
            else:
                print("❗️❗️❗️ Image with no annotations.", idd)
        if mode == 'train':
            for name, parameter in model.named_parameters():
                if parameter.grad is not None: 
                    grad_norm = parameter.grad.norm()
                    if grad_norm < 1e-8: 
                        print(f'❗️Layer {name} has vanishing gradients: {grad_norm}')
            # for name, parameter in model.named_parameters():
            # 	if parameter.grad is not None:
            # 		print(name, parameter.grad.abs().mean())
                    
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            del image_embedding, negative_mask, input_image, idd, image
            torch.cuda.empty_cache()

        losses.append(batch_loss.item() / batch_size)
            
    return np.mean(losses), model


# In[32]:


# torch.cuda.empty_cache()
predictor = SamPredictor(mobile_sam_model)


# In[33]:


train_losses = []
valid_losses = []
num_epochs = 50
best_valid_loss = float('inf')
n_epochs_stop = 5 + num_epochs/10

for epoch in range(num_epochs):

    # train
    epoch_loss, model = train_validate_step(mobile_sam_model, train_dataloader, train_gt_masks, train_bboxes, device, optimizer, mode='train')
    train_losses.append(epoch_loss)
    
    # validate
    with torch.no_grad():
        epoch_val_loss, model = train_validate_step(mobile_sam_model, val_dataloader, val_gt_masks, val_bboxes, device, optimizer, mode='validate')
        valid_losses.append(epoch_val_loss)
        
        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            best_model = model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping initiated.")
                early_stop = True
                break

    # Logging
    if use_wandb:
        wandb.log({'epoch training loss': epoch_loss, 'epoch validation loss': epoch_val_loss})

    print(f'EPOCH: {epoch}. Training loss: {epoch_loss}')
    print(f'EPOCH: {epoch}. Validation loss: {epoch_val_loss}.')

torch.save(best_model.state_dict(), f'ft_mobile_sam_final.pth')


# In[ ]:


print('epoch train losses:', epoch_loss)
print('epoch val losses:', epoch_val_loss)


# In[58]:


# import sys
# import PIL
# from PIL import Image

# sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-master/')
# import mobile_sam
# from mobile_sam import sam_model_registry as orig_mobile_sam_model_registry, \
#                        SamAutomaticMaskGenerator as orig_mobile_SamAutomaticMaskGenerator, \
#                        SamPredictor as orig_mobile_SamPredictor

# # orig_mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/XMM_OM_code_git/ft_mobile_sam_final.pth"
# orig_mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/MobileSAM-master/weights/mobile_sam.pt"
# orig_mobile_sam_model = orig_mobile_sam_model_registry["vit_t"](checkpoint=orig_mobile_sam_checkpoint)
# orig_mobile_sam_model.to(device);
# orig_mobile_sam_model.eval();

# mobile_sam_model.eval();
# predictor = orig_mobile_SamPredictor(mobile_sam_model)

# valid_losses = []
# for epoch in range(1):
#     with torch.no_grad():
#         epoch_val_loss = train_validate_step(orig_mobile_sam_model, val_dataloader, val_gt_masks, val_bboxes, device, optimizer, mode='validate')
#         valid_losses.append(epoch_val_loss)
#     print(f'EPOCH: {epoch}. Validation loss: {epoch_val_loss}.')


# In[49]:


# validate_model_AMG(mobile_sam_model, orig_mobile_sam_model) # fine-tuning the model on prompts and testing it on AMG is not funny, especially of the model was not fine-tuned on all data


# In[35]:


# import matplotlib.pyplot as plt

# # last run
# losses = [52.27958413326379, 40.61563884850705, 34.62209655299331, 32.76141236045144, 
#           33.59805725560044, 29.96991842443293, 28.418979529178504, 27.279344472018156, 
#           27.800395011901855, 28.727664427323774]

# # epoch_train_losses = [4.036007727172985, 3.0183080784686203, 2.458762989622174, 2.387210329064043, 2.6231899189226553, 2.43935736908025, 2.0195047118447045, 1.8985469161690056, 1.935919627895603, 1.9179415566580635, 1.741007660168074, 1.7309433288904497, 1.7882272414830855, 1.8295290135718012, 1.7947708214516247, 1.581464247270064, 1.6929189818246024, 1.563185892476664]
# # epoch val losses = [3.4105085817972816, 2.7414819908142087, 3.0362232716878252, 2.5541095511118574, 2.410716915130615, 2.6273675505320235, 2.604374787012736, 2.379702302614848, 2.5730778217315673, 2.4738360436757403, 2.5458037058512373, 2.659708590507507, 2.540311522483826, 3.1431035836537675, 2.3909687360127765, 2.6023658625284836, 2.5277215989430744, 2.45750465075175]

# epochs = list(range(1, len(losses) + 1))

# plt.figure(figsize=(10, 6))
# plt.plot(epochs, losses, label='Training loss\nbboxes + coords', marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=8)
# scatter_epochs = [8, 8]
# scatter_values = [41.602, 57.308]
# k=0
# scatter_labels = ['fine_tuned_MobileSAM', 'original_MobileSAM'] 
# for epoch, value, label in zip(scatter_epochs, scatter_values, scatter_labels):
#     if k<1:
#         plt.scatter(epoch, value, color='magenta', s=100, label='Validation loss',  marker='o',linestyle='-', zorder=5)
#         k+=1
#     else:
#         plt.scatter(epoch, value, color='magenta', s=100, zorder=5)
#     plt.text(epoch, value, f'  {label}', verticalalignment='bottom', color='magenta')

# plt.title('Loss Over 10 Epochs \nbatch_size=10, lr=1e-4')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.legend(loc='upper left')

# # Saving the plot with scatter points and labels to a file
# plt.savefig('loss_plot_with_scatter_and_labels.png', dpi=300)

# plt.show()


# In[36]:


# from IPython.display import display
# display(dot)


# In[37]:


# import json
# import graphviz

# json_file_path = './graph_0_summary_598a07b5e588b04c9de6.graph.json'

# with open(json_file_path, 'r') as file:
#     model_data = json.load(file)

# dot = graphviz.Digraph(comment='Model Visualization')

# for node in model_data['nodes']:
#     label = f"{node['name']}\n{node['class_name']}"
#     dot.node(str(node['id']), label=label)
# dot.render('output', view=False)


# In[ ]:


# plt.plot(list(range(len(losses))), losses)
# plt.title('Mean epoch loss \n mask with sigmoid')
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.savefig('loss_mask_sigmoid_training.png')
# plt.show()
# plt.close()