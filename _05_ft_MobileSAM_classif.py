# %% [markdown]
# # Set up

# %%
import os
import torch
print(torch.__version__)
print(torch.__file__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"

from torch.profiler import profile, record_function, ProfilerActivity

# import torch.autograd.profiler as profiler

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
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import copy
import time
from torch.cuda.amp import GradScaler, autocast

np.set_printoptions(precision=15)

torch.cuda.set_device(7)  # specifically set a GPU (this solves the CUDA out of memeory error when someone else allocates memory with "cuda" instead of a specific "cuda:idx", even when I alocate with "cuda:idx")

# Ensure deterministic behavior (cannot control everything though)
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# my utils
from importlib import reload
import dataset_utils
reload(dataset_utils)
from dataset_utils import *

import predictor_utils
reload(predictor_utils)
from predictor_utils import *

# %%
# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="GGtO5x2eJ77Wa0rLpQSt")
# project = rf.workspace("orij").project("xmm_om_images-contrast-512-v5")
# dataset = project.version(3).download("coco")

# %%
input_dir = '/workspace/raid/OM_DeepLearning/XMM_OM_code_git/xmm_om_images-contrast-512-v5-3/train/'
json_file_path = '/workspace/raid/OM_DeepLearning/XMM_OM_code_git/xmm_om_images-contrast-512-v5-3/train/_annotations.coco.json'

with open(json_file_path) as f:
    data = json.load(f)

ground_truth_masks, bbox_coords, classes, class_categories = get_coords_and_masks_from_json(input_dir, data) # type: ignore

# %%
class_categories, classes

# %%
image_paths_no_augm = [input_dir+img_data['file_name'] for img_data in data['images']]
len(image_paths_no_augm)

# %% [markdown]
# ## Augmentation
# 
# This algorithm performs augmentations and updates the negative masks in the case of a geometric transformations. Otherwise, it masks the result of a contrastive transformation given the mask of the initial image. Usually, the geometrical and contrastive transformations are not used simultaneously, as the notion of $<0$ pixels (negative mask) is lost.
# 
# **!! For augmentation, the bboxes are expected to be in the XYHW format, not XYXY format (used by SAM). However, the SAM AMG generated results are in the XYHW format (converted from XYXY).**
# 

# %%
import os
import glob

files = glob.glob(f'{input_dir}/*augm*')

for file in files:
    os.remove(file)

# %%
# https://albumentations.ai/docs/examples/showcase/
import os
import numpy as np
import cv2
from matplotlib import category, pyplot as plt
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

# define some augmentations
geometrical_augmentations = A.Compose([
    A.Flip(),
    A.RandomRotate90(),
    A.RandomSizedCrop((512 - 150, 512 - 50), 512, 512),
], bbox_params={'format':'coco', 'min_area': 0.1, 'min_visibility': 0.5, 'label_fields': ['category_id']}, p=1)

noise_blur_augmentations = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=1),
    A.GaussNoise(var_limit=(10.0, 30.0), p=1),
    A.ISONoise(p=0.8),
], bbox_params={'format':'coco', 'min_area': 0.01, 'min_visibility': 0.2, 'label_fields': ['category_id']}, p=1)

image_paths = []
for image_path in image_paths_no_augm:
    image_paths.append(image_path)
    # print(image_path)
    # image_ = cv2.imread(image_path)
    # image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    
    # masks = [value_i for key_i, value_i in ground_truth_masks.items() if image_path.split('/')[-1] in key_i]

    # image_size = (image_.shape[0], image_.shape[1])
    
    # # Enlarge the bounding boxes
    # bboxes_ = [enlarge_bbox(np.array([value_i[0], math.floor(value_i[1]),  math.floor(value_i[2] - value_i[0]), \
    #                                 math.floor(value_i[3] - value_i[1])]), 5, image_size) \
    #                                 for key_i, value_i in bbox_coords.items() if image_path.split('/')[-1] in key_i]   

    # label_ids = [classes[key_i] for key_i, value_i in bbox_coords.items() if image_path.split('/')[-1] in key_i]
    # # print('label_ids:', len(label_ids))
    
    # img_negative_mask = (image_>0).astype(int)

    # augmented1 = augment_and_show(geometrical_augmentations, image_, masks, bboxes_, label_ids, class_categories, show_=False)
    # new_image_negative_mask = (augmented1['image']>0).astype(int) # to mask the transform which is derived from the geometric transform
    
    # augmented3 = augment_and_show(noise_blur_augmentations, image_, masks, bboxes_, label_ids, class_categories, show_=False)
    
    # # mask the transform using the image negative mask
    # augmented3['image'] = augmented3['image'] * img_negative_mask
        
    # new_filename1 = image_path.replace('.'+image_path.split('.')[-1], '_augm1.jpg')
    # new_filename3 = image_path.replace('.'+image_path.split('.')[-1], '_augm3.jpg')

    # # print('aug masks1:', len(augmented1['masks']), 'bboxes:', len(augmented1['bboxes']))
    # # print('aug masks3:',len(augmented3['masks']), 'bboxes:', len(augmented3['bboxes']))

    # update_dataset_with_augms(augmented1, new_filename1, bbox_coords, ground_truth_masks, image_paths, classes)
    # update_dataset_with_augms(augmented3, new_filename3, bbox_coords, ground_truth_masks, image_paths, classes)

# %%
len(image_paths)

# %%
# augmented3['image'].shape, len(augmented3['bboxes']), len(augmented3['masks']) 

# %%
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

# %%
np.mean(means), np.mean(stds) 

# %%
np.std(means), np.std(stds)

# %% [markdown]
# ### Dataset split

# %%
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
    
training_size, test_size, val_size = (0.7, 0.2, 0.1)
splits = split_list(image_paths, [training_size, test_size, val_size])
training_image_paths, test_image_paths, val_image_paths = splits[0], splits[1], splits[2]

train_gt_masks, train_bboxes = create_dataset(training_image_paths, ground_truth_masks, bbox_coords)
test_gt_masks, test_bboxes = create_dataset(test_image_paths, ground_truth_masks, bbox_coords)
val_gt_masks, val_bboxes = create_dataset(val_image_paths, ground_truth_masks, bbox_coords)


del ground_truth_masks, bbox_coords 

# %%
len(training_image_paths), len(test_image_paths), len(val_image_paths)

# %%
image_paths[0]

# %%
image_id2 = image_paths[0].split('/')[-1]
image_masks_ids = [key for key in train_gt_masks.keys() if key.startswith(image_id2)]
image_ = cv2.imread(image_paths[0])
image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5,5))
plt.imshow(image_)
for name in image_masks_ids:
        show_box(train_bboxes[name], plt.gca())
        show_mask(train_gt_masks[name], plt.gca())
plt.axis('off')
plt.show()

# %% [markdown]
# ## 🚀 Prepare Mobile SAM Fine Tuning

# %%
import sys
import PIL
from PIL import Image

sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/')
import ft_mobile_sam
from ft_mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/weights/mobile_sam.pt"
device = "cuda:7" if torch.cuda.is_available() else "cpu"
print("device:", device)

mobile_sam_model = sam_model_registry["vit_t"](checkpoint=mobile_sam_checkpoint)
mobile_sam_model.to(device);
mobile_sam_model.eval(); #!!!

# %%
use_wandb = False

if use_wandb:
    from datetime import datetime
    # !pip install wandb
    # !wandb login --relogin
    import wandb
    wandb.login()
    run = wandb.init(project="OM_AI_v1", name=f"ft_MobileSAM {datetime.now()}")

    # wandb.watch(mobile_sam_model, log='all', log_graph=True)

# %% [markdown]
# # Convert the input images into a format SAM's internal functions expect.

# %%
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
        
        mobile_sam_model.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        mobile_sam_model.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        transformed_data = {}
        negative_mask = np.where(image > 0, True, False)
        negative_mask = torch.from_numpy(negative_mask)  
        negative_mask = negative_mask.permute(2, 0, 1)
        negative_mask = resize(negative_mask, [1024, 1024], antialias=True) 
        negative_mask = negative_mask.unsqueeze(0)
        # scales the image to 1024x1024 by longest side 
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, dtype=torch.float32, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        # normalization and padding
        input_image = mobile_sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        input_image[~negative_mask] = 0
        transformed_data['image'] = input_image
        transformed_data['input_size'] = input_size 
        transformed_data['image_id'] = k
        transformed_data['original_image_size'] = original_image_size
    
        return transformed_data

# %%
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths, bbox_coords, gt_masks, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.bbox_coords = bbox_coords
        self.gt_masks = gt_masks
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_id = self.image_paths[idx].split("/")[-1]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # plt.imshow(image)
        # plt.show()
        # plt.close()

        if self.transform is not None:
            return self.transform(image, img_id)
        else:
            print("❗️No transform❗️")
            return image
        
batch_size = 64
dataset = ImageDataset(training_image_paths, train_bboxes, train_gt_masks, transform_image) 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
torch.cuda.empty_cache() 

# print(torch.cuda.memory_summary(device=device, abbreviated=False))

# %%
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import torch.nn as nn

def dice_loss(pred, target, negative_mask = None, area=None, smooth = 1): 
    pred = pred.contiguous()
    target = target.contiguous()    
    
    if negative_mask is not None:
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

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
    return loss.mean()

# %%
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated()/(1024**2)) #MB
print(torch.cuda.memory_reserved()/(1024**2))

# %% [markdown]
# ## Model weights

# %%
for name, param in mobile_sam_model.named_parameters():
    # if "mask_decoder" in name:
    layers_to_fine_tune = ['mask_decoder.output_hypernetworks_mlps','mask_decoder.iou_prediction_head', 'mask_decoder.output_upscaling', 'mask_decoder.mask_tokens', 'mask_decoder.iou_token']
    # if any(s in name for s in layers_to_fine_tune): # or "image_encoder.patch_embed" in name:
    if False:
        param.requires_grad = True
    else:
        param.requires_grad = False

# %%
# def check_requires_grad(model, show=True):
#     for name, param in model.named_parameters():
#         if param.requires_grad and show:
#             print("✅ Param", name, " requires grad.")
#         elif param.requires_grad == False:
#             print("❌ Param", name, " doesn't require grad.")

# %%
# print(f"🚀 The model has {sum(p.numel() for p in mobile_sam_model.parameters() if p.requires_grad)} trainable parameters.\n")
# check_requires_grad(mobile_sam_model)

# %% [markdown]
# ## The classifier

# %%
import torch
import torch.nn as nn

# class ExtendedSAMModel(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super().__init__()
      
#         self.classifier = nn.Sequential(
#             nn.Linear(input_size, 512), # TODO: use nn.AdaptiveAvgPool2d layer, as it is more robust to image size changes
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, mask, iou_predictions):
#         # Flatten and concatenate mask and IOU predictions
#         flattened_mask = mask.view(-1)
#         iou_predictions_flat = iou_predictions.view(-1)
#         combined_features = torch.cat((flattened_mask, iou_predictions_flat), dim=0)

#         # Pass through the classifier
#         class_logits = self.classifier(combined_features)
#         return class_logits

# num_classes = len(class_categories.values())
# extended_model = ExtendedSAMModel(786435, num_classes).to(device) # TODO: this hard-coded value only works for 512x512 masks and 3 IoUs per mask !!!
# extended_model

class ExtendedSAMModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
      
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512), # TODO: use nn.AdaptiveAvgPool2d layer, as it is more robust to image size changes
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, mask):
        # Flatten and concatenate mask and IOU predictions
        flattened_mask = mask.view(-1)

        # Pass through the classifier
        class_logits = self.classifier(flattened_mask)
        return class_logits

num_classes = len(class_categories.values())
extended_model = ExtendedSAMModel(512*512*3, num_classes).to(device) # TODO: this hard-coded value only works for 512x512 masks and 3 IoUs per mask !!!
extended_model
# %% [markdown]
# ## Run fine tuning

# train the classfier alone with gt masks and classes

import torch
import torch.optim as optim
import torch.nn.functional as F

lr=3e-4
wd=0.01
optimizer = torch.optim.AdamW(extended_model.parameters(), lr=lr, weight_decay=wd)

ce_loss_fn = nn.CrossEntropyLoss()

scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-8) # not very helpful

num_epochs= 1
losses = []

# def validate_model_classif(batch_size):
#     dataset_val = ImageDataset(val_image_paths, val_bboxes, val_gt_masks, transform_image) 
#     dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

#     validation_loss = []
#     with torch.no_grad():
#         top1_accs = 0.0
#         top5_accs = 0.0
#         top1_accuracy = []
#         top5_accuracy = []
#         for inputs in tqdm(dataloader_val): # take image ids
#             image_loss = {}
#             batch_loss = 0.0
#             batch_size = min(len(inputs['image']), batch_size) # at the end, there's less than batch_size imgs in a batch (sometimes)

#             top1_batch_accs = 0.0
#             top5_batch_accs = 0.0

#             with autocast(): 
#                 for i in range(batch_size): 
#                     image_loss[i]=0.0
                    
#                     image_masks = {k for k in val_gt_masks.keys() if k.startswith(inputs['image_id'][i])}
#                     # input_image = inputs['image'][i].to(device)

#                     image = cv2.imread(input_dir+inputs['image_id'][i])
#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                     input_image = torch.as_tensor(image, dtype=torch.float, device=device)
#                     input_image = input_image/input_image.max()

#                     # RUN PREDICTION ON IMAGE
#                     for mask_id in image_masks:
#                         # Expand the mask dimensions to (x, y, 1)
#                         mask = np.expand_dims(val_gt_masks[mask_id], axis=-1)
#                         mask_input_torch = torch.as_tensor(mask, dtype=torch.float, device=device)
#                         mask_input_torch = mask_input_torch/mask_input_torch.max()

#                         # print(input_image.shape, mask_input_torch.shape)

#                         # Forward pass
#                         outputs = extended_model(mask_input_torch*input_image).unsqueeze(0) 
#                         # print((mask_input_torch*input_image).dtype, (mask_input_torch*input_image).min(), (mask_input_torch*input_image).max())
#                         # plt.imshow((mask_input_torch*input_image).detach().cpu().numpy())
#                         # plt.title(f'Predicted Class: {class_categories[torch.argmax(outputs).item()]}. Ground Truth Class: {class_categories[classes[mask_id]]}')
#                         # plt.show()
#                         # plt.close()
                    
#                         loss = F.cross_entropy(outputs, torch.tensor([classes[mask_id]]).to(device))
#                         image_loss[i] += loss

#                         # top-k accuracy
#                         top1_acc, top5_acc = topk_accuracy(outputs, torch.tensor([classes[mask_id]]).to(device), topk=(1, 5)) # percentage

#                         print(top1_acc, top5_acc)
#                         top1_batch_accs += top1_acc.item()
#                         top5_batch_accs += top5_acc.item()

#                     image_loss[i] = image_loss[i]/len(image_masks)
#                     top1_accs += (top1_batch_accs/len(image_masks))
#                     top5_accs += (top5_batch_accs/len(image_masks))

#                     del input_image

#                     batch_loss += image_loss[i]
#                     torch.cuda.empty_cache()
        
#             batch_loss /= batch_size
#             top1_accs /= batch_size
#             top5_accs /= batch_size 

#             validation_loss.append(batch_loss.item()) # type: ignore
#             top1_accuracy.append(top1_accs)
#             top5_accuracy.append(top5_accs)
#             del batch_loss, image_loss
#             torch.cuda.empty_cache()

#     return np.mean(validation_loss), np.mean(top1_accuracy), np.mean(top5_accuracy)

# %%
def topk_accuracy(output, target, topk=(1,)):

    print('output', output)
    print('target', target)
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        print('pred:', pred)
        pred = pred.t()
        print('target:', target.view(1, -1))
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        print('correct:', correct)
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0)) 
        return res

# %%

# val_losses = []
# for epoch in range(num_epochs):
#     top1_accs = 0.0
#     top5_accs = 0.0
#     epoch_losses = []
#     top1_accuracy = []
#     top5_accuracy = []
#     for inputs in tqdm(dataloader): # take image ids
#         image_loss = {}
#         batch_loss = 0.0
#         batch_size = min(len(inputs['image']), batch_size) # at the end, there's less than batch_size imgs in a batch (sometimes)

#         top1_batch_accs = 0.0
#         top5_batch_accs = 0.0

#         with autocast(): 
#             for i in range(batch_size): 
#                 image_loss[i]=0.0
                
#                 image_masks = {k for k in train_gt_masks.keys() if k.startswith(inputs['image_id'][i])}
#                 # input_image = inputs['image'][i].to(device)

#                 image = cv2.imread(input_dir+inputs['image_id'][i])
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 input_image = torch.as_tensor(image, dtype=torch.float, device=device)
#                 input_image = input_image/input_image.max()

#                 # RUN PREDICTION ON IMAGE
#                 for mask_id in image_masks:
#                     # Expand the mask dimensions to (x, y, 1)
#                     mask = np.expand_dims(train_gt_masks[mask_id], axis=-1)
#                     mask_input_torch = torch.as_tensor(mask, dtype=torch.float, device=device)
#                     mask_input_torch = mask_input_torch/mask_input_torch.max()

#                     # print(input_image.shape, mask_input_torch.shape)

#                     # Forward pass
#                     outputs = extended_model(mask_input_torch*input_image).unsqueeze(0) 
#                     # print((mask_input_torch*input_image).dtype, (mask_input_torch*input_image).min(), (mask_input_torch*input_image).max())
#                     # plt.imshow((mask_input_torch*input_image).detach().cpu().numpy())
#                     # plt.title(f'Predicted Class: {class_categories[torch.argmax(outputs).item()]}. Ground Truth Class: {class_categories[classes[mask_id]]}')
#                     # plt.show()
#                     # plt.close()
                   
#                     loss = F.cross_entropy(outputs, torch.tensor([classes[mask_id]]).to(device))
#                     image_loss[i] += loss

#                     # top-k accuracy
#                     top1_acc, top5_acc = topk_accuracy(outputs, torch.tensor([classes[mask_id]]).to(device), topk=(1, 5)) # percentage
#                     top1_batch_accs += top1_acc.item()
#                     top5_batch_accs += top5_acc.item()

#                 image_loss[i] = image_loss[i]/len(image_masks)
#                 top1_batch_accs = top1_batch_accs/len(image_masks)
#                 top5_batch_accs = top5_batch_accs/len(image_masks)

#                 top1_accs += top1_batch_accs
#                 top5_accs += top5_batch_accs

#                 del input_image

#                 batch_loss += image_loss[i]
#                 torch.cuda.empty_cache()

#         batch_loss /= batch_size
#         top1_accs /= batch_size
#         top5_accs /= batch_size 

#         optimizer.zero_grad()
#         batch_loss.backward() # type: ignore
#         optimizer.step()

#         epoch_losses.append(batch_loss.item()) # type: ignore
#         top1_accuracy.append(top1_accs)
#         top5_accuracy.append(top5_accs)
#         del image_loss
#         torch.cuda.empty_cache()

#     losses.append(np.mean(epoch_losses))

#     # validation loss
#     extended_model.eval();
#     val_loss, val_top1_acc, val_top5_acc = validate_model_classif(batch_size)
#     val_losses.append(val_loss)
#     extended_model.train();

#     torch.cuda.empty_cache()

#     if use_wandb:
#         wandb.log({'epoch classif train loss': np.mean(epoch_losses)})
#         wandb.log({'epoch classif val loss': val_loss})
#     print(f'EPOCH: {epoch}. Training classes loss: {np.mean(epoch_losses)}.Validation classes loss: {np.mean(val_losses)} ')
#     print(f'Top-1 accuracy: {np.mean(top1_accuracy)}. Top-5 accuracy: {np.mean(top5_accuracy)}. Validation Top-1 accuracy: {val_top1_acc}. Validation Top-5 accuracy: {val_top5_acc} ')
    
# torch.save(extended_model.state_dict(), 'classif_checkpoint.pth')

# if use_wandb:
#     wandb.run.summary["batch_size"] = batch_size
#     wandb.run.summary["num_epochs"] = num_epochs
#     wandb.run.summary["learning rate"] = lr
#     wandb.run.summary["used area for DICE loss"] = 'YES'
#     run.finish()

# %%
def process_batch(inputs, model, optimizer, is_training, gt_masks, classes, class_categories, device):
    image_loss = {}
    batch_loss = 0.0
    batch_size = len(inputs['image'])

    top1_batch_accs = 0.0
    top5_batch_accs = 0.0

    with torch.set_grad_enabled(is_training):
        with autocast():
            for i in range(batch_size):
                image_loss[i] = 0.0

                image_masks = {k for k in gt_masks.keys() if k.startswith(inputs['image_id'][i])}

                image = cv2.imread(input_dir + inputs['image_id'][i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_image = torch.as_tensor(image, dtype=torch.float, device=device)
                input_image = input_image / input_image.max()

                for mask_id in image_masks:
                    mask = np.expand_dims(gt_masks[mask_id], axis=-1)
                    mask_input_torch = torch.as_tensor(mask, dtype=torch.float, device=device)
                    mask_input_torch = mask_input_torch / mask_input_torch.max()

                    outputs = model(mask_input_torch * input_image).unsqueeze(0)
                    loss = F.cross_entropy(outputs, torch.tensor([classes[mask_id]]).to(device))
                    image_loss[i] += loss

                    top1_acc, top5_acc = topk_accuracy(outputs, torch.tensor([classes[mask_id]]).to(device), topk=(1, 5))
                    top1_batch_accs += top1_acc.item()
                    top5_batch_accs += top5_acc.item()

                image_loss[i] = image_loss[i] / len(image_masks)
                top1_batch_accs = top1_batch_accs / len(image_masks)
                top5_batch_accs = top5_batch_accs / len(image_masks)

                batch_loss += image_loss[i]
                torch.cuda.empty_cache()

        batch_loss /= batch_size
        top1_batch_accs /= batch_size
        top5_batch_accs /= batch_size

        if is_training:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    return batch_loss.item(), top1_batch_accs, top5_batch_accs
# %%
dataset_val = ImageDataset(val_image_paths, val_bboxes, val_gt_masks, transform_image) 
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

epoch_train_losses = []
epoch_val_losses = []
top1_train_accuracy = []
top5_train_accuracy = []
top1_val_accuracy = []
top5_val_accuracy = []

# Training and Validation Loops
for epoch in range(num_epochs):
    extended_model.train()
    for inputs in tqdm(dataloader):
        train_loss, train_top1_acc, train_top5_acc = process_batch(inputs, model, optimizer, True, train_gt_masks, classes, class_categories, device)
    
        epoch_train_losses.append(train_loss)
        top1_train_accuracy.append(train_top1_acc)
        top5_train_accuracy.append(train_top5_acc)

    if use_wandb:
        wandb.log({'epoch classif train loss': np.mean(epoch_train_losses)})
        wandb.log({'epoch top-1 acc': np.mean(top1_train_accuracy)})
        wandb.log({'epoch top-5 acc': np.mean(top5_train_accuracy)})

    print(f'EPOCH: {epoch}. Training classes loss: {np.mean(epoch_train_losses)}.Top-1 accuracy: {np.mean(top1_train_accuracy)}. Top-5 accuracy: {np.mean(top5_train_accuracy)}')

    extended_model.eval()
    with torch.no_grad():
        for inputs in tqdm(dataloader_val):
            val_loss, val_top1_acc, val_top5_acc = process_batch(inputs, model, None, False, val_gt_masks, classes, class_categories, device)

            epoch_val_losses.append(val_loss)
            top1_val_accuracy.append(val_top1_acc)
            top5_val_accuracy.append(val_top5_acc)

        if use_wandb:
            wandb.log({'epoch classif val loss': np.mean(epoch_val_losses)})
            wandb.log({'epoch val top-1 acc': np.mean(top1_val_accuracy)})
            wandb.log({'epoch val top-5 acc': np.mean(top5_val_accuracy)})

        print(f'EPOCH: {epoch}. Validation classes loss: {np.mean(epoch_val_losses)}. Top-1 accuracy: {np.mean(top1_val_accuracy)}. Top-5 accuracy: {np.mean(top5_val_accuracy)}')

# %%
plt.plot(list(range(len(losses))), losses, label='Training Loss')
plt.plot(list(range(len(val_losses))), val_losses, label='Validation Loss')
plt.title('Mean epoch loss \n mask with sigmoid')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./plots/loss_mask_valid_points_prompt.png')
plt.show()

# # %%

# # %% [markdown]
# # **AutomaticMixedPrecision**
    
# scaler = torch.cuda.amp.GradScaler()

# # %%
# len(train_gt_masks.keys())

# # # %%
# # def validate_model():
# #     validation_loss = []
# #     total_area= 0.0
# #     with torch.no_grad():  
# #         for k in val_gt_masks.keys():
# #             prompt_box = np.array(val_bboxes[k])
# #             box = predictor.transform.apply_boxes(prompt_box, original_image_size)
# #             box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
# #             box_torch = box_torch[None, :]
            
# #             mask_input_torch = torch.as_tensor(val_gt_masks[k], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

# #             point_coords = np.array([(val_bboxes[k][2]+val_bboxes[k][0])/2.0, (val_bboxes[k][3]+val_bboxes[k][1])/2.0])
# #             point_labels = np.array([1])
# #             point_coords = predictor.transform.apply_coords(point_coords, original_image_size)
# #             coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device).unsqueeze(0)
# #             labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
# #             coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

# #             sparse_embeddings, dense_embeddings = mobile_sam_model.prompt_encoder(
# #               points=(coords_torch,labels_torch),
# #               boxes=box_torch,
# #               masks=None, 
# #             )

# #             low_res_masks, iou_predictions = mobile_sam_model.mask_decoder(
# #             image_embeddings=image_embedding,
# #             image_pe=mobile_sam_model.prompt_encoder.get_dense_pe(),
# #             sparse_prompt_embeddings=sparse_embeddings,
# #             dense_prompt_embeddings=dense_embeddings,
# #             multimask_output=True,
# #             )
            
# #             downscaled_masks = mobile_sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
# #             binary_mask = torch.sigmoid(downscaled_masks - mobile_sam_model.mask_threshold)
            
# #             numpy_binary_mask = binary_mask.detach().cpu().numpy()
# #             gt_mask_resized = torch.from_numpy(np.resize(val_gt_masks[k], (1, 1, val_gt_masks[k].shape[0], val_gt_masks[k].shape[1]))).to(device)
# #             gt_binary_mask = torch.as_tensor(gt_mask_resized >0, dtype=torch.float32) 
# #             numpy_gt_binary_mask = gt_binary_mask.contiguous().detach().cpu().numpy()
            
# #             mask_area = np.sum(val_gt_masks[k])
# #             validation_loss.append(dice_loss(binary_mask, gt_binary_mask, negative_mask, mask_area) * mask_area)
# #             total_area += mask_area

# #             # print(f'val one image{k}: Allocated memory:', torch.cuda.memory_allocated()/(1024**2), 'MB. Reserved memory:', torch.cuda.memory_reserved()/(1024**2), 'MB')
            
# #             torch.cuda.empty_cache()
            
# #             # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            
# #             # # Ground truth mask
# #             # axs[0].imshow(numpy_gt_binary_mask[0, 0], cmap='viridis')
# #             # axs[0].set_title('Ground Truth Mask')
            
# #             # # Predicted mask
#     #          # axs[1].imshow(numpy_binary_mask[0, 0], cmap='viridis')
#     #         # axs[1].set_title('Predicted Mask')

#     #         # plt.show()
#     #         # plt.close()

#     # validation_loss = torch.sum(validation_loss)
#     # validation_loss /= total_area

#     # return validation_loss

# import predictor_utils
# reload(predictor_utils)
# from predictor_utils import *

# def validate_model_AMG(ft_mobile_sam_model, mobile_sam_model_orig):
#     validation_loss = []

#     for val_img_path in val_image_paths:
#         with torch.no_grad():

#             print('Original MobileSAM')
#             annotated_image_orig_mobile_sam, annotated_image_orig_mobile_sam_loss = amg_predict(mobile_sam_model_orig, orig_mobile_SamAutomaticMaskGenerator, \
#                                                        val_gt_masks, 'ft_MobileSAM', val_img_path, mask_on_negative=False, show_plot=True)

#             print('Fine-tuned MobileSAM')
#             annotated_img, loss = amg_predict(mobile_sam_model, SamAutomaticMaskGenerator, val_gt_masks, 'ft_MobileSAM', \
#                                               val_img_path, mask_on_negative=True, show_plot=True)
#             # print('validation loss =', loss)
#             validation_loss.append(loss)

#     return np.mean(validation_loss)

# # %%
# def one_image_predict(image_masks):
#     image_loss=[]
#     mask_result = []
#     total_area = 0.0
#     ce_loss = 0.0
#     for k in image_masks: 
#         # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True, with_modules=True) as prof: # needs cudnn 
#         # with profiler.profile(profile_memory=True, use_cuda=True, record_shapes=True) as prof: # this needs pytorch build from Git for TORCH_USE_CUDA_DSA=1
#         # with torch.autograd.profiler.profile() as prof:
#         # if True:
#                 # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True, with_modules=True) as prof:
#                 if True:
#                     # process bboxes
#                     prompt_box = np.array(train_bboxes[k])
#                     box = predictor.transform.apply_boxes(prompt_box, original_image_size)
#                     box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
#                     box_torch = box_torch[None, :]

#                 # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))

#                 # process masks
#                 mask_input_torch = torch.as_tensor(train_gt_masks[k], dtype=torch.float, device=predictor.device).unsqueeze(0).unsqueeze(0)

#                 # process coords and labels
#                 x_min, y_min, x_max, y_max = train_bboxes[k]
#                 x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        
#                 point_coords = np.array([(train_bboxes[k][2]+train_bboxes[k][0])/2.0, (train_bboxes[k][3]+train_bboxes[k][1])/2.0])
#                 point_labels = np.array([1])
#                 point_coords = predictor.transform.apply_coords(point_coords, original_image_size)
#                 coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=predictor.device).unsqueeze(0)
#                 labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
#                 coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

#                 # mask_copy = train_gt_masks[k].copy()
#                 # mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2BGR)
#                 # mask_copy = mask_copy * 255
#                 # print(x_min, y_min, x_max, y_max)
#                 # cv2.rectangle(mask_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 # plt.imshow(mask_copy)
#                 # plt.plot((train_bboxes[k][2]+train_bboxes[k][0])/2.0, (train_bboxes[k][3]+train_bboxes[k][1])/2.0, 'ro')
#                 # plt.show()

#                 # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True, with_modules=True) as prof:
#                 if True:
#                     sparse_embeddings, dense_embeddings = mobile_sam_model.prompt_encoder(
#                     points=(coords_torch, labels_torch),
#                     boxes=box_torch, #None,
#                     masks=None,
#                     )
#                 # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))

#                 # print(box_torch.shape, mask_input_torch.shape)

#                 del box_torch, mask_input_torch, coords_torch, labels_torch
#                 torch.cuda.empty_cache()

#                 # print(image_embedding.shape, mobile_sam_model.prompt_encoder.get_dense_pe().shape, sparse_embeddings.shape, dense_embeddings.shape)
#                 # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True, with_modules=True) as prof:
#                 if True: 
#                     low_res_masks, iou_predictions = mobile_sam_model.mask_decoder(
#                     image_embeddings=image_embedding,
#                     image_pe=mobile_sam_model.prompt_encoder.get_dense_pe(),
#                     sparse_prompt_embeddings=sparse_embeddings,
#                     dense_prompt_embeddings=dense_embeddings,
#                     multimask_output=True, # this works better for ambiguous prompts (single points)
#                     )
#                 # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))
        
#                 # print(low_res_masks.shape)

#                 downscaled_masks = mobile_sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
#                 # binary_mask = normalize(threshold(downscaled_masks, 0.0, 0))
#                 binary_mask = torch.sigmoid(downscaled_masks - mobile_sam_model.mask_threshold)
#                 gt_mask_resized = torch.from_numpy(np.resize(train_gt_masks[k], (1, 1, train_gt_masks[k].shape[0], train_gt_masks[k].shape[1]))).to(device)
#                 gt_binary_mask = torch.as_tensor(gt_mask_resized>0, dtype=torch.float32) 
#                 numpy_gt_binary_mask = gt_binary_mask.contiguous().detach().cpu().numpy()

#                 # CE loss over mask
#                 flattened_mask = binary_mask.view(-1)
#                 iou_predictions_flat = iou_predictions.view(-1)
#                 combined_features = torch.cat((flattened_mask, iou_predictions_flat), dim=0)
#                 # print(len(combined_features))
#                 class_output = extended_model(binary_mask, iou_predictions).unsqueeze(0)
#                 # print('class_output', class_output)
#                 gt_class = torch.as_tensor([classes[k]], dtype=torch.long, device=device)
#                 # find the index where the max value is in the class_output
#                 # max_index = torch.argmax(class_output)
#                 # print('class_output shape:', class_output, 'gt class', gt_class)
#                 ce_loss += ce_loss_fn(class_output, gt_class)

#                 # print('CE loss:', ce_loss_fn(class_output, gt_class))

#                 # print(iou_predictions)
#                 mask_result.append(binary_mask[0][0].detach().cpu())

#                 # plt.imshow(numpy_gt_binary_mask[0][0])
#                 # plt.show()
        
#                 # compute weighted dice loss
#                 # print(train_gt_masks[k].shape,np.sum(train_gt_masks[k]))
#                 mask_area = np.sum(train_gt_masks[k])

#                 if mask_area > 0:
#                   image_loss.append(dice_loss(binary_mask, gt_binary_mask, negative_mask, mask_area) * mask_area)
#                   total_area += mask_area

#                 del binary_mask 
#                 del gt_mask_resized, numpy_gt_binary_mask 
#                 del low_res_masks, iou_predictions 
#                 del downscaled_masks, gt_binary_mask, gt_class, class_output
        
#                 torch.cuda.empty_cache()
#                 # print('one mask', k, 'done')
#                 # # print(f'one image{k}: Allocated memory:', torch.cuda.memory_allocated()/(1024**2), 'MB. Reserved memory:', torch.cuda.memory_reserved()/(1024**2), 'MB')
#                 import sys
#                 sys.exit()
#         # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))

#     image_loss = torch.stack(image_loss)
#     image_loss = torch.sum(image_loss)
#     image_loss /= total_area
#     if len(image_masks) > 0:
#         ce_loss /= len(image_masks)
#     image_loss = (image_loss + ce_loss)/2.0
#     # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
    
#     # plt.figure(figsize=(10, 5))
#     # fig, axs = plt.subplots(1, 2)

#     # axs[0].imshow(image)
#     # show_masks(mask_result, axs[0])
#     # axs[0].set_title('Predicted')

#     # axs[1].imshow(image)
#     # show_masks([train_gt_masks[k] for k in image_masks], axs[1])
#     # axs[1].set_title('Ground truth')
    
#     # plt.show()
#     # plt.close()
#     # print('image loss:', image_loss)
#     return image_loss

# # %%

# num_epochs = 1
# losses = []
# valid_losses = []
# valid_bboxes_losses = []
# predictor = SamPredictor(mobile_sam_model)

# for epoch in range(num_epochs):
	
#     epoch_losses = []
#     for inputs in tqdm(dataloader): # take image ids
#         image_loss = {}
#         batch_loss = 0.0
#         batch_size = min(len(inputs['image']), batch_size) # at the end, there's less than batch_size imgs in a batch (sometimes)

#         with autocast(): 
#             for i in range(batch_size): 
#                     image_loss[i]=0.0
                    
#                     image_masks = {k for k in train_gt_masks.keys() if k.startswith(inputs['image_id'][i])}
#                     input_image = inputs['image'][i].to(device)

#                     image = cv2.imread(input_dir+inputs['image_id'][i])
#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#                     original_image_size = image.shape[:-1]
#                     input_size = (1024, 1024)
#                     # np_image = np.transpose(input_image[0].detach().cpu().numpy(), (1, 2, 0))
#                     # print(np.min(np_image), np.max(np_image))
#                     # plt.imshow(input_image[0][0].detach().cpu(), cmap='viridis')
#                     # plt.title(f'{inputs["image_id"][i]}.png')
#                     # plt.show()

#                     # IMAGE ENCODER
#                     image_embedding = mobile_sam_model.image_encoder(input_image)

#                     del input_image

#                     # negative_mask has the size of the image
#                     negative_mask = np.where(image>0, True, False)
#                     negative_mask = torch.from_numpy(negative_mask)  
#                     negative_mask = negative_mask.permute(2, 0, 1)
#                     negative_mask = negative_mask[0]
#                     negative_mask = negative_mask.unsqueeze(0).unsqueeze(0)
#                     negative_mask = negative_mask.to(device)
#                     # del image

#                     # RUN PREDICTION ON IMAGE
#                     image_loss[i] = one_image_predict(image_masks)
                    
#                     batch_loss += image_loss[i]
#                     del image_embedding, negative_mask 
#                     torch.cuda.empty_cache()
        
#         batch_loss /= batch_size
#         optimizer.zero_grad()

#         # AMP
#         # scaler.scale(batch_loss).backward()
#         batch_loss.backward()
#         # scaler.step(optimizer)
#         optimizer.step()
#         # scaler.update()

#         epoch_losses.append(batch_loss.item())
#         torch.cuda.empty_cache()
#         del batch_loss, image_loss
#         # print('after emptying cache: Allocated memory:', torch.cuda.memory_allocated()/(1024**2), 'MB. Reserved memory:', torch.cuda.memory_reserved()/(1024**2), 'MB')
#         print(epoch_losses)

#     # validation loss
#     losses.append(np.mean(epoch_losses))
#     # mobile_sam_model.eval();
#     # # valid_losses.append(validate_model_AMG())
#     # # valid_bboxes_losses.append(validate_model())
#     # mobile_sam_model.train();
#     torch.cuda.empty_cache()

#     if use_wandb:
#         wandb.log({'epoch training loss': np.mean(epoch_losses)})
#         wandb.log({'epoch validation loss': np.mean(valid_bboxes_losses)})
#     # print(f'EPOCH: {epoch}. Training loss: {np.mean(epoch_losses)}. Validation AMG loss: {np.mean(valid_losses)}. Validation bboxes loss: {np.mean(valid_bboxes_losses)} ')
#     print(f'EPOCH: {epoch}. Training loss: {np.mean(epoch_losses)}.Validation bboxes loss: {np.mean(valid_bboxes_losses)} ')
#     # print(f'EPOCH: {epoch}. Training loss: {np.mean(epoch_losses)}.')
#     if  epoch%8 == 0:
#         torch.save(mobile_sam_model.state_dict(), f'mobile_sam_model_checkpoint{epoch}.pth')  

# torch.save(mobile_sam_model.state_dict(), 'mobile_sam_model_checkpoint.pth')
# torch.save(extended_model.state_dict(), 'classif_checkpoint.pth')

# print('Training losses:', losses)
# print('valid_bboxes_losses losses:', valid_bboxes_losses)
# # print('Val losses:', valid_losses)

# if use_wandb:
#     wandb.run.summary["batch_size"] = batch_size
#     wandb.run.summary["num_epochs"] = num_epochs
#     wandb.run.summary["learning rate"] = lr
#     wandb.run.summary["used area for DICE loss"] = 'YES'
#     run.finish()

# # %% [markdown]
# # **original MobileSAM checkpoint**

# # %%
# import sys
# sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-master/')
# import mobile_sam
# from mobile_sam import sam_model_registry as orig_mobile_sam_registry, \
# SamAutomaticMaskGenerator as orig_mobile_SamAutomaticMaskGenerator, \
# SamPredictor as orig_mobile_SamPredictor

# orig_mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/MobileSAM-master/weights/mobile_sam.pt"
# print("device:", device)

# mobile_sam_model_orig = orig_mobile_sam_registry["vit_t" ](checkpoint=orig_mobile_sam_checkpoint)
# mobile_sam_model_orig.to(device);
# mobile_sam_model_orig.eval();

# # %%
# # validate_model_AMG(mobile_sam_model, mobile_sam_model_orig)

# # %%
# print('valid_losses', valid_losses)
# print('valid_bboxes_losses', valid_bboxes_losses)

# # %%
# # check the GPU memory after training
# torch.cuda.empty_cache()
# print(torch.cuda.memory_allocated()/(1024**2)) #MB
# print(torch.cuda.memory_reserved()/(1024**2))

# # %%
# plt.plot(list(range(len(losses))), losses, label='Training Loss')
# plt.plot(list(range(len(valid_bboxes_losses))), valid_bboxes_losses, label='Validation Loss')
# plt.title('Mean epoch loss \n mask with sigmoid')
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('./plots/loss_mask_valid_points_prompt.png')
# plt.show()

# # %%
# # plt.plot(list(range(len(losses))), losses)
# # plt.title('Mean epoch loss \n mask with sigmoid')
# # plt.xlabel('Epoch Number')
# # plt.ylabel('Loss')
# # plt.savefig('loss_mask_sigmoid_training.png')
# # plt.show()
# # plt.close()

# # %%


# # %%


# # %%



