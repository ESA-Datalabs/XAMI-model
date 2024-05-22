# In[1]:
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3" # replace with the GPU IDs that are available
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"

from datetime import datetime
import torch
import numpy as np
import json
import cv2
from pycocotools import mask as maskUtils
np.set_printoptions(precision=9)
import albumentations as A
import matplotlib.pyplot as plt

# to help with reproducibility
seed=0
import torch.backends.cudnn as cudnn 
np.random.seed(seed) 
torch.manual_seed(seed) 
cudnn.benchmark, cudnn.deterministic = False, True

# Append project path when running in CLI
# Otherwise, the project path is already in the sys.path
# relative_project_path = os.path.join(sys.path[0], '../')
relative_project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(relative_project_path)
print('Project path:', relative_project_path)
from dataset import dataset_utils
from sam_predictor import load_dataset, astro_sam #, residualAttentionBlock

if len(sys.argv)==3:
    kfold_iter = int(sys.argv[1])
    device_id = int(sys.argv[2])
if len(sys.argv)==1:
    kfold_iter = 0
    device_id = 0
    
lr=3e-4
wd=0.0005
wandb_track=True
num_epochs=60
use_lr_warmup=True
n_epochs_stop = 15 # Early stopping
work_dir = './output_sam'
# torch.cuda.set_device(device_id)
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

# this variable is the original batch size
# the effective batch size will be: original_batch_size * (number of augmentations + 1) 
# if applying augmentations
# high values (e.g., 8, 16) may lead to OOM errors
batch_size=4

print("DEVICE", device)

# Load checkpoints
mobile_sam_dir = os.getcwd()+'/../mobile_sam/'
mobile_sam_checkpoint = mobile_sam_dir+"weights/mobile_sam.pt"

if not os.path.exists(work_dir):
	os.makedirs(work_dir)
else:
    print(f"Output directory {work_dir} already exists.")
    
# Dataset split

input_dir = f'../xami_dataset/'
train_dir = input_dir+f'train/'
valid_dir = input_dir+f'valid/'
json_train_path, json_valid_path = train_dir+'_annotations.coco.json', valid_dir+'_annotations.coco.json'

with open(json_train_path) as f1, open(json_valid_path) as f2:
    train_data_in = json.load(f1)
    valid_data_in = json.load(f2)

training_image_paths = [train_dir+image['file_name'] for image in train_data_in['images']]
val_image_paths = [valid_dir+image['file_name'] for image in valid_data_in['images']]

train_data = dataset_utils.load_json(json_train_path)
valid_data = dataset_utils.load_json(json_valid_path)
    
train_gt_masks, train_bboxes, train_classes, train_class_categories = dataset_utils.get_coords_and_masks_from_json(
    train_dir, 
    train_data) # type: ignore
val_gt_masks, val_bboxes, val_classes, val_class_categories = dataset_utils.get_coords_and_masks_from_json(
    valid_dir, 
    valid_data) # type: ignore

train_data_in['categories']

# In[10]:

# Visualize some training images

# for one_path in training_image_paths[10:15]:
#     image_id2 = one_path.split('/')[-1]
#     print(image_id2)
#     image_masks_ids = [key for key in train_gt_masks.keys() if key.startswith(image_id2)]
#     image_ = cv2.imread(one_path)
#     image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(8,8))
#     plt.imshow(image_)
#     for name in image_masks_ids:
#         dataset_utils.show_box(train_bboxes[name], plt.gca())
#         dataset_utils.show_mask(maskUtils.decode(train_gt_masks[name]), plt.gca())
#     plt.axis('off')
#     plt.show()
#     plt.close()

# 🚀 Prepare Mobile SAM Fine Tuning

# In[12]:

import sys
sys.path.append(mobile_sam_dir)
from mobile_sam import sam_model_registry, SamPredictor#, build_efficientvit_l2_encoder

# Segment Anything Model
model = sam_model_registry["vit_t"](checkpoint=mobile_sam_checkpoint)
model.to(device);
predictor = SamPredictor(model)

# # Residual Attention Block
# residualAttentionBlock = residualAttentionBlock.ResidualAttentionBlock(d_model=256, n_head=8, mlp_ratio=4.0).to(device)
# for name, param in residualAttentionBlock.named_parameters():
# 	param.requires_grad = True # this is usually already true

## setup the AstroSAM model
astrosam_model = astro_sam.AstroSAM(model, device, predictor) #, residualAttentionBlock)

if wandb_track:
    # !pip install wandb
    # !wandb login --relogin
    import wandb
    wandb.login()
    run = wandb.init(project="sam", name=f"sam_{kfold_iter}_{datetime.now()}")
    wandb.watch(astrosam_model.model, log='all', log_graph=True)

## Convert the input images into a format SAM's internal functions expect.
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader

# Dataset Loaders and Pre-processing
transform = ResizeLongestSide(astrosam_model.model.image_encoder.img_size)
train_set = load_dataset.ImageDataset(training_image_paths, astrosam_model.model, transform, device) 
val_set =  load_dataset.ImageDataset(val_image_paths, astrosam_model.model, transform, device) 
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Optimizer
for name, param in astrosam_model.model.named_parameters():
    if 'mask_decoder' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

parameters_to_optimize = [param for param in astrosam_model.model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(parameters_to_optimize, lr=lr, weight_decay=wd) 

# Scheduler
scheduler=None

if use_lr_warmup:
    initial_lr = lr
    final_lr = 6e-5
    total_steps = 16  # total steps over which the learning rate should decrease
    lr_decrement = (initial_lr - final_lr) / total_steps

    def lr_lambda(current_step):
        if current_step < total_steps:
            return 1 - current_step * lr_decrement / initial_lr
        return final_lr / initial_lr
		
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Model weights
print(f"🚀  The SAM model has {sum(p.numel() for p in astrosam_model.model.parameters() if p.requires_grad)} trainable parameters.\n")
# print(f"🚀 The ResAttnBlock has {sum(p.numel() for p in astrosam_model.residualAttentionBlock.parameters() if p.requires_grad)} trainable parameters.\n")
# predictor_utils.check_requires_grad(astrosam_model.model)

# Run Training
train_losses = []
valid_losses = []
best_valid_loss = float('inf')

# # Augmentations
# geometrical_augmentations = A.Compose([
#         A.Flip(p=0.8),
#         A.RandomRotate90(p=0.7),
#         A.RandomSizedCrop((512 - 20, 512 - 20), 512, 512, p=0.8),
#     ], bbox_params={'format':'coco', 'label_fields': ['category_id']}, p=1)
    
# noise_blur_augmentations = A.Compose([
#         A.GaussianBlur(blur_limit=(3, 3), p=0.9),
#         A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
#         A.ISONoise(p=0.7),
#     ], bbox_params={'format':'coco', 'label_fields': ['category_id']}, p=1)

combined_augmentations = A.Compose([
    # Geometric transformations 
    A.Flip(p=0.5),  
    A.RandomRotate90(p=0.5),  
    A.RandomSizedCrop((492, 492), 512, 512, p=0.6),  

    # Noise and blur transformations
    A.GaussianBlur(blur_limit=(3, 7), p=0.7), 
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.6), 
    A.ISONoise(p=0.5), 

], bbox_params={'format': 'coco', 'label_fields': ['category_id']}, p=1)
cr_transforms = [combined_augmentations]

if len(cr_transforms) > 0:
    print(f"🚀  Using {len(cr_transforms)} augmentations.")

# Intro

print(f"🚀  Training {astrosam_model.model.__class__.__name__} with {len(training_image_paths)} training images and {len(val_image_paths)} validation images.")
print(f"🚀  Training for {num_epochs} epochs with effective batch size {batch_size * (len(cr_transforms) + 1)} and learning rate {lr}.")
print(f"🚀  Initial learning rate: {lr}. Weight decay: {wd}.")
print(f"🚀  Using learning rate warmup scheduler: {use_lr_warmup}. ")
print(f"🚀  Early stopping after {n_epochs_stop} epochs without improvement.")
print(f"🚀  Training started.\n")

for epoch in range(num_epochs):

    # Train
    astrosam_model.model.train()
    # astrosam_model.residualAttentionBlock.train()
	
    epoch_loss = astrosam_model.train_validate_step(
        train_dataloader, 
        train_dir, 
        train_gt_masks, 
        train_bboxes, 
        optimizer, 
        mode='train',
        cr_transforms=cr_transforms,
        scheduler=scheduler)
    
    train_losses.append(epoch_loss)
    
    # Validate
    astrosam_model.model.eval()
    # astrosam_model.residualAttentionBlock.eval()
	
    with torch.no_grad():
        epoch_val_loss =  astrosam_model.train_validate_step(
            val_dataloader, 
            valid_dir, 
            val_gt_masks, 
            val_bboxes, 
            optimizer, 
            mode='validate',
            cr_transforms = None,
            scheduler=None)
            
    valid_losses.append(epoch_val_loss)

    # Logging
    if wandb_track:
        wandb.log({'epoch training loss': epoch_loss, 'epoch validation loss': epoch_val_loss})

    print(f'EPOCH: {epoch}. Training loss: {epoch_loss}')
    print(f'EPOCH: {epoch}. Validation loss: {epoch_val_loss}.')

    if epoch_val_loss < best_valid_loss:
        best_valid_loss = epoch_val_loss
        best_epoch = epoch
        best_model = astrosam_model.model
        # best_residual_attn = residualAttentionBlock
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print("Early stopping initiated.")
            early_stop = True
            break
    
    print(f"Best epoch: {best_epoch}. Best validation loss: {best_valid_loss}.\n")
    torch.save(best_model.state_dict(), f'{work_dir}/sam_{kfold_iter}_best.pth')
                
torch.save(best_model.state_dict(), f'{work_dir}/sam_{kfold_iter}_{datetime.now()}_best.pth')
torch.save(astrosam_model.model.state_dict(), f'{work_dir}/sam_{kfold_iter}_{datetime.now()}_last.pth')
# torch.save(astrosam_model.residualAttentionBlock.state_dict(), f'{work_dir}/residual_attn_blk_{kfold_iter}_{datetime.now()}_best.pth')

if wandb_track:
    wandb.run.summary["batch_size"] = batch_size * (len(cr_transforms) + 1)
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["best_valid_loss"] = best_valid_loss
    wandb.run.summary["num_epochs"] = num_epochs
    wandb.run.summary["learning rate"] = lr
    wandb.run.summary["weight_decay"] = wd
    wandb.run.summary["# train images"] = len(train_dataloader)
    wandb.run.summary["# validation images"] = len(val_dataloader)
    wandb.run.summary["checkpoint"] = mobile_sam_checkpoint
    run.finish()