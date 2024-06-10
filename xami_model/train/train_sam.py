# In[1]:
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3" # replace with the GPU IDs that are available
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

from xami_model.dataset import dataset_utils, load_dataset
from xami_model.model_predictor import xami, predictor_utils

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
use_lr_initial_decay=True
n_epochs_stop = 100 #15 # Early stopping
use_CR = True # Use Consistency Regularization
work_dir = './output_sam'
input_dir = f'../data/xami_dataset/' # path to the dataset

work_dir = predictor_utils.get_next_directory_name(work_dir)
os.makedirs(work_dir)
print(f"Working directory: {work_dir}")

# torch.cuda.set_device(device_id)
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# this variable is the original batch size
# the effective batch size will be: original_batch_size * (number of augmentations + 1) 
# if applying augmentations
# high values (e.g., 8, 16) may lead to OOM errors
batch_size=4

# Load checkpoints
mobile_sam_dir = os.path.join(os.getcwd(), '..', 'mobile_sam')
mobile_sam_checkpoint = os.path.join(mobile_sam_dir,"weights/mobile_sam.pt")

# Dataset split
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
from xami_model.mobile_sam.mobile_sam import sam_model_registry, SamPredictor#, build_efficientvit_l2_encoder

# Segment Anything Model
model = sam_model_registry["vit_t"](checkpoint=mobile_sam_checkpoint)
model.to(device);
predictor = SamPredictor(model)
xami_model = xami.XAMI(model, device, predictor, apply_segm_CR=use_CR) #, residualAttentionBlock)

# # # Residual Attention Block
# residual_block = residualAttentionBlock.ResidualAttentionBlock(d_model=768, n_head=8, mlp_ratio=4.0).to(device)
# checkpoint = torch.load('./weights/ViT-B-32__openai.pth')
# param_mapping = {
#     'visual.transformer.resblocks.0.ln_1.weight': 'ln_1.weight',
#     'visual.transformer.resblocks.0.ln_1.bias': 'ln_1.bias',
#     'visual.transformer.resblocks.0.attn.in_proj_weight': 'attn.in_proj_weight',
#     'visual.transformer.resblocks.0.attn.in_proj_bias': 'attn.in_proj_bias',
#     'visual.transformer.resblocks.0.attn.out_proj.weight': 'attn.out_proj.weight',
#     'visual.transformer.resblocks.0.attn.out_proj.bias': 'attn.out_proj.bias',
#     'visual.transformer.resblocks.0.ln_2.weight': 'ln_2.weight',
#     'visual.transformer.resblocks.0.ln_2.bias': 'ln_2.bias',
#     'visual.transformer.resblocks.0.mlp.c_fc.weight': 'mlp.c_fc.weight',
#     'visual.transformer.resblocks.0.mlp.c_fc.bias': 'mlp.c_fc.bias',
#     'visual.transformer.resblocks.0.mlp.c_proj.weight': 'mlp.c_proj.weight',
#     'visual.transformer.resblocks.0.mlp.c_proj.bias': 'mlp.c_proj.bias'
# }

# model_state_dict = residual_block.state_dict()

# for checkpoint_param_name, model_param_name in param_mapping.items():
#     if checkpoint_param_name in checkpoint and model_param_name in model_state_dict:
#         model_state_dict[model_param_name] = checkpoint[checkpoint_param_name]

# residual_block.load_state_dict(model_state_dict)

# for name, param in residual_block.named_parameters():
#     param.requires_grad = True 

# xami_model = xami.XAMI(
#     model, 
#     device, 
#     predictor, 
#     use_yolo_masks=False, 
#     # wt_threshold=0.6, 
#     # wt_classes_ids = [1.0, 4.0],
#     apply_segm_CR=True,
#     residualAttentionBlock=residual_block)

if wandb_track:
    # !pip install wandb
    # !wandb login --relogin
    import wandb
    wandb.login()
    run = wandb.init(project="sam", name=f"sam_{kfold_iter}_{datetime.now()}")
    wandb.watch(xami_model.model, log='all', log_graph=True)

## Convert the input images into a format SAM's internal functions expect.
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader

# Dataset Loaders and Pre-processing
transform = ResizeLongestSide(xami_model.model.image_encoder.img_size)
train_set = load_dataset.ImageDataset(training_image_paths, xami_model.model, transform, device) 
val_set =  load_dataset.ImageDataset(val_image_paths, xami_model.model, transform, device) 
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Optimizer
for name, param in xami_model.model.named_parameters():
    if 'mask_decoder' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

parameters_to_optimize = [param for param in xami_model.model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(parameters_to_optimize, lr=lr, weight_decay=wd) 

# Scheduler
scheduler=None

if use_lr_initial_decay:
    initial_lr = lr
    final_lr = 6e-5
    total_steps = 100 #16  # total steps over which the learning rate should decrease
    lr_decrement = (initial_lr - final_lr) / total_steps

    def lr_lambda(current_step):
        if current_step < total_steps:
            return 1 - current_step * lr_decrement / initial_lr
        return final_lr / initial_lr
		
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Model weights
print(f"🚀  The SAM model has {sum(p.numel() for p in xami_model.model.parameters() if p.requires_grad)} trainable parameters.\n")

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

print(f"🚀  Training {xami_model.model.__class__.__name__} with {len(training_image_paths)} training images and {len(val_image_paths)} validation images.")
print(f"🚀  Training for {num_epochs} epochs with effective batch size {batch_size * (len(cr_transforms) + 1)} and learning rate {lr}.")
print(f"🚀  Initial learning rate: {lr}. Weight decay: {wd}.")
print(f"🚀  Using learning rate initial decay scheduler: {use_lr_initial_decay}. ")
print(f"🚀  Early stopping after {n_epochs_stop} epochs without improvement.")
print(f"🚀  Training started.\n")

iou_eval_thresholds = [0.5, 0.75, 0.9]
    
for epoch in range(num_epochs):

    # Train
    xami_model.model.train()
    if xami_model.residualAttentionBlock is not None:
        xami_model.residualAttentionBlock.train()
	
    epoch_loss, _, _, _ = xami_model.train_validate_step(
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
    xami_model.model.eval()
    if xami_model.residualAttentionBlock is not None:
        xami_model.residualAttentionBlock.eval()
	
    with torch.no_grad():
        epoch_val_loss, all_image_ids, all_gt_masks, all_pred_masks =  xami_model.train_validate_step(
            val_dataloader, 
            valid_dir, 
            val_gt_masks, 
            val_bboxes, 
            optimizer, 
            mode='validate',
            cr_transforms = None,
            scheduler=None)
            
        valid_losses.append(epoch_val_loss)
        p_metric_name, p_means, p_stds, p_thresholds = predictor_utils.compute_scores('precision', all_pred_masks, all_gt_masks, iou_eval_thresholds)
        r_metric_name, r_means, r_stds, r_thresholds = predictor_utils.compute_scores('recall', all_pred_masks, all_gt_masks, iou_eval_thresholds)
        f_metric_name, f_means, f_stds, f_thresholds = predictor_utils.compute_scores('f1_score', all_pred_masks, all_gt_masks, iou_eval_thresholds)
        a_metric_name, a_means, a_stds, a_thresholds = predictor_utils.compute_scores('accuracy', all_pred_masks, all_gt_masks, iou_eval_thresholds)
        print('Precision', p_means, 'Recall', r_means, 'F1-score', f_means, 'Accuracy', a_means)
        del all_image_ids, all_gt_masks, all_pred_masks, p_metric_name, r_metric_name, f_metric_name, a_metric_name
        del p_stds, p_thresholds, r_stds, r_thresholds, f_stds, f_thresholds, a_stds, a_thresholds
        
    # Logging
    if wandb_track:
        wandb.log({'epoch training loss': epoch_loss, 'epoch validation loss': epoch_val_loss})
        wandb.log({'Precision': p_means, 'Recall': r_means, 'F1-score': f_means, 'Accuracy': a_means})

    print(f'EPOCH: {epoch}. Training loss: {epoch_loss}')
    print(f'EPOCH: {epoch}. Validation loss: {epoch_val_loss}.')

    if epoch_val_loss < best_valid_loss:
        best_valid_loss = epoch_val_loss
        best_epoch = epoch
        best_model = xami_model.model
        best_residual_attn = xami_model.residualAttentionBlock
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
torch.save(xami_model.model.state_dict(), f'{work_dir}/sam_{kfold_iter}_{datetime.now()}_last.pth')
if best_residual_attn is not None:
    torch.save(best_residual_attn.state_dict(), f'{work_dir}/residual_attn_blk_{kfold_iter}_{datetime.now()}_best.pth')

if wandb_track:
    wandb.run.summary["batch_size"] = batch_size * (len(cr_transforms) + 1)
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["best_valid_loss"] = best_valid_loss
    wandb.run.summary["num_epochs"] = num_epochs
    wandb.run.summary["learning rate"] = lr
    wandb.run.summary["weight_decay"] = wd
    wandb.run.summary["# train_dataloader"] = len(train_dataloader)
    wandb.run.summary["# val_dataloader"] = len(val_dataloader)
    wandb.run.summary["checkpoint"] = mobile_sam_checkpoint
    run.finish()