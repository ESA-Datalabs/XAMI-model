#!/usr/bin/env python
# coding: utf-8

# ### 00 Install segment anything

# source: https://maxjoas.medium.com/finetune-segment-anything-sam-for-images-with-multiple-masks-34514ee811bb

# In[1]:


# !pip install segment_anything


# ### Import packages

# In[2]:


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import glob
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

import json
import toml
import os
import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[19]:


from torch.types import Device
#data
global train
global test
global annot

train_path = "./roboflow_datasets/xmm_om_artefacts_512-9-COCO/train"
test = "./roboflow_datasets/xmm_om_artefacts_512-9-COCO/valid"

annot_train = "_annotations.coco.json"
annot_test = "_annotations.coco.json"

#model
global batch_size
global epochs
global lr
global weight_decay
global DEVICE

# TODOD adjust, if needed!!!!
batch_size = 1
epochs = 50
lr = 6e-4
weight_decay = 0.00005
loss_scaling_factor = 1e4  # to save some very small gradients (on some attention params) from extinction 

device_id = 3
torch.cuda.set_device(device_id)

DEVICE = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# loss
global alpha
global gamma
alpha = 0.8
gamma = 2


# In[20]:


use_wandb = True
if use_wandb:
        from datetime import datetime
        import wandb
        wandb.login()
        run = wandb.init(project="OM_AI_all_masks", name=f"ft_MobileSAM {datetime.now()}")


# In[21]:


# ## 01 Segment image with SAM

# In[6]:


from typing import List, Dict, Any
def build_totalmask(pred: List[Dict[str, Any]]) -> np.ndarray:
    """Builds a total mask from a list of segmentations
    ARGS:
        pred (list): list of dicts with keys 'segmentation' and others
    RETURNS:
        total_mask (np.ndarray): total mask

    """

    total_mask = np.zeros(pred[0]['segmentation'].shape, dtype=np.uint8)
    for seg in pred:
        total_mask += seg['segmentation']
    # use cv2 to make image black and white
    _, total_mask = cv2.threshold(total_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    return total_mask


# In[7]:


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="../MobileSAM-fine-tuning/weights/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
sam.to(DEVICE)


# ### Summary Segment Anything format:
# 
# `SamAutomaticMaskGenerator` returns a `list` of masks, where each mask is a `dict` containing various information about the mask:
# 
# * `segmentation` - `[np.ndarray]` - the mask with `(W, H)` shape, and `bool` type
# * `area` - `[int]` - the area of the mask in pixels
# * `bbox` - `[List[int]]` - the boundary box of the mask in `xywh` format
# * `predicted_iou` - `[float]` - the model's own prediction for the quality of the mask
# * `point_coords` - `[List[List[float]]]` - the sampled input point that generated this mask
# * `stability_score` - `[float]` - an additional measure of mask quality
# * `crop_box` - `List[int]` - the crop of the image used to generate this mask in `xywh` format
# 
# - The mask generator uses a grid of points as prompts and generates masks for each point.[see here](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py)
# 
# - The coco annotations categories is a list of dictionaries with keys:
#   * `id` - the id of the class
#   * `name` - the name corresponding to the id
#   * `supercategory` 

# In[22]:


import numpy as np
from torchvision.transforms.functional import resize

class COCODataset(Dataset):
    """
    A PyTorch Dataset to load data from a json file in COCO format.

    ...
    Attributes
    ----------
    root_dir : str
        the root directory containing the images and annotations
    annotation_file : str
        name of the json file containing the annotations (in root_dir)
    transform : callable
        a function/transform to apply to each image

    Methods
    -------
    __getitem__(idx)
        returns the image, image path, and masks for the given index
    buid_total_mask(masks)
        combines the masks into a single mask
    """


    def __init__(self, root_dir, annotation_file, transform=None, split=None, percentage=None):

        assert (split is None and percentage is None) or (split is not None and percentage is not None), "Either both split and percentage should be None or neither."
        assert (split=='train') or (split=='val'), "Specify the split by using 'train' or 'val' keywords."
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids_1 = list(self.coco.imgs.keys())
        self.categories = self.coco.loadCats(self.coco.getCatIds()) 
        # Filter out image_ids without any annotations
        self.image_ids_1 = [image_id for image_id in self.image_ids_1 if len(self.coco.getAnnIds(imgIds=image_id)) > 0]
        self.categories = {self.categories[i]['id']: self.categories[i]['name'] for i in range(len(self.categories))} # id:name dict
        self.image_ids = []

        for idx in range(len(self.image_ids_1)):
            image_id = self.image_ids_1[idx]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            good_image = False
            self.image_ids.append(image_id)

        if split == 'train':
            percentage = int(len(self.image_ids) * percentage)
            self.image_ids = self.image_ids[:percentage]
        else:
            percentage = int(len(self.image_ids) * (1-percentage))
            self.image_ids = self.image_ids[percentage:]
                

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            # if not self.categories[ann['category_id']].endswith('star') and (not self.categories[ann['category_id']].endswith('streak')): # this is safe because images with only stars are already filtered out in __init__
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
      
        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, image_path, torch.tensor(masks).float()

    def get_totalmask(self, masks):
        """get all masks in to one image
        ARGS:
            masks (List[Tensor]): list of masks
        RETURNS:
            total_gt (Tensor): all masks in one image

        """
        total_gt = torch.zeros_like(masks[0][0,:,:])
        for k in range(len(masks[0])):
            total_gt += masks[0][k,:,:]
        return total_gt



class ResizeAndPad:
    """
    Resize and pad images and masks to a target size.

    ...
    Attributes
    ----------
    target_size : int
        the target size of the image
    transform : ResizeLongestSide
        a transform to resize the image and masks
    """

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape

        negative_map = (np.where(image > 0, 1, 0)).astype(np.uint8)
        negative_map = torch.from_numpy(negative_map)  
        negative_map = negative_map.permute(2, 0, 1)
        negative_map = resize(negative_map, [1024, 1024], antialias=True) 
        negative_map = negative_map.to(torch.uint8)
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # image = image * negative_map # mask -ve pixels
        # print(negative_map.shape, image.shape, np.unique(negative_map.detach().cpu().numpy()))
        # plt.imshow(image[0].detach().cpu().numpy())
        # plt.show()
        # plt.close()

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes


def load_datasets(img_size):
    """ load the training and validation datasets in PyTorch DataLoader objects
    ARGS:
        img_size (Tuple(int, int)): image size
    RETURNS:
        train_dataloader (DataLoader): training dataset
        val_dataloader (DataLoader): validation dataset

    """
    transform = ResizeAndPad(1024)
    traindata = COCODataset(root_dir=train_path,
                        annotation_file=os.path.join(train_path, annot_train),
                        transform=transform,
                        split='train',
                        percentage=1.0)
    valdata = COCODataset(root_dir=test,
                      annotation_file=os.path.join(test, annot_test),
                      transform=transform,
                      split='val', 
                      percentage=1.0)
    train_dataloader = DataLoader(traindata,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1)
    val_dataloader = DataLoader(valdata,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1)
    return train_dataloader, val_dataloader


# In[24]:


trainloader, validloader = load_datasets(1024)
sample_img = trainloader.dataset[0][0]
sample_mask = trainloader.dataset[0][2]

print('#train images:', len(trainloader.dataset), '#validation images:', len(validloader.dataset))


# In[25]:


print(f'shape of sample_img: {sample_img.shape}')
print(f'shape fo sample_mask: {sample_mask.shape}')


# ### Summary functions and classes:
# Now we have transformed our images and coco annotations to torch tensors, that we can use for training. For training (fine-tuning SAM) we need to define a Neural net with PyTorch first, we do this in the next class. It's pretty well documented, so I'll leave you with the code:

# In[26]:


import torch.nn as nn
class ModelSimple(nn.Module):
    """
    Wrapper for the sam model to to fine-tune the model on a new dataset

    ...
    Attributes:
    -----------
    freeze_encoder (bool): freeze the encoder weights
    freeze_decoder (bool): freeze the decoder weights
    freeze_prompt_encoder (bool): freeze the prompt encoder weights
    transform (ResizeLongestSide): resize the images to the model input size

    Methods:
    --------
    setup(): load the model and freeze the weights
    forward(images, points): forward pass of the model, returns the masks and iou_predictions
    """

    def __init__(self, freeze_encoder=True, freeze_decoder=False, freeze_prompt_encoder=True):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        # we need this to make the input image size compatible with the model
        self.transform = ResizeLongestSide(1024) #This is 1024, because sam was trained on 1024x1024 images

    def setup(self, MODEL_TYPE, chekpoint):
        self.model = sam_model_registry[MODEL_TYPE](chekpoint)
        # to speed up training time, we normally freeze the encoder and decoder
        if self.freeze_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        self.transfrom = ResizeLongestSide(self.model.image_encoder.img_size)
        
    def forward(self, images, negative_maps):

        # print('images shape:', images.shape)
        _, _, H, W = images.shape # batch, channel, height, width - well, batch_size cannot be greater than 1 apparently
        
        image_embeddings = self.model.image_encoder(images) # shape: (1, 256, 64, 64)
        # get prompt embeddings without acutally any prompts (uninformative)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # get low resolution masks and iou predictions
        # mulitmask_output=False means that we only get one mask per image,
        # otherwise we would get three masks per image
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings, # sparse_embeddings shape: (1, 0, 256)
            dense_prompt_embeddings=dense_embeddings, # dense_embeddings shape: (1, 256, 256)
            multimask_output=False,
        )

        # print('low_res_masks', low_res_masks.shape)
        # postprocess the masks to get the final masks and resize them to the original image size
        masks = F.interpolate(
            low_res_masks, # shape: (1, 1, 256, 256)
            (H, W),
            mode="bilinear",
            align_corners=False,
        )
        
        masks = masks * negative_maps

        # print(masks.shape)
        # shape masks after interpolate: torch.Size([1, 1, 1024, 1024])
        return masks, iou_predictions


# ## Models, classes functions for fine-tuning

# In[27]:


import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm
from tqdm import tqdm
import time

def get_totalmask(masks):
    """get all masks in to one image
    ARGS:
        masks (torch.Tensor): shape: (N, H, W) where N is the number of masks
                              masks H,W is usually 1024,1024
    RETURNS:
        total_gt (torch.Tensor): all masks in one image

    """
    total_gt = torch.zeros_like(masks[0,:,:])
    for k in range(len(masks)):
        total_gt += masks[k,:,:]
    return total_gt


class FocalLoss(nn.Module):
    """ Computes the Focal loss. """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):

        inputs = inputs.flatten(0,2)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    """ Computes the Dice loss. """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(0,2)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        return 1 - dice



def criterion(x, y):
    """ Combined dice and focal loss.
    ARGS:
        x: (torch.Tensor) the model output
        y: (torch.Tensor) the target
    RETURNS:
        (torch.Tensor) the combined loss

    """
    focal, dice = FocalLoss(), DiceLoss()
    y = y.to(DEVICE)
    x = x.to(DEVICE)
    return loss_scaling_factor * (20 * focal(x, y) + dice(x, y)) # 📍📍📍

def validate_step(model, validloader):
    # trainloader, validloader = load_datasets(1024)

    with torch.no_grad():
        
        model.eval()
        running_vloss = 0.
        with torch.no_grad():
            for images, path, masks in validloader:
                orig_image = cv2.imread(path[0])
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                
                negative_map = (np.where(orig_image > 0, 1, 0)).astype(np.uint8)
                negative_map = torch.from_numpy(negative_map)  
                negative_map = negative_map.permute(2, 0, 1)
                negative_map = resize(negative_map, [1024, 1024], antialias=True) 
                negative_map = negative_map.to(torch.uint8).squeeze(0).to(DEVICE)
                
                # model.to(DEVICE)
                images = images.to(DEVICE)
                masks = masks[0].to(DEVICE)
                total_mask = get_totalmask(masks)
                total_mask = total_mask.to(DEVICE)
                preds, iou = model(images, negative_map)
                preds = preds.to(DEVICE)
                
                # was criterion(preds, total_mask) before. i changed because the multimask_output was set on True. anyways, it treats both cases now
                vloss = criterion(preds[0][0].unsqueeze(0).unsqueeze(0), total_mask) 
                running_vloss += vloss.item()
                np_image = images[0].permute(1,2,0).detach().cpu().numpy()
                np_mask = masks.detach().cpu().numpy()
                np_pred = preds[0].detach().cpu().numpy()
        
                binary_pred_mask = ((F.sigmoid(preds[0]).cpu().numpy()[0]) > 0.5).astype(int) * 255
                binary_pred_mask = ((preds[0]).cpu().numpy()[0] > 0).astype(int) * 255
                
                print(np.mean((F.sigmoid(preds[0]).cpu().numpy()[0])), np.min((F.sigmoid(preds[0]).cpu().numpy()[0])), np.max((F.sigmoid(preds[0]).cpu().numpy()[0])))
                print(np.unique((F.sigmoid(preds[0]).cpu().numpy()[0]) > 0.5).astype(int))
                print("validation loss: ", running_vloss)
                
                fig, axs = plt.subplots(1, 3)
                
                axs[0].imshow(np_image)
                axs[0].axis('off')
            
                axs[1].imshow(get_totalmask(masks).detach().cpu().numpy())
                axs[1].set_title('Ground truth masks', fontsize=10)
                axs[1].axis('off')
                
                axs[2].imshow(binary_pred_mask)
                axs[2].set_title('Predicted masks', fontsize=10)
                axs[2].axis('off')
                # plt.savefig('on_img_train.png', dpi=300)
            
                plt.show()
                plt.close()
    return running_vloss/len(validloader)
    
def train_one_epoch(model, trainloader, optimizer, epoch_idx):
    """ Runs forward and backward pass for one epoch and returns the average
    batch loss for the epoch.
    ARGS:
        model: (nn.Module) the model to train
        trainloader: (torch.utils.data.DataLoader) the dataloader for training
        optimizer: (torch.optim.Optimizer) the optimizer to use for training
        epoch_idx: (int) the index of the current epoch
        tb_writer: (torch.utils.tensorboard.writer.SummaryWriter) the tensorboard writer
    RETURNS:
        last_loss: (float) the average batch loss for the epoch

    """
    running_loss = 0.
    for i, (image, path, masks) in tqdm(enumerate(trainloader)):
        orig_image = cv2.imread(path[0])
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        negative_map = (np.where(orig_image > 0, 1, 0)).astype(np.uint8)
        negative_map = torch.from_numpy(negative_map)  
        negative_map = negative_map.permute(2, 0, 1)
        negative_map = resize(negative_map, [1024, 1024], antialias=True) 
        negative_map = negative_map.to(torch.uint8).squeeze(0).to(DEVICE)
        del orig_image
        image = image.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(image, negative_map) 
        # print('pred', pred.shape)
        # pred, _ = model(255-image, negative_map) # 📌📌📌
        masks = masks[0].to(DEVICE) # (number of masks, 1024, 1024)

        # # match predicted with ground truth masks (using the Hungarian method)
        # matched_indices, iou_scores_tensor = associate_masks_tensor(pred, masks)
        # print(matched_indices, iou_scores_tensor)
        total_mask = get_totalmask(masks)
        a_mask_on_prediction = total_mask > 0.0 # the predicted total mask will ignore stars and other non-annotated objects from prediction
        # pred[0][0] = pred[0][0] * a_mask_on_prediction
        pred = pred.to(DEVICE)
        loss = criterion(pred[0][0].unsqueeze(0).unsqueeze(0), total_mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        binary_pred_mask = ((pred.detach().cpu().numpy()[0]) > 0.0).astype(int)

        # for name, parameter in model.named_parameters():
        #         if parameter.grad is not None: 
        #             grad_norm = parameter.grad.norm()
        #             if grad_norm < 1e-8: 
        #                 print(f'❗️Layer {name} has vanishing gradients: {grad_norm}')
                        
        # fig, axs = plt.subplots(1, 3)
        
        # axs[0].imshow(image[0].permute(1,2,0).detach().cpu().numpy())
        # axs[0].axis('off')
    
        # axs[1].imshow(total_mask.detach().cpu().numpy())
        # axs[1].set_title('Ground truth masks', fontsize=10)
        # axs[1].axis('off')
        
        # # axs[2].imshow(pred.detach().cpu().numpy()[0][0])
        # axs[2].imshow(binary_pred_mask[0])
        # axs[2].set_title('Predicted masks', fontsize=10)
        # axs[2].axis('off')
        # # plt.savefig('on_img_train.png', dpi=300)
    
        # plt.show()
        # plt.close()


    i = len(trainloader)
    last_loss = running_loss / i
    # print(f'batch_loss for batch {i}: {last_loss}')
    tb_x = epoch_idx * len(trainloader) + i + 1
    # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    running_loss = 0.
    return last_loss


def train(n_epochs_stop):
    """ Trains the model for the given number of epochs."""
    model = ModelSimple()
    model.setup('vit_h', "../MobileSAM-fine-tuning/weights/sam_vit_h_4b8939.pth")

    if use_wandb:
        wandb.watch(model, log='all', log_graph=True)

    # model.load_state_dict(torch.load('model_final.pth'))
    model.to(DEVICE)
    img_size = model.model.image_encoder.img_size
    trainloader, validloader = load_datasets(img_size=img_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_loss = float('inf')

    train_losses, valid_losses = [], []
    for epch in range(epochs):
        #train
        running_vloss = 0.
        model.train()
        avg_batchloss = train_one_epoch(model, trainloader, optimizer, epch)/loss_scaling_factor
        train_losses.append(avg_batchloss)
        
        # validate
        eval = True
        if not eval: 
            continue
        running_vloss = validate_step(model, validloader)/loss_scaling_factor
        valid_losses.append(running_vloss)
        
        # save model
        print(f'epoch: {epch}, train loss: {avg_batchloss}.')
        print(f'epoch: {epch}, validloss: {running_vloss}')
        if running_vloss < best_valid_loss:
            best_valid_loss = running_vloss
            best_model = model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping initiated.")
                break

        print(f'best valid loss: {best_valid_loss}')

        # Logging
        if use_wandb:
            wandb.log({'epoch_train_loss': avg_batchloss, 'epoch_val_loss': running_vloss})
            
    return best_model, train_losses, valid_losses


# ### Train

# In[ ]:


use_wandb = True
model, train_losses, val_losses = train(n_epochs_stop=10)


# In[ ]:


torch.save(model.state_dict(), f'model_final_all_masks.pth')

if use_wandb:
    wandb.run.summary["batch_size"] = batch_size
    wandb.run.summary["num_epochs"] = epochs
    wandb.run.summary["learning rate"] = lr
    wandb.run.summary["weight_decay"] = weight_decay
    wandb.run.summary["loss_scaling_factor"] = loss_scaling_factor
    wandb.run.summary["alpha"] = alpha
    wandb.run.summary["gamma"] = gamma
    wandb.run.summary["#train images"] = len(trainloader.dataset)
    wandb.run.summary["#validation images"] = len(validloader.dataset)

    run.finish()


# In[ ]:


torch.save(model.state_dict(), f'model_final_all_masks.pth')


# In[ ]:


print('train losses:', train_losses)
print('val losses:', val_losses)


# In[ ]:


plt.plot(list(range(len(train_losses))), train_losses, label='Training Loss')
plt.plot(list(range(len(val_losses))), val_losses, label='Validation Loss')
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./plots/amg_losses.png')
plt.show()


# In[ ]:


# # sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
# mask_generator = SamAutomaticMaskGenerator(model.model)
# # sam.to(DEVICE)
# predicted_masks = mask_generator.generate(sample_img) # this doesn't really work this way, but rather in the way the model was fine-tuned
# print(len(predicted_masks))

# total_mask = build_totalmask(predicted_masks)
# plt.imshow(total_mask, cmap='gray')


# ## Predict trained model

# In[ ]:


# predict trainied model

def evaluate_model(fine_tuned = True):
    trainloader, validloader = load_datasets(1024)

    with torch.no_grad():
        model = ModelSimple()
        model.setup('vit_h', "../MobileSAM-fine-tuning/weights/sam_vit_h_4b8939.pth")

        if fine_tuned:
            model.load_state_dict(torch.load('model_final_all_masks.pth', map_location=torch.device(device_id)))
            
        model.eval()
        img_size = model.model.image_encoder.img_size
        running_vloss = 0.
        for images, path, masks in validloader:
            orig_image = cv2.imread(path[0])
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            
            negative_map = (np.where(orig_image > 0, 1, 0)).astype(np.uint8)
            negative_map = torch.from_numpy(negative_map)  
            negative_map = negative_map.permute(2, 0, 1)
            negative_map = resize(negative_map, [1024, 1024], antialias=True) 
            negative_map = negative_map.to(torch.uint8).squeeze(0).to(DEVICE)
            
            # print(images)
            model.to(DEVICE)
            images = images.to(DEVICE)
            masks = masks[0].to(DEVICE)
            total_mask = get_totalmask(masks)
            total_mask = total_mask.to(DEVICE)
            model.eval()
            preds, iou = model(images, negative_map)
            preds = preds.to(DEVICE)
            vloss = criterion(preds[0][0].unsqueeze(0).unsqueeze(0), total_mask) # was criterion(preds, total_mask) before. i changed because the multimask_output is set on True
            running_vloss += vloss.item()
            np_image = images[0].permute(1,2,0).detach().cpu().numpy()
            np_mask = masks.detach().cpu().numpy()
            np_pred = preds[0].detach().cpu().numpy()
    
            binary_pred_mask = ((F.sigmoid(preds[0]).cpu().numpy()[0]) > 0.5).astype(int) * 255
            # binary_pred_mask = ((preds[0]).cpu().numpy()[0] > 0).astype(int) * 255
            
            print(np.mean((F.sigmoid(preds[0]).cpu().numpy()[0])), np.min((F.sigmoid(preds[0]).cpu().numpy()[0])), np.max((F.sigmoid(preds[0]).cpu().numpy()[0])))
            print(np.unique((F.sigmoid(preds[0]).cpu().numpy()[0]) > 0.5).astype(int))
            print("validation loss: ", running_vloss)
            
            fig, axs = plt.subplots(1, 3)
            
            axs[0].imshow(np_image)
            axs[0].axis('off')
        
            axs[1].imshow(get_totalmask(masks).detach().cpu().numpy())
            axs[1].set_title('Ground truth masks', fontsize=10)
            axs[1].axis('off')
            
            axs[2].imshow(binary_pred_mask)
            axs[2].set_title('Predicted masks', fontsize=10)
            axs[2].axis('off')
            # plt.savefig('on_img_train.png', dpi=300)
        
            plt.show()
            plt.close()
            
    return running_vloss/len(validloader)


# In[ ]:


eval = True

if eval:
    fine_tuned_loss = []
    fine_tuned_loss.append(evaluate_model(fine_tuned = True))


# In[ ]:


if eval:
    orig_loss = []
    orig_loss.append(evaluate_model(fine_tuned = False))
    print('Fine-tuned loss: ', np.mean(fine_tuned_loss), 'Original model loss:', np.mean(orig_loss))

