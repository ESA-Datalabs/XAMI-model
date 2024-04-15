# # Set up

# In[1]:
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"

from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import json
import sys
from pycocotools import mask as maskUtils
np.set_printoptions(precision=15)
import albumentations as A
from dataset import dataset_utils
from losses import loss_utils
from sam_predictor import load_dataset, astro_sam, predictor_utils

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR

# In[2]:


# !nvidia-smi


# **train config**

# In[3]:

kfold_iter = sys.argv[1]
lr=3e-4
wd=0.0005
batch_size = 10
num_epochs = 100
work_dir = './output_sam'
device_id = int(sys.argv[2])
torch.cuda.set_device(device_id)
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
print("DEVICE", device)
wandb_track = True

if not os.path.exists(work_dir):
	os.makedirs(work_dir)
else:
    print(f"Output directory {work_dir} already exists.")
    
# # Dataset split

# In[4]:

split_dataset_here = False


# In[5]:

input_dir = './roboflow_datasets/xmm_om_artefacts_512-20-COCO-splits/'

train_dir = input_dir+f'train_{kfold_iter}/'
valid_dir = input_dir+f'valid_{kfold_iter}/'
json_train_path, json_valid_path = train_dir+'skf_train_annotations.coco.json', valid_dir+'skf_valid_annotations.coco.json'

with open(json_train_path) as f:
    train_data_in = json.load(f)
with open(json_valid_path) as f:
    valid_data_in = json.load(f)

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


# In[8]:


print('# dataset images: \ntrain', len(training_image_paths), '\nvalid', len(val_image_paths))


# In[9]:


train_data_in['categories']


# **Visualize some annotations**

# In[10]:


for one_path in training_image_paths[:4]:
    image_id2 = one_path.split('/')[-1]
    print(image_id2)
    image_masks_ids = [key for key in train_gt_masks.keys() if key.startswith(image_id2)]
    image_ = cv2.imread(one_path)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(image_)
    for name in image_masks_ids:
        dataset_utils.show_box(train_bboxes[name], plt.gca())
        dataset_utils.show_mask(maskUtils.decode(train_gt_masks[name]), plt.gca())
    plt.axis('off')
    plt.show()
    plt.close()


# ## 🚀 Prepare Mobile SAM Fine Tuning

# In[12]:

import sys
sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/')
from ft_mobile_sam import sam_model_registry, SamPredictor, build_efficientvit_l2_encoder

mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/weights/mobile_sam.pt"
model = sam_model_registry["vit_t"](checkpoint=mobile_sam_checkpoint)

efficient_vit_enc = '../MobileSAM-fine-tuning/MobileSAMv2/weight/l2.pt'
model.image_encoder = build_efficientvit_l2_encoder(efficient_vit_enc)

model.to(device);
predictor = SamPredictor(model)
# ## Import the model for training

# In[13]:

astrosam_model = astro_sam.AstroSAM(model, device, predictor)


# In[14]:


if wandb_track:
    # !pip install wandb
    # !wandb login --relogin
    import wandb
    wandb.login()
    run = wandb.init(project="OM_AI_v1", name=f"ft_MobileSAM {datetime.now()}")
    wandb.watch(astrosam_model.model, log='all', log_graph=True)


# ## Convert the input images into a format SAM's internal functions expect.

# In[15]:


import torch
from segment_anything.utils.transforms import ResizeLongestSide


# In[16]:


from torch.utils.data import DataLoader

transform = ResizeLongestSide(astrosam_model.model.image_encoder.img_size)
train_set = load_dataset.ImageDataset(training_image_paths, astrosam_model.model, transform, device) 
val_set =  load_dataset.ImageDataset(val_image_paths, astrosam_model.model, transform, device) 
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# In[17]:

import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import torch.nn as nn

parameters_to_optimize = [param for param in astrosam_model.model.mask_decoder.parameters() 
                          if param.requires_grad]
optimizer = torch.optim.AdamW(parameters_to_optimize, lr=lr, weight_decay=wd) 

# ## Model weights

# In[18]:

for name, param in astrosam_model.model.named_parameters():
    if 'mask_decoder' in name:
    # if True:
        param.requires_grad = True
    else:
        param.requires_grad = False

# In[20]:


print(f"🚀 The model has {sum(p.numel() for p in astrosam_model.model.parameters() if p.requires_grad)} trainable parameters.\n")
predictor_utils.check_requires_grad(astrosam_model.model)


# ## Run fine tuning

# In[21]:

train_losses = []
valid_losses = []
best_valid_loss = float('inf')
n_epochs_stop = num_epochs/5

geometrical_augmentations = A.Compose([
        A.Flip(p=1),
        A.RandomRotate90(p=1),
        A.RandomSizedCrop((512 - 50, 512 - 50), 512, 512, always_apply=True, p=1),
    # ], bbox_params={'format':'coco', 'min_area': 0.4, 'min_visibility': 0.3, 'label_fields': ['category_id']}, p=1)
    ], bbox_params={'format':'coco', 'label_fields': ['category_id']}, p=1)
    
noise_blur_augmentations = A.Compose([
        A.GaussianBlur(blur_limit=(3, 3), p=0.9),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
        A.ISONoise(p=0.7),
    # ], bbox_params={'format':'coco', 'min_area': 0.4, 'min_visibility': 0.3, 'label_fields': ['category_id']}, p=1)
    ], bbox_params={'format':'coco', 'label_fields': ['category_id']}, p=1)

cr_transforms = [geometrical_augmentations, noise_blur_augmentations]

for epoch in range(num_epochs):

    # train
    astrosam_model.model.train()
	
    epoch_loss, model = astrosam_model.train_validate_step(
        train_dataloader, 
        train_dir, 
        train_gt_masks, 
        train_bboxes, 
        optimizer, 
        mode='train',
        cr_transforms=cr_transforms)
    train_losses.append(epoch_loss)
    
    epoch_loss, model= 0, None
    
    # validate
    astrosam_model.model.eval()
    with torch.no_grad():
        epoch_val_loss, model =  astrosam_model.train_validate_step(
            val_dataloader, 
            valid_dir, 
            val_gt_masks, 
            val_bboxes, 
            optimizer, 
            mode='validate',
        cr_transforms = None)
            
        valid_losses.append(epoch_val_loss)
        
        # # Logging
        if wandb_track:
            wandb.log({'epoch training loss': epoch_loss, 'epoch validation loss': epoch_val_loss})
        
        print(f'EPOCH: {epoch}. Training loss: {epoch_loss}')
        print(f'EPOCH: {epoch}. Validation loss: {epoch_val_loss}.')
        
        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            best_epoch = epoch
            best_model = model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping initiated.")
                early_stop = True
                break

    if epoch % 10 == 0:
        torch.save(best_model.state_dict(), f'{work_dir}/ft_mobile_sam_final{epoch}.pth')

torch.save(best_model.state_dict(), f'{work_dir}/ft_mobile_sam_final_{datetime.now()}.pth')
torch.save(astrosam_model.model.state_dict(), f'{work_dir}/ft_mobile_sam_final_{datetime.now()}_last.pth')

if wandb_track:
    wandb.run.summary["batch_size"] = batch_size
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["best_valid_loss"] = best_valid_loss
    wandb.run.summary["num_epochs"] = num_epochs
    wandb.run.summary["learning rate"] = lr
    wandb.run.summary["weight_decay"] = wd
    wandb.run.summary["# train images"] = len(train_dataloader.dataset)
    wandb.run.summary["# validation images"] = len(val_dataloader.dataset)
    wandb.run.summary["checkpoint"] = mobile_sam_checkpoint
    run.finish()


# In[ ]:


import sys

check_orig = False

if check_orig:
    sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-master/')
    
    # orig_mobile_sam_checkpoint = "./ft_mobile_sam_final80.pth"
    # # orig_mobile_sam_checkpoint = "/workspace/raid/OM_DeepLearning/MobileSAM-master/weights/mobile_sam.pt"
    # orig_mobile_sam_model = sam_model_registry["vit_t"](checkpoint=orig_mobile_sam_checkpoint)
    # orig_mobile_sam_model.to(device);
    # orig_mobile_sam_model.eval();
    
    valid_losses = []
    with torch.no_grad():

        epoch_loss, model = astrosam_model.train_validate_step(
        train_dataloader, 
        train_dir, 
        train_gt_masks, 
        train_bboxes, 
        optimizer, 
        mode='validate')
             
        # epoch_val_loss, _ = astrosam_model.train_validate_step(
        #     val_dataloader, 
        #     valid_dir, 
        #     val_gt_masks,
        #     val_bboxes, 
        #     optimizer, 
        #     mode='validate')
        valid_losses.append(epoch_val_loss)
    print(f'EPOCH: {epoch}. Validation loss: {epoch_val_loss}.')


# In[ ]:

# from IPython.display import display
# display(dot)


# In[ ]:


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