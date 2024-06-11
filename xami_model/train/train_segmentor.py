import os
import sys
import yaml
import torch
import numpy as np
import json
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from datetime import datetime
from pycocotools import mask as maskUtils
from torch.utils.data import DataLoader
from segment_anything.utils.transforms import ResizeLongestSide

from xami_model.dataset import dataset_utils, load_dataset
from xami_model.model_predictor import xami, predictor_utils
from xami_model.mobile_sam.mobile_sam import sam_model_registry, SamPredictor

# For reproducibility
seed = 0
import torch.backends.cudnn as cudnn 
np.random.seed(seed) 
torch.manual_seed(seed) 
cudnn.benchmark, cudnn.deterministic = False, True

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config):
    # Environment setup
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_visible_devices']

    # Load configuration parameters
    kfold_iter = config['kfold_iter']
    device_id = int(config['device_id'])
    lr = float(config['learning_rate'])
    wd = float(config['weight_decay'])
    wandb_track = config['wandb_track']
    num_epochs = int(config['num_epochs'])
    use_lr_initial_decay = config['use_lr_initial_decay']
    use_on_plateau_lr_sched = config['use_on_plateau_lr_sched']
    n_epochs_stop = int(config['n_epochs_stop'])
    use_CR = config['use_CR']
    work_dir = config['work_dir']
    input_dir = config['input_dir']
    # The batch size before applying augmentations. The effective batch size is batch_size * (#augmentations + 1)
    # If the effective batch_size is bigger than 8, the model may run into OOM errors due to allocation of memory
    batch_size = int(config['initial_batch_size'])
    mobile_sam_checkpoint = config['mobile_sam_checkpoint']
    
    # Create working directory
    work_dir = predictor_utils.get_next_directory_name(work_dir)
    os.makedirs(work_dir)
    print(f"Working directory: {work_dir}")

    # Setup device
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    train_dir = os.path.join(input_dir, 'train/')
    valid_dir = os.path.join(input_dir, 'valid/')
    json_train_path = os.path.join(train_dir, '_annotations.coco.json')
    json_valid_path = os.path.join(valid_dir, '_annotations.coco.json')

    with open(json_train_path) as f1, open(json_valid_path) as f2:
        train_data_in = json.load(f1)
        valid_data_in = json.load(f2)

    training_image_paths = [os.path.join(train_dir, image['file_name']) for image in train_data_in['images']]
    val_image_paths = [os.path.join(valid_dir, image['file_name']) for image in valid_data_in['images']]

    train_data = dataset_utils.load_json(json_train_path)
    valid_data = dataset_utils.load_json(json_valid_path)
    
    train_gt_masks, train_bboxes, train_classes, train_class_categories = dataset_utils.get_coords_and_masks_from_json(
        train_dir, train_data) 
    val_gt_masks, val_bboxes, val_classes, val_class_categories = dataset_utils.get_coords_and_masks_from_json(
        valid_dir, valid_data)

    # Initialize model
    model = sam_model_registry["vit_t"](checkpoint=mobile_sam_checkpoint)
    model.to(device)
    predictor = SamPredictor(model)
    xami_model_instance = xami.XAMI(model, device, predictor, apply_segm_CR=use_CR)

    if wandb_track:
        import wandb
        wandb.login()
        run = wandb.init(project="sam", name=f"sam_{kfold_iter}_{datetime.now()}")
        wandb.watch(xami_model_instance.model, log='all', log_graph=True)

    # Prepare data loaders
    transform = ResizeLongestSide(xami_model_instance.model.image_encoder.img_size)
    train_set = load_dataset.ImageDataset(training_image_paths, xami_model_instance.model, transform, device) 
    val_set = load_dataset.ImageDataset(val_image_paths, xami_model_instance.model, transform, device) 
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Optimizer
    for name, param in xami_model_instance.model.named_parameters():
        param.requires_grad = 'mask_decoder' in name

    parameters_to_optimize = [param for param in xami_model_instance.model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters_to_optimize, lr=lr, weight_decay=wd) 

    # Scheduler
    scheduler = None
    if use_lr_initial_decay:
        initial_lr = lr
        final_lr = float(config['final_lr'])
        total_steps = config['total_steps']
        lr_decrement = (initial_lr - final_lr) / total_steps

        def lr_lambda(current_step):
            if current_step < total_steps:
                return 1 - current_step * lr_decrement / initial_lr
            return final_lr / initial_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    # Training loop
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')

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

    print(f"🚀  Training {xami_model_instance.model.__class__.__name__} with {len(training_image_paths)} training images and {len(val_image_paths)} validation images.")
    print(f"🚀  Training for {num_epochs} epochs with effective batch size {batch_size * (len(cr_transforms) + 1)} and learning rate {lr}.")
    print(f"🚀  Initial learning rate: {lr}. Final learning rate: {final_lr} after {total_steps} steps. Weight decay: {wd}.")
    print(f"🚀  Using learning rate initial decay scheduler: {use_lr_initial_decay}. ")
    print(f"🚀  Early stopping after {n_epochs_stop} epochs without improvement.")
    print(f"🚀  Training started.\n")

    iou_eval_thresholds = [0.5, 0.75, 0.9]
    
    for epoch in range(num_epochs):
        # Train
        xami_model_instance.model.train()
        epoch_loss, _, _, _ = xami_model_instance.train_validate_step(
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
        xami_model_instance.model.eval()
        with torch.no_grad():
            epoch_val_loss, all_image_ids, all_gt_masks, all_pred_masks =  xami_model_instance.train_validate_step(
                val_dataloader, 
                valid_dir, 
                val_gt_masks, 
                val_bboxes, 
                optimizer, 
                mode='validate',
                cr_transforms=[],
                scheduler=None)
            
            valid_losses.append(epoch_val_loss)
            p_metric_name, p_means, p_stds = predictor_utils.compute_scores('precision', all_pred_masks, all_gt_masks, iou_eval_thresholds)
            r_metric_name, r_means, r_stds = predictor_utils.compute_scores('recall', all_pred_masks, all_gt_masks, iou_eval_thresholds)
            f_metric_name, f_means, f_stds = predictor_utils.compute_scores('f1_score', all_pred_masks, all_gt_masks, iou_eval_thresholds)
            a_metric_name, a_means, a_stds = predictor_utils.compute_scores('accuracy', all_pred_masks, all_gt_masks, iou_eval_thresholds)
            print('Precision', p_means, 'Recall', r_means, 'F1-score', f_means, 'Accuracy', a_means)
        
        # Logging
        if wandb_track:
            wandb.log({'epoch training loss': epoch_loss, 'epoch validation loss': epoch_val_loss})
            wandb.log({'Precision': p_means, 'Recall': r_means, 'F1-score': f_means, 'Accuracy': a_means})

        print(f'EPOCH: {epoch}. Training loss: {epoch_loss}')
        print(f'EPOCH: {epoch}. Validation loss: {epoch_val_loss}.')

        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            best_epoch = epoch
            best_model = xami_model_instance.model
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
    torch.save(xami_model_instance.model.state_dict(), f'{work_dir}/sam_{kfold_iter}_{datetime.now()}_last.pth')

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_yolo_sam.py <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = read_config(config_path)
    main(config)