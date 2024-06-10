from pyparsing import with_class
import torch
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.patches as patches
import random 
import torch.nn.functional as F
import albumentations as A

from ..losses import loss_utils
from ..dataset import dataset_utils
from . import predictor_utils
from ..yolo_predictor import yolo_predictor_utils

# ensuring reproducibility
seed=0
import torch.backends.cudnn as cudnn 
random.seed(seed) 
np.random.seed(seed) 
torch.manual_seed(seed) 
cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False) 

# import sys
# import warnings
# import traceback

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = file if hasattr(file, 'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback

# torch.autograd.set_detect_anomaly(True)

class XAMI:
    def __init__(
        self,  
        model, 
        device, 
        predictor, 
        use_yolo_masks=None,
        wt_threshold=None,
        wt_classes_ids=None, 
        apply_segm_CR=False,
        residualAttentionBlock=None):
        
        self.model = model
        self.device = device
        self.predictor = predictor
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.residualAttentionBlock = residualAttentionBlock  
        self.use_yolo_masks = use_yolo_masks # whether to output YOLO masks for faint sources
        self.wt_threshold = wt_threshold # threshold for wavelet transform
        self.wt_classes_ids = wt_classes_ids # which classes to apply wavelet transform on given a threshold
        self.only_geometric = A.Compose([
                # Geometric transformations 
                A.Flip(p=0.5),  
                A.RandomRotate90(p=0.5),  
                A.RandomSizedCrop((492, 492), 512, 512, p=0.6),  
                ], bbox_params={'format': 'coco', 'label_fields': ['category_id']}, p=1)
        self.apply_segm_CR = apply_segm_CR
        
    def one_image_predict(
        self,
        mode,
        image_masks, 
        input_masks, 
        input_bboxes, 
        image_embedding, 
        original_image_size, 
        input_size, 
        input_image, 
        cr_transforms=None, 
        show_plot=True):

        boxes = []
        gt_rle_to_masks, mask_areas = [], []
        gt_numpy_bboxes = []
        for k in image_masks: 
            prompt_box = np.array(input_bboxes[k])
            gt_numpy_bboxes.append(prompt_box)
            box = self.predictor.transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            boxes.append(box_torch)

            # process masks
            rle_to_mask = maskUtils.decode(input_masks[k]) # RLE to array
            gt_rle_to_masks.append(torch.from_numpy(rle_to_mask).to(self.device))
            mask_areas.append(np.sum(rle_to_mask))

        boxes = torch.stack(boxes, dim=0)
        gt_rle_to_masks = torch.stack(gt_rle_to_masks, dim=0)
        
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=boxes, masks=None)

        if torch.isnan(image_embedding).any(): 
            print('NAN in image_embedding')
            return None
    
        gt_threshold_masks = torch.as_tensor(gt_rle_to_masks, dtype=torch.float32).to(device=self.device)

        pred_masks, threshold_masks, iou_predictions = self.decode_and_postprocess(
            image_embedding,
            sparse_embeddings,
            dense_embeddings,
            input_size,
            original_image_size)
        
        # Segmentation and IoU loss
        image_loss, iou_image_loss = self.one_to_one_loss(
            threshold_masks, 
            gt_threshold_masks, 
            iou_predictions, 
            mask_areas, 
            pred_masks,
            focal_loss_factor=20)
     
        # Augmentation
        transformed_losses, cr_loss = [], None
        if cr_transforms is not None:
            transformed_losses, cr_loss = self.augment_with_predict(
                cr_transforms, 
                input_image, 
                gt_numpy_bboxes, 
                gt_rle_to_masks, 
                original_image_size, 
                input_size, 
                threshold_masks,
                apply_CR=self.apply_segm_CR)

        image_loss = image_loss + torch.mean(iou_image_loss) 
        if len(transformed_losses)>0:
            for i in range(len(transformed_losses)):
                image_loss += transformed_losses[i]
            if cr_loss is not None:
                image_loss += cr_loss
                
        # if show_plot:
        #     for i in range(threshold_masks.shape[0]):
        #         fig, axs = plt.subplots(1, 3, figsize=(40, 20))
        #         axs[0].imshow(threshold_masks.permute(1, 0, 2, 3)[0][i].detach().cpu().numpy())
        #         axs[0].set_title(f'gt iou:{ious[i].item()}, \n'+\
        #                 f'pred iou: {iou_predictions[i].item()}\n img: {iou_image_loss[i].item()}\n {image_loss.item()}', fontsize=30)
                    
        #         axs[1].imshow(gt_threshold_masks[i].detach().cpu().numpy())
        #         axs[1].set_title(f'GT masks', fontsize=40)
                
        #         axs[2].imshow(pred_masks[i][0].detach().cpu().numpy())
        #         axs[2].set_title(f'Predicted mask', fontsize=30)
            
        #         plt.show()
        #         plt.close()
        #     fig, axs = plt.subplots(1, 3, figsize=(40, 20))
        #     axs[0].imshow(input_image)
        #     axs[0].set_title(f'{k.split(".")[0]}', fontsize=40)
            
        #     axs[1].imshow(input_image) 
        #     dataset_utils.show_masks(gt_threshold_masks.detach().cpu().numpy(), axs[1], random_color=False)
        #     axs[1].set_title(f'GT masks ', fontsize=40)
            
        #     axs[2].imshow(input_image) 
        #     dataset_utils.show_masks(threshold_masks.permute(1, 0, 2, 3)[0].detach().cpu().numpy(), axs[2], random_color=False)
        #     axs[2].set_title('Pred masks', fontsize=40)
        #     # plt.savefig(f'./{k.split(".")[0]}_masks.png')
        #     plt.show()
        #     plt.close()
            
        # del threshold_masks
        del iou_predictions 
        # del pred_masks, gt_threshold_masks
        del rle_to_mask
        torch.cuda.empty_cache()
        
        if mode == 'validate':
            return image_loss, gt_threshold_masks, threshold_masks>0.5

        return image_loss
    
    def one_image_predict_transform(
            self,
            transformed_image, 
            transformed_masks,
            original_image_size,
            input_size,
            show_plot=True):
        
            boxes = []
            mask_areas = []
            transform = ResizeLongestSide(self.model.image_encoder.img_size)
            input_image = predictor_utils.transform_image(self.model, transform, transformed_image, 'dummy_augm_id', self.device)
            input_image = torch.as_tensor(input_image['image'], dtype=torch.float, device=self.predictor.device) # (B, C, 1024, 1024)
            # image_embedding = self.model.image_encoder(input_image)
            
            # IMAGE ENCODER
            image_embedding = self.model.image_encoder(input_image) # [1, img_emb_size, 64, 64]

            transformed_masks = np.array([transformed_masks[i] for i in range(len(transformed_masks)) \
                if np.any(transformed_masks[i]) and np.sum(transformed_masks[i])>20])
            
            # for each mask, compute the bbox enclosing the mask
            transformed_boxes_from_masks = []
            for k in range(len(transformed_masks)):
                mask_to_box = dataset_utils.mask_to_bbox(transformed_masks[k])
                box = (mask_to_box[0], mask_to_box[1], mask_to_box[2]-mask_to_box[0], mask_to_box[3]-mask_to_box[1])
                transformed_boxes_from_masks.append(box)
                
            transformed_boxes_from_masks = np.array(transformed_boxes_from_masks)
            for mask in transformed_masks:
                mask_area = np.sum(mask)
                mask_areas.append(mask_area)
                
            for k in range(len(transformed_masks)): 
                prompt_box = np.array(dataset_utils.mask_to_bbox(transformed_masks[k])) # XYXY format
                box = self.predictor.transform.apply_boxes(prompt_box, original_image_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
                boxes.append(box_torch)
                
            if len(boxes)>0:
                boxes = torch.stack(boxes, dim=0)
            else:
                print("After augm, image has no bboxes. Skipping...")
                del image_embedding, boxes
                return None, []
                
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=boxes, masks=None)
            gt_threshold_masks = torch.as_tensor(transformed_masks, dtype=torch.float32).to(device=self.device)
            pred_masks, threshold_masks, iou_predictions = self.decode_and_postprocess(
            image_embedding,
            sparse_embeddings,
            dense_embeddings,
            input_size,
            original_image_size)
        
            image_loss, iou_image_loss = self.one_to_one_loss(
            threshold_masks, 
            gt_threshold_masks, 
            iou_predictions, 
            mask_areas, 
            pred_masks)
        
            image_loss = image_loss + torch.mean(iou_image_loss)
            
            # if show_plot:
            #     for i in range(threshold_masks.shape[0]):
            #         fig, axs = plt.subplots(1, 3, figsize=(40, 20))
            #         axs[0].imshow(threshold_masks.permute(1, 0, 2, 3)[0][i].detach().cpu().numpy())
            #         axs[0].set_title(f'gt iou:{ious[i].item()}, \n'+\
            #             f'pred iou: {iou_predictions[i].item()}\n img: {iou_image_loss[i].item()}\n {image_loss.item()}', fontsize=30)
                    
            #         axs[1].imshow(gt_threshold_masks[i].detach().cpu().numpy())
            #         axs[1].set_title(f'GT masks', fontsize=40)
                    
            #         axs[2].imshow(pred_masks[i][0].detach().cpu().numpy())
            #         axs[2].set_title(f'Predicted mask', fontsize=30)
                
            #         plt.show()
            #         plt.close()
                    
            #     fig, axs = plt.subplots(1, 3, figsize=(40, 20))
            #     axs[0].imshow(transformed_image)
            #     axs[0].set_title(f'Image', fontsize=40)
                
            #     axs[1].imshow(transformed_image)
            #     dataset_utils.show_masks(gt_threshold_masks.detach().cpu().numpy(), axs[1], random_color=False)
            #     axs[1].set_title(f'GT masks ', fontsize=40)
                
            #     axs[2].imshow(transformed_image)
            #     dataset_utils.show_masks(threshold_masks.permute(1, 0, 2, 3)[0].detach().cpu().numpy(), axs[2], random_color=False)
            #     axs[2].set_title('Pred masks', fontsize=40)
            #     # plt.savefig(f'./{random.randint(0, 100)}_tr_masks.png')
            #     plt.show()
            #     plt.close()

            # del threshold_masks
            del iou_predictions 
            del pred_masks, gt_threshold_masks
            torch.cuda.empty_cache()

            return image_loss, threshold_masks
            
    def train_validate_step(
        self, 
        dataloader, 
        input_dir, 
        gt_masks, 
        gt_bboxes, 
        optimizer, 
        mode,
        cr_transforms = None,
        scheduler=None,
        ):
        
        assert mode in ['train', 'validate'], "Mode must be 'train' or 'validate'"
        losses = []
        all_gt_masks, all_pred_masks = [], []
        all_image_ids = []
        for inputs in tqdm(dataloader, desc=f'{mode[0].upper()+mode[1:]} Progress', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            batch_loss = torch.tensor(0.0, device=self.device)
            batch_size = len(inputs['image']) # sometimes, at the last iteration, there are fewer images than batch size
            processed_images = batch_size
            
            for i in range(batch_size):
                image_masks = [k for k in gt_masks.keys() if k.startswith(inputs['image_id'][i])]
                input_image = torch.as_tensor(inputs['image'][i], dtype=torch.float, device=self.predictor.device) # (B, C, 1024, 1024)
                image = cv2.imread(input_dir+inputs['image_id'][i])
                original_image_size = image.shape[:-1]
                input_size = (1024, 1024)
                
                # IMAGE ENCODER with residual block
                image_embedding = self.model.image_encoder(input_image) # [1, img_emb_size, 64, 64]

                # RUN PREDICTION ON IMAGE
                if len(image_masks)>0:
                    if mode == 'validate':
                        with torch.no_grad():
                            image_loss, gt_threshold_masks, pred_masks = self.one_image_predict(mode, image_masks, gt_masks, gt_bboxes, image_embedding,
                                                                                                original_image_size, input_size, image, cr_transforms)
                            batch_loss = batch_loss+image_loss
                            all_gt_masks.append(gt_threshold_masks)
                            all_pred_masks.append(pred_masks)
                            all_image_ids.append(inputs['image_id'][i])
                    if mode == 'train':
                            batch_loss = batch_loss+(self.one_image_predict(mode, image_masks, gt_masks, gt_bboxes, image_embedding, 
                                                                original_image_size, input_size, image, cr_transforms))
                else:
                    processed_images -=1
                    
            if processed_images == 0:
                continue
            
            if mode == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                if scheduler is not None: 
                    scheduler.step()
                    # print("Current LR:", optimizer.param_groups[0]['lr'])
                
            losses.append(batch_loss.item()/processed_images)
    
            del batch_loss, image_embedding, input_image, image
            torch.cuda.empty_cache()
   
        return np.mean(losses), all_image_ids, all_gt_masks, all_pred_masks
      
    def run_yolo_sam_epoch(
        self, 
        yolov8_pretrained_model,
        phase, 
        batch_size, 
        image_files, 
        images_dir, 
        num_batches, 
        optimizer=None):
        assert phase in ['train', 'val'], "Phase must be 'train' or 'val'"
        
        if phase == 'train':
            self.model.train()  
        else:
            self.model.eval() 

        epoch_sam_loss = []
        all_preds, all_gts, all_pred_cls, all_gt_cls, all_iou_scores, all_mask_areas, pred_images = [], [], [], [], [], [], []
        for batch_idx in tqdm(range(num_batches), desc=f'{phase[0].upper()+phase[1:]} Progress', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_files = image_files[start_idx:end_idx]

            batch_losses_sam = []

            for image_name in batch_files:
                image_path = images_dir + image_name
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                obj_results = yolov8_pretrained_model.predict(image_path, verbose=False, conf=0.2) 
                gt_masks = yolo_predictor_utils.get_masks_from_image(images_dir, image_name) 
                gt_classes = yolo_predictor_utils.get_classes_from_image(images_dir, image_name) 
                
                if len(obj_results[0]) == 0 or len(gt_masks) == 0: # SAM specific
                    del obj_results
                    continue
                
                gt_masks_tensor = torch.stack([torch.from_numpy(mask).unsqueeze(0) for mask in gt_masks], dim=0).to(self.device)
                
                # set a specific mean for each image
                input_image = predictor_utils.set_mean_and_transform(image, self.model, self.transform, self.device)

                # IMAGE ENCODER
                image_embedding = self.model.image_encoder(input_image) # [1, 256, 64, 64]
                
                mask_areas = [np.sum(gt_mask) for gt_mask in gt_masks]
                input_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
                input_boxes = self.predictor.transform.apply_boxes(input_boxes, image.shape[:-1])
                input_boxes = torch.from_numpy(input_boxes).to(self.device)
                sam_mask, yolo_masks = [], []
                
                if self.use_yolo_masks:
                    non_resized_masks = obj_results[0].masks.data.cpu().numpy()
                
                    for i in range(len(non_resized_masks)):
                            yolo_masks.append(cv2.resize(non_resized_masks[i], image.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)) 

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=input_boxes, masks=None)
                pred_masks, threshold_masks, iou_predictions = self.decode_and_postprocess(
                image_embedding,
                sparse_embeddings,
                dense_embeddings,
                (1024, 1024),
                image.shape[:-1])
                
                sam_mask_pre = (threshold_masks > 0.5)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))

                # segm_loss_sam, preds, gts, gt_classes_match, pred_classes_match, ious_match, mask_areas = loss_utils.segm_loss_match_iou_based(
                #     self.use_yolo_masks,
                #     threshold_masks, 
                #     gt_masks_tensor, 
                #     obj_results[0].boxes.cls.detach().cpu().numpy(), 
                #     gt_classes, 
                #     iou_predictions,
                #     mask_areas,
                #     image,
                #     yolo_masks,
                #     self.wt_classes_ids,
                #     self.wt_threshold)
                
                segm_loss_sam, preds, gts, gt_classes_match, pred_classes_match, ious_match  = loss_utils.segm_loss_match_hungarian(
                    self.use_yolo_masks,
                    threshold_masks,
                    gt_masks_tensor, 
                    obj_results[0].boxes.cls.detach().cpu().numpy(), 
                    gt_classes, 
                    iou_predictions,
                    mask_areas,
                    image,
                    yolo_masks,
                    self.wt_classes_ids,
                    self.wt_threshold)
                    
                # ious, iou_image_loss = predictor_utils.calculate_iou_loss(np.array(preds), np.array(gts), ious_match, mask_areas)
                threshold_preds = np.array([preds[i][0]>0.5*1 for i in range(len(preds))])
                all_preds.append(threshold_preds)
                all_gts.append(gts)
                all_gt_cls.append(gt_classes_match)
                all_pred_cls.append(pred_classes_match)
                all_iou_scores.append(ious_match)
                all_mask_areas.append(mask_areas)
                pred_images.append(image_name)
                
                batch_losses_sam.append(segm_loss_sam) #+iou_image_loss)
                del sparse_embeddings, dense_embeddings, image_embedding
                del segm_loss_sam, threshold_masks, pred_masks, sam_mask_pre
                torch.cuda.empty_cache()

                # if phase == 'val' and len(gt_masks) == 0:
                #     print(image_name)
                #     fig, axes = plt.subplots(1, 4, figsize=(18, 6)) 
                    
                #     # Plot 1: GT Masks
                #     axes[0].imshow(image)
                #     axes[0].set_title('GT Masks')
                #     if len(gt_masks) > 0:
                #         dataset_utils.show_masks(gt_masks_tensor.squeeze(1).squeeze(1).detach().cpu().numpy(), axes[0], random_color=True)
        
                #     # Plot 2: YOLO Masks
                #     axes[1].imshow(image)
                #     axes[1].set_title('YOLOv8n predicted Masks')
                #     dataset_utils.show_masks(yolo_masks, axes[1], random_color=True)
                    
                #     # Plot 3: Bounding Boxes
                #     image1 = cv2.resize(image, (1024, 1024))
                #     for bbox in input_boxes:
                #         x1, y1, x2, y2 = bbox.detach().cpu().numpy()
                #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #         cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                #     image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                #     axes[2].imshow(image1_rgb)
                #     axes[2].set_title('YOLOv8n predicted Bboxes')
                    
                #     # Plot 4: SAM Masks
                #     axes[3].imshow(image)
                #     dataset_utils.show_masks(threshold_preds, axes[3], random_color=True)
                #     axes[3].set_title('MobileSAM predicted masks')
                #     plt.tight_layout() 
                #     # plt.savefig(f'./plots/combined_plots.png')
                #     plt.show()

                del obj_results, image, sam_mask, yolo_masks, input_boxes
                del threshold_preds, preds, gts, gt_classes_match, pred_classes_match, ious_match
                torch.cuda.empty_cache()
                
            mean_loss_sam = torch.mean(torch.stack(batch_losses_sam))
            epoch_sam_loss.append(mean_loss_sam.item())
            
            if phase == 'train' and optimizer is not None:
                optimizer.zero_grad()
                mean_loss_sam.backward()
                optimizer.step()

        return np.mean(epoch_sam_loss), all_preds, all_gts, all_gt_cls, all_pred_cls, all_iou_scores, all_mask_areas, pred_images
    
    def add_residual(self, image, norm_shape): # [1, 3, 1024, 1024]
        
        transform_layer = nn.Sequential(
            nn.Conv2d(3, norm_shape, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(norm_shape, norm_shape, kernel_size=3, stride=4, padding=1),
            nn.ReLU()
        ).to(self.device)
                
        image_embedding = transform_layer(image)
            
        # Flatten spatial dimensions
        sequence_length = image_embedding.shape[-2] * image_embedding.shape[-1]
        batch_size = image_embedding.shape[0]
        d_model = image_embedding.shape[1]

        # Reshape the tensor to [sequence_length, batch_size, d_model]
        reshaped_tensor = image_embedding.permute(2, 3, 0, 1).reshape(sequence_length, batch_size, d_model)
        output_tensor = self.residualAttentionBlock(reshaped_tensor)
        residual_image_embedding = output_tensor.view(image_embedding.shape[-2], image_embedding.shape[-1], batch_size, d_model).permute(2, 3, 0, 1)
        
        return residual_image_embedding

    def augment_with_predict(self, cr_transforms, input_image, gt_numpy_bboxes, masks, original_image_size, input_size, threshold_masks, apply_CR=True):
        
        transformed_losses = []
        cr_CE_loss = None
        
        for cr_transform in cr_transforms:
            bboxes = np.array([np.array([box[0], box[1], box[2]-box[0], box[3]-box[1]]) for box in gt_numpy_bboxes])
            seed = random.randint(0, 10000)
            self.set_seed(seed)
            
            transformed = cr_transform(
                image=input_image, 
                bboxes=bboxes.reshape(-1,4), 
                masks=masks.detach().cpu().numpy(),
                category_id= [1] * bboxes.shape[0]) # we don't use labels here

            transformed_image = transformed['image']
            transformed_masks = transformed['masks']
            transformed_loss, transformed_threshold_masks = self.one_image_predict_transform(
                transformed_image, 
                transformed_masks,
                original_image_size, 
                input_size,
                self.device)
            
            if transformed_loss is None or len(transformed_threshold_masks) == 0:
                continue

            transformed_losses.append(transformed_loss)
            
            # consistency regularization
            if apply_CR:
                
                self.set_seed(seed)
                
                concatenated_pred_masks = (torch.mean(threshold_masks, dim=0, keepdim=True)*255).to(torch.uint8)
                concatenated_pred_masks = torch.stack([concatenated_pred_masks, concatenated_pred_masks, concatenated_pred_masks], dim=1)[0] # To RGB
                concatenated_transformed_pred_masks = torch.mean(transformed_threshold_masks, dim=0, keepdim=True) # type: ignore
                
                # Apply the same transformation to the concatenated predicted masks
                concatenated_pred_masks = concatenated_pred_masks.permute(2, 3, 0, 1).squeeze(dim=-1)
                concatenated_pred_masks = concatenated_pred_masks.detach().cpu().numpy()
                
                pred_transformed_masks_image = concatenated_pred_masks
                if self.only_geometric is not None: # in case there are geometric transformations
                    pred_transformed_masks_image = self.only_geometric( # transform concatenated_pred_masks with dummy input
                        image=concatenated_pred_masks,
                        bboxes=[[0, 0, 1, 1]],
                        masks=np.zeros((1, concatenated_pred_masks.shape[0],concatenated_pred_masks.shape[1]), dtype=np.uint8),
                        category_id=[1] 
                    )['image']/255.0
                
                pred_transformed_masks_image = pred_transformed_masks_image[..., 0]  # Taking the red channel
                pred_transformed_masks_image = torch.tensor(pred_transformed_masks_image).unsqueeze(0).unsqueeze(0).to(self.device)

                # Compute the cross-entropy loss
                cr_CE_loss = F.binary_cross_entropy_with_logits(pred_transformed_masks_image, concatenated_transformed_pred_masks)
                # def plot_tensor(tensor, title):
                #     tensor = tensor.squeeze().cpu().detach().numpy()
                #     plt.imshow(tensor)
                #     plt.title(title)
                #     plt.axis('off')

                # plt.figure(figsize=(18, 6))

                # plt.subplot(1, 3, 1)
                # plot_tensor(pred_transformed_masks_image, 'Predicted transformed')

                # plt.subplot(1, 3, 2)
                # plot_tensor(concatenated_transformed_pred_masks, 'Transformed predicted. CR loss: {:.4f}'.format(cr_CE_loss.item()))
                
                # plt.subplot(1, 3, 3)
                # plt.imshow(input_image)
                # plt.title('Image')
                # plt.savefig(f'CR_masks {seed}.png')
                # plt.show()
        
        return transformed_losses, cr_CE_loss
    
    def decode_and_postprocess(
        self,
        image_embedding, 
        sparse_embeddings, 
        dense_embeddings, 
        input_size, 
        original_image_size):
       
        # MASK DECODER
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )

        max_low_res_masks = torch.zeros((low_res_masks.shape[0], 1, 256, 256))
        max_ious = torch.zeros((iou_predictions.shape[0], 1))

        # Select the masks corresponding to the maximum predicted IoU
        for i in range(low_res_masks.shape[0]):
            max_iou_index = torch.argmax(iou_predictions[i])
            max_low_res_masks[i] = low_res_masks[i][max_iou_index].unsqueeze(0)
            max_ious[i] = iou_predictions[i][max_iou_index]

        low_res_masks = max_low_res_masks
        iou_predictions = max_ious

        # Post-process masks
        pred_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size).to(self.device)
        threshold_masks = torch.sigmoid(10 * (pred_masks - self.model.mask_threshold))  # Apply sigmoid with steepness

        return pred_masks, threshold_masks, iou_predictions 

    def one_to_one_loss(
        self, 
        threshold_masks, 
        gt_threshold_masks, 
        iou_predictions, 
        mask_areas, 
        pred_masks,
        focal_loss_factor=20):
        
        # IoU loss
        ious, iou_image_loss = predictor_utils.calculate_iou_loss(threshold_masks, gt_threshold_masks, iou_predictions, mask_areas)

        # Segmentation losses
        focal = loss_utils.focal_loss_per_mask_pair(torch.squeeze(pred_masks, dim=1), gt_threshold_masks, mask_areas)
        dice = loss_utils.dice_loss_per_mask_pair(torch.squeeze(threshold_masks, dim=1), gt_threshold_masks, mask_areas)
        # mse = F.mse_loss(threshold_masks.squeeze(1), gt_threshold_masks)

        # Combine losses
        image_loss = focal_loss_factor * focal + dice
        
        return image_loss, iou_image_loss
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    