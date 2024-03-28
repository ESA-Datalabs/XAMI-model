import torch
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from losses import loss
from dataset import dataset_utils

class AstroSAM:
    def __init__(self, model, device, predictor):
        self.model = model
        self.device = device
        self.predictor = predictor
            
    def one_image_predict(
        self,
        image_masks, 
        input_masks, 
        input_bboxes, 
        image_embedding, 
        original_image_size, 
        input_size, 
        negative_mask, 
        input_image, 
        cr_transforms, 
        show_plot=True):

        ious = []
        image_loss=[]
        boxes, masks, coords, coords_labels = [], [], [], []
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
            mask_input_torch = torch.as_tensor(rle_to_mask, dtype=torch.float, device=self.predictor.device).unsqueeze(0)
            mask_input_torch = F.interpolate(
                mask_input_torch.unsqueeze(0), 
                size=(256, 256), 
                mode='bilinear', 
                align_corners=False)
            masks.append(mask_input_torch.squeeze(0))
            mask_areas.append(np.sum(rle_to_mask))

            # process coords and labels
            x_min, y_min, x_max, y_max = input_bboxes[k]
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            point_coords = np.array([(input_bboxes[k][2]+input_bboxes[k][0])/2.0, (input_bboxes[k][3]+input_bboxes[k][1])/2.0])
            point_labels = np.array([1])
            point_coords = self.predictor.transform.apply_coords(point_coords, original_image_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.predictor.device).unsqueeze(0)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.predictor.device)
            coords.append(coords_torch)
            coords_labels.append(labels_torch)

        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)
        coords = torch.stack(coords, dim=0)
        coords_labels = torch.stack(coords_labels, dim=0)
        points = (coords, coords_labels)
        gt_rle_to_masks = torch.stack(gt_rle_to_masks, dim=0)
        
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
        points=None,
        boxes=boxes,
        masks=None, 
        )

        del box_torch, coords_torch, labels_torch
        torch.cuda.empty_cache()
        
        low_res_masks, iou_predictions = self.model.mask_decoder( # iou_pred [N, 1] where N - number of masks
        image_embeddings=image_embedding,
        image_pe=self.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False, # True value works better for ambiguous prompts (single points)
        )

        iou_image_loss = []
        pred_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size).to(self.device)

        # Gaussian kernel
        kernel_size = 5
        sigma = 2
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma).to(self.device)

        # Apply the 2D Gaussian filter (ensuring images and gaussian_kernel are on the same device)
        pred_masks = torch.nn.functional.conv2d(pred_masks, gaussian_kernel, padding=kernel_size//2)
        threshold_mask = torch.sigmoid(10 * (pred_masks - self.model.mask_threshold)) # sgmoid with steepness
        gt_threshold_mask = torch.as_tensor(gt_rle_to_masks, dtype=torch.float32) 
        numpy_gt_threshold_mask = gt_threshold_mask.contiguous().detach().cpu().numpy()
        mask_result = torch.squeeze(threshold_mask, dim=1).detach().cpu().numpy()
        for i in range(threshold_mask.shape[0]):
            iou_per_mask = loss.iou_single(threshold_mask[i][0], gt_threshold_mask[i])
            ious.append(iou_per_mask)
            iou_image_loss.append(torch.abs(iou_predictions.permute(1, 0)[0][i] - iou_per_mask))

        # compute weighted dice loss (smaller weights on smaller objects)
        focal = loss.focal_loss_per_mask_pair(torch.squeeze(pred_masks, dim=1), gt_threshold_mask, mask_areas)
        dice = loss.dice_loss_per_mask_pair(torch.squeeze(threshold_mask, dim=1), gt_threshold_mask, mask_areas, negative_mask) 
        image_loss.append(20 * focal + dice ) # used in SAM paper
        # Apply consistency regulation
        if cr_transforms:
            for cr_transform in cr_transforms:
                bboxes = [np.array([box[0], box[1], box[2]-box[0], box[3]-box[1]]) for box in gt_numpy_bboxes]
                transformed = cr_transform(
                    image=input_image, 
                    bboxes=bboxes.reshape(-1,4), # flatten bboxes
                    masks=gt_rle_to_masks.detach().cpu().numpy(),
                    category_id=[0] * boxes.shape[0]) # I don't have labels for the moment 
                
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_masks = transformed['masks']
                tranformed_loss = one_image_predict_transform(
                    transformed_image, 
                    transformed_bboxes, 
                    transformed_mask,
                    original_image_size, 
                    self.device)
        image_loss = torch.stack(image_loss)
        iou_image_loss = torch.stack(iou_image_loss)
        image_loss = 0.6 * torch.mean(image_loss) + 0.4 * torch.mean(iou_image_loss) #* loss_scaling_factor

        # if show_plot:
        #     for i in range(threshold_mask.shape[0]):
        #         fig, axs = plt.subplots(1, 3, figsize=(40, 20))
        #         axs[0].imshow(threshold_mask.permute(1, 0, 2, 3)[0][i].detach().cpu().numpy())
        #         axs[0].set_title(f'gt iou:{ious[i].item()}, \n pred iou: {iou_predictions[i].item()}\n img: {iou_image_loss[i].item()}', fontsize=30)
                
        #         axs[1].imshow(gt_threshold_mask[i].detach().cpu().numpy())
        #         axs[1].set_title(f'GT masks', fontsize=40)
                
        #         axs[2].imshow(pred_masks[i][0].detach().cpu().numpy())
        #         axs[2].set_title(f'Predicted mask', fontsize=30)
            
        #         plt.show()
        #         plt.close()
        #     print(f'{k.split(".")[0]}')
        #     fig, axs = plt.subplots(1, 3, figsize=(40, 20))
        #     axs[0].imshow(input_image)
        #     axs[0].set_title(f'{k.split(".")[0]}', fontsize=40)
            
        #     axs[1].imshow(input_image) 
        #     dataset_utils.show_masks(gt_threshold_mask.detach().cpu().numpy(), axs[1], random_color=False)
        #     axs[1].set_title(f'GT masks ', fontsize=40)
            
        #     axs[2].imshow(input_image) 
        #     dataset_utils.show_masks(threshold_mask.permute(1, 0, 2, 3)[0].detach().cpu().numpy(), axs[2], random_color=False)
        #     axs[2].set_title('Pred masks', fontsize=40)
            
        #     plt.show()
        #     plt.close()
            
        del threshold_mask
        del numpy_gt_threshold_mask 
        del low_res_masks, iou_predictions 
        del pred_masks, gt_threshold_mask
        del rle_to_mask
        torch.cuda.empty_cache()


        return image_loss

    def one_image_predict_transform(
        self,
        transformed_image, 
        transformed_bboxes, 
        transformed_mask,
        original_image_size,
        device,
        show_plot=True):

        boxes = []
        ious = []
        image_loss=[]
        gt_rle_to_masks, mask_areas = [], []
        image_embedding = self.model.image_encoder(transformed_image)

        for k in range(len(transformed_mask)): 
            prompt_box = np.array(transformed_bboxes[k])
            box = self.predictor.transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            boxes.append(box_torch)

            # process masks
            rle_to_mask = maskUtils.decode(gt_masks[k]) # RLE to array
            gt_rle_to_masks.append(torch.from_numpy(rle_to_mask).to(device))
    
        boxes = torch.stack(boxes, dim=0)
        gt_rle_to_masks = torch.stack(gt_rle_to_masks, dim=0)
        
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
        points=None,
        boxes=boxes,
        masks=None, 
        )

        del box_torch
        torch.cuda.empty_cache()
        
        low_res_masks, iou_predictions = self.model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=self.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False, # True value works better for ambiguous prompts (single points)
        )

        # plt.imshow(low_res_masks[0][0].detach().cpu().numpy())
        # plt.show()
        # plt.close()
        
        pred_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        threshold_mask = torch.sigmoid(pred_masks - self.model.mask_threshold)
        gt_threshold_mask = torch.as_tensor(gt_rle_to_masks>0.5, dtype=torch.float32) 
        numpy_gt_threshold_mask = gt_threshold_mask.contiguous().detach().cpu().numpy()
        mask_result = torch.squeeze(threshold_mask, dim=1).detach().cpu().numpy()
        ious.append(loss.iou_single(threshold_mask, numpy_gt_threshold_mask))
        print(loss.iou_single(threshold_mask, numpy_gt_threshold_mask))

        # compute weighted dice loss (smaller weights on smaller objects)
        focal = loss.focal_loss_per_mask_pair(torch.squeeze(pred_masks, dim=1), gt_threshold_mask, mask_areas)
        dice = loss.dice_loss_per_mask_pair(torch.squeeze(threshold_mask, dim=1), gt_threshold_mask, mask_areas, negative_mask) 
        image_loss.append(20 * focal + dice ) # used in SAM paper
        
        del threshold_mask
        del numpy_gt_threshold_mask 
        del low_res_masks, iou_predictions 
        del pred_masks, gt_threshold_mask
        del rle_to_mask
        torch.cuda.empty_cache()

        image_loss = torch.stack(image_loss)
        image_loss = torch.mean(image_loss) 

        if show_plot:
            fig, axs = plt.subplots(1, 3, figsize=(40, 20))
            axs[0].imshow(input_image) 
            axs[0].set_title(f'{k.split(".")[0]}', fontsize=40)
            
            axs[1].imshow(input_image)
            dataset_utils.show_masks(mask_result, axs[1], random_color=False)
            axs[1].set_title('Predicted masks', fontsize=40)
            
            axs[2].imshow(input_image) 
            dataset_utils.show_masks(gt_rle_to_masks.detach().cpu().numpy(), axs[2], random_color=False)
            axs[2].set_title('Ground truth masks', fontsize=40)
            
            plt.show()
            plt.close()

        return image_loss

    def train_validate_step(
        self, 
        dataloader, 
        input_dir, 
        gt_masks, 
        gt_bboxes, 
        optimizer, 
        mode='train',
        cr_transforms = None):
        
        assert mode in ['train', 'validate'], "Mode must be 'train' or 'validate'"
        losses = []

        for inputs in tqdm(dataloader, desc=f'{mode[0].upper()+mode[1:]} Progress', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            batch_loss = 0.0
            batch_size = len(inputs['image']) # sometimes, at the last iteration, there are fewer images than batch size
            for i in range(batch_size):
                image_masks = [k for k in gt_masks.keys() if k.startswith(inputs['image_id'][i])]
                input_image = torch.as_tensor(inputs['image'][i], dtype=torch.float, device=self.predictor.device) # (B, C, 1024, 1024)
                # plt.imshow(input_image[0].permute(1,2,0).detach().cpu().numpy())
                # plt.show()
                # plt.close()
                image = cv2.imread(input_dir+inputs['image_id'][i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_image_size = image.shape[:-1]
                input_size = (1024, 1024)

                # IMAGE ENCODER
                image_embedding = self.model.image_encoder(input_image)
                
                # negative_mask has the size of the image
                negative_mask = np.where(image>0, True, False)
                negative_mask = torch.from_numpy(negative_mask)  
                negative_mask = negative_mask.permute(2, 0, 1)
                negative_mask = negative_mask[0]
                negative_mask = negative_mask.unsqueeze(0).unsqueeze(0)
                negative_mask = negative_mask.to(self.device)
                     
                # RUN PREDICTION ON IMAGE
                   
                if mode == 'validate':
                    with torch.no_grad():
                        if len(image_masks)>0:
                            batch_loss += (self.one_image_predict(image_masks, gt_masks, gt_bboxes, image_embedding, 
                                                            original_image_size, input_size, negative_mask, image, cr_transforms)) 
                        else:
                            print(f"{inputs['image_id'][i]} has no annotations❗️")

                if mode == 'train':
                    if len(image_masks)>0:
                        batch_loss += (self.one_image_predict(image_masks, gt_masks, gt_bboxes, image_embedding, 
                                                        original_image_size, input_size, negative_mask, image, cr_transforms)) 
                    else:
                        print(f"{inputs['image_id'][i]} has no annotations❗️")
                    
            if mode == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                del image_embedding, negative_mask, input_image, image
                torch.cuda.empty_cache()

                losses.append(batch_loss.item()/batch_size) #/loss_scaling_factor)
            else:
                losses.append(batch_loss.detach().cpu()/batch_size) #/loss_scaling_factor)
			
                
        return np.mean(losses), self.model
    
        
    def create_gaussian_kernel(self, kernel_size=5, sigma=2, in_channels=1, out_channels=1):
        """Generate a 2D Gaussian kernel."""
        # Create a coordinate grid
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*np.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
        # Make sure the sum of values in the gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(out_channels, in_channels, 1, 1)
        
        return gaussian_kernel

    
    def check_requires_grad(self, show=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and show:
                print("✅ Param", name, " requires grad.")
            elif param.requires_grad == False:
                print("❌ Param", name, " doesn't require grad.")
