from sympy import I
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def iou_single(pred_mask, gt_mask):
    """Compute IoU between a single predicted mask and a single ground truth mask."""
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask) + torch.sum(gt_mask) - intersection
    if union == 0:
        return torch.tensor(0.)
    else:
        return intersection / union

def compute_median_intensity(mask, image):
    """
    Compute the median intensity for the object represented by the mask.
    
    Parameters:
    - mask: Binary mask of the object (Tensor of shape [1, H, W])
    - image: Image tensor corresponding to the mask (Tensor of shape [H, W, C])
    
    Returns:
    - median_intensity: median intensity of the object
    """
    print(mask.shape, image.shape)
    
    mask = mask.squeeze(0).unsqueeze(2).repeat(1, 1, 3)
    object_pixels = image * mask.detach().cpu().numpy()
    print(np.max(object_pixels), np.min(object_pixels)) 
    
    median_intensity = np.median(object_pixels[object_pixels>0])/image.shape[0]
    plt.imshow(image * mask.detach().cpu().numpy())
    plt.show()
    plt.close()
    
    return median_intensity
	
def compute_iou_matrix(pred_masks, gt_masks):
    """
    Compute a matrix of IoU scores for each pair of predicted and GT masks.
    
    Parameters:
    - pred_masks: Tensor of shape [num_pred, H, W]
    - gt_masks: Tensor of shape [num_gt, H, W]
    
    Returns:
    - iou_matrix: Tensor of shape [num_pred, num_gt]
    """
    num_pred = pred_masks.shape[0]
    num_gt = gt_masks.shape[0]
    iou_matrix = torch.zeros((num_pred, num_gt))
    
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = iou_single(pred_masks[i], gt_masks[j])
    
    return iou_matrix

# inspired from here: https://www.kaggle.com/code/aakashnain/diving-deep-into-focal-loss
def compute_focal_loss(y_pred, y_true, alpha=0.7, gamma=2.0):
    """
    Compute the focal loss between `y_true` and `y_pred`.
    
    Args:
    - y_true (torch.Tensor): Ground truth labels, shape [H, W]
    - y_pred (torch.Tensor): Predicted logits, shape [H, W]
    - alpha (float): Weighting factor.
    - gamma (float): Focusing parameter.

    Returns:
    - torch.Tensor: Computed focal loss.
    """
    p = torch.sigmoid(y_pred)
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    p_t = y_true * p + (1 - y_true) * (1 - p)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = torch.pow((1 - p_t), gamma)
    focal_loss = alpha_factor * modulating_factor * bce

    return torch.mean(focal_loss)

def compute_dice_loss(pred_mask, gt_mask):
    """
    Compute the Dice loss between a single predicted mask and a single ground truth mask.
    Both masks should be floating-point tensors with the same shape.
    """
    pred_flat = torch.flatten(pred_mask)
    gt_flat = torch.flatten(gt_mask) 
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum()
    dice_coefficient = (2. * intersection + 1e-6) / (union + 1e-6)  # Adding a small epsilon to avoid division by zero

    return 1 - dice_coefficient

def segm_loss_match_hungarian(
	pred_masks, 
	gt_masks, 
	all_pred_classes, 
	all_gt_classes, 
	iou_scores,
    mask_areas=None):

    # Compute IoU matrix for all pairs
    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)  
    preds = []
    gts = []
    gt_classes, pred_classes, iou_scores_sam = [], [], []
    # Hungarian matching
    cost_matrix = -iou_matrix  # Negate IoU for minimization
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().numpy())

    # Compute loss for matched pairs
    total_dice_loss = 0
    total_focal_loss = 0
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        preds.append(pred_masks[pred_idx])
        gts.append(gt_masks[gt_idx])
        pred_classes.append(int(all_pred_classes[pred_idx]))
        gt_classes.append(all_gt_classes[gt_idx])
        iou_scores_sam.append(iou_scores[pred_idx])
        # median_intensity = compute_average_intensity(gt_masks[gt_idx], image) # weight the loss by object intensity
        dice_loss = compute_dice_loss(pred_masks[pred_idx], gt_masks[gt_idx])
        focal_loss = compute_focal_loss(pred_masks[pred_idx].float(), gt_masks[gt_idx].float())
        if mask_areas is not None:
            total_dice_loss += (dice_loss * mask_areas[gt_idx]/sum(mask_areas)) # weighted loss given mask size
            total_focal_loss += (focal_loss * mask_areas[gt_idx]/sum(mask_areas)) # weighted loss given mask size

    # Normalize the losses
    mean_dice_loss = total_dice_loss / len(row_ind)
    mean_focal_loss = total_focal_loss / len(row_ind)

    # Combine losses
    total_loss = mean_dice_loss + 20 * mean_focal_loss

    return total_loss, preds, gts, gt_classes, pred_classes, iou_scores_sam 

def segm_loss_match_hungarian_compared(
	pred_masks,
    yolo_pred_masks,
	gt_masks, 
	all_pred_classes, 
	all_gt_classes, 
	iou_scores,
    image = None,
    mask_areas=None):

    # Compute IoU matrix for all pairs
    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)  
    preds = []
    gts = []
    gt_classes, pred_classes, iou_scores_sam = [], [], []
    # Hungarian matching
    cost_matrix = -iou_matrix  # Negate IoU for minimization
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().numpy())

    # Compute loss for matched pairs
    total_dice_loss = 0
    total_focal_loss = 0
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        
        dice_loss = compute_dice_loss(pred_masks[pred_idx], gt_masks[gt_idx])
        focal_loss = compute_focal_loss(pred_masks[pred_idx].float(), gt_masks[gt_idx].float())
        dice_loss_yolo = compute_dice_loss(yolo_pred_masks[pred_idx], gt_masks[gt_idx])
        focal_loss_yolo = compute_focal_loss(yolo_pred_masks[pred_idx].float(), gt_masks[gt_idx].float())
        
        good_mask = pred_masks[pred_idx]
        if image is not None:
            # check if the original image bounded by the ground truth mask has a higher average intensity than the predicted mask
            median_intensity_gt = compute_median_intensity(gt_masks[gt_idx], image)
            print("median intensity", median_intensity_gt)

            if median_intensity_gt < 0.1: # TODO: make this less hard-coded
                print("median intensity of the object is too low")
                dice_loss = dice_loss_yolo
                focal_loss = focal_loss_yolo
                good_mask = yolo_pred_masks[pred_idx]
                
        # good_mask = pred_masks[pred_idx]
        # if dice_loss_yolo+focal_loss_yolo > dice_loss+focal_loss: # if yolo mask is better (usually it is better than SAM for very faint objects)
        #     print(gt_masks.shape, pred_masks.shape, yolo_pred_masks.shape)
        #     print("Yolo mask is better")
        #     #plot the masks as comparison
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # ax[0].imshow(gt_masks[gt_idx][0].detach().cpu().numpy())
        # ax[0].set_title('Ground Truth Mask')
        # ax[0].axis('off')
        # ax[1].imshow(pred_masks[pred_idx][0].detach().cpu().numpy())
        # ax[1].set_title('SAM Mask')
        # ax[1].axis('off')
        # ax[2].imshow(yolo_pred_masks[pred_idx][0].detach().cpu().numpy())
        # ax[2].set_title('Yolo Mask')
        # ax[2].axis('off')
        # plt.tight_layout()
        # plt.show()
        # plt.close()
            
        #     dice_loss = dice_loss_yolo
        #     focal_loss = focal_loss_yolo
        #     good_mask = yolo_pred_masks[pred_idx]

        preds.append(good_mask)
        gts.append(gt_masks[gt_idx])
        pred_classes.append(int(all_pred_classes[pred_idx]))
        gt_classes.append(all_gt_classes[gt_idx])
        iou_scores_sam.append(iou_scores[pred_idx])
        # median_intensity = compute_median_intensity(gt_masks[gt_idx], image) # weight the loss by object intensity
            
        if mask_areas is not None:
            total_dice_loss += (dice_loss * mask_areas[gt_idx]/sum(mask_areas)) # weighted loss given mask size
            total_focal_loss += (focal_loss * mask_areas[gt_idx]/sum(mask_areas)) # weighted loss given mask size

    # Normalize the losses
    mean_dice_loss = total_dice_loss / len(row_ind)
    mean_focal_loss = total_focal_loss / len(row_ind)

    # Combine losses
    total_loss = mean_dice_loss + 20 * mean_focal_loss

    return total_loss, preds, gts, gt_classes, pred_classes, iou_scores_sam 

def segm_loss_match_iou_based(
	pred_masks, 
	gt_masks, 
	all_pred_classes, 
	all_gt_classes, 
	model_iou_scores,
    mask_areas=None):
    
    # Compute IoU matrix for all pairs
    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)  
    preds = []
    gts = []
    # Compute loss for matched pairs
    total_dice_loss = 0
    total_focal_loss = 0
    gt_classes, pred_classes, iou_scores_sam = [], [], []
    for pred_idx in range(iou_matrix.shape[0]):
        # Find the ground truth mask with the highest IoU for each predicted mask
        iou_scores = iou_matrix[pred_idx]
        gt_idx = torch.argmax(iou_scores).item()     
        preds.append(pred_masks[pred_idx])
        gts.append(gt_masks[gt_idx])
        pred_classes.append(int(all_pred_classes[pred_idx]))
        gt_classes.append(all_gt_classes[gt_idx])
        iou_scores_sam.append(model_iou_scores[pred_idx])
        # median_intensity = compute_median_intensity(gt_masks[gt_idx], image) # weight the loss by object intensity
        dice_loss = compute_dice_loss(pred_masks[pred_idx], gt_masks[gt_idx])
        focal_loss = compute_focal_loss(pred_masks[pred_idx].float(), gt_masks[gt_idx].float())
        if mask_areas is not None:
            total_dice_loss += (dice_loss * mask_areas[gt_idx]/sum(mask_areas)) # weighted loss given mask size
            total_focal_loss += (focal_loss * mask_areas[gt_idx]/sum(mask_areas)) # weighted loss given mask size

    # Normalize the losses
    mean_dice_loss = total_dice_loss / len(gts)
    mean_focal_loss = total_focal_loss / len(gts)

    # Combine losses
    total_loss = mean_dice_loss + 20 * mean_focal_loss

    return total_loss, preds, gts, gt_classes, pred_classes, iou_scores_sam 

def dice_loss_per_mask_pair(pred, target, mask_areas, negative_mask=None):
    
    assert pred.size() == target.size(), "Prediction and target must have the same shape"
    batch_size, height, width = pred.size()
    total_masks_area = np.array(mask_areas).sum()
    dice_loss = 0.0
    
    for i in range(batch_size):
        pred_mask = pred[i].contiguous()
        target_mask = target[i].contiguous()
        # if negative_mask is not None:
        #     neg_mask = negative_mask[i].bool()
        #     pred_mask = pred_mask * neg_mask
        #     target_mask = target_mask * neg_mask
        
        dice_loss += (compute_dice_loss(pred_mask, target_mask) * mask_areas[i]/total_masks_area) # weighted loss given mask size
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
        # # Assuming masks are 2D tensors, convert them to numpy for plotting
        # pred_mask_np = pred_mask.detach().cpu().numpy() if pred_mask.is_cuda else pred_mask.numpy()
        # target_mask_np = target_mask.detach().cpu().numpy() if target_mask.is_cuda else target_mask.numpy()
        
        # ax[0].imshow(pred_mask_np)
        # ax[0].set_title(f'Predicted Mask\n dice_loss: {((1 - mask_loss) * mask_areas[i] / total_masks_area):.4f}')
        # ax[0].axis('off')
        
        # ax[1].imshow(target_mask_np)
        # ax[1].set_title('Target Mask')
        # ax[1].axis('off')
        
        # plt.tight_layout()
        # plt.show()
        
    return dice_loss/batch_size

def focal_loss_per_mask_pair(inputs, targets, mask_areas):
    
    assert inputs.size() == targets.size(), "Inputs and targets must have the same shape"
    batch_size, height, width = inputs.size()
    total_masks_area = np.array(mask_areas).sum()
    focal_loss = 0.0
    
    for i in range(batch_size):
        input_mask = inputs[i].unsqueeze(0)
        target_mask = targets[i].unsqueeze(0)
        focal_loss += (compute_focal_loss(input_mask, target_mask) * mask_areas[i] / total_masks_area) # weighted loss given mask size
        
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
        # # Assuming masks are 2D tensors, convert them to numpy for plotting
        # pred_masks_np = inputs.detach().cpu().numpy() if inputs.is_cuda else inputs.numpy()
        # target_mask_np = target_mask.detach().cpu().numpy() if target_mask.is_cuda else target_mask.numpy()
        
        # ax[0].imshow(pred_masks_np[i])
        # ax[0].set_title(f'Predicted Mask\n focal_loss: {(mask_focal_loss * mask_areas[i] / total_masks_area):.4f}')
        # ax[0].axis('off')
        
        # ax[1].imshow(target_mask_np[0])
        # ax[1].set_title('Target Mask')
        # ax[1].axis('off')
        
        # plt.tight_layout()
        # plt.show()
        
    focal_loss /= batch_size
    
    return focal_loss
