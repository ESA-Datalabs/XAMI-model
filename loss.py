import torch
from scipy.optimize import linear_sum_assignment

def iou_single(pred_mask, gt_mask):
    """Compute IoU between a single predicted mask and a single ground truth mask."""
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask) + torch.sum(gt_mask) - intersection
    if union == 0:
        return torch.tensor(0.)
    else:
        return intersection / union

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

def compute_dice_loss(pred_mask, gt_mask):
    """
    Compute the Dice loss between a single predicted mask and a single ground truth mask.
    Both masks should be floating-point tensors with the same shape.
    """
    # Flatten the masks to ensure compatibility with different shapes
    pred_flat = pred_mask.view(-1).float()
    gt_flat = gt_mask.view(-1).float()
    
    # Compute intersection and union
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum()
    
    # Compute the Dice coefficient and loss
    dice_coefficient = (2. * intersection + 1e-6) / (union + 1e-6)  # Adding a small epsilon to avoid division by zero
    dice_loss = 1 - dice_coefficient
    
    return dice_loss

def segm_loss_match(pred_masks, gt_masks):

	# Compute IoU matrix for all pairs
	iou_matrix = compute_iou_matrix(pred_masks, gt_masks)  
	cost_matrix = -iou_matrix  # Negate IoU for minimization
	row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().numpy())
	
	# Compute loss for matched pairs
	total_loss = 0
	for pred_idx, gt_idx in zip(row_ind, col_ind):
	    total_loss += compute_dice_loss(pred_masks[pred_idx], gt_masks[gt_idx])
	
	# TODO: Add any additional penalties for unmatched masks if necessary
	
	return total_loss/len(row_ind)