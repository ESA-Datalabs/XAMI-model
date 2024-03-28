import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

def iou_single(pred_mask, gt_mask):
    """Compute IoU between a single predicted mask and a single ground truth mask."""
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask) + torch.sum(gt_mask) - intersection
    if union == 0:
        return torch.tensor(0.)
    else:
        return intersection / union

def compute_average_intensity(mask, image):
    """
    Compute the average intensity for the object represented by the mask.
    
    Parameters:
    - mask: Binary mask of the object (Tensor of shape [H, W])
    - image: Image tensor corresponding to the mask (Tensor of shape [C, H, W] or [H, W])
    
    Returns:
    - average_intensity: Average intensity of the object
    """
    if image.dim() == 3:  # If the image has channels, reduce it to grayscale or use a specific channel
        image = image.mean(dim=0)  # Simple way to convert to grayscale, may be adjusted
    object_pixels = image[mask.bool()]
    average_intensity = object_pixels.mean()
    return average_intensity
	
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

# def compute_focal_loss(input, target, alpha=0.7, gamma=2.0):
#     """Computes the focal loss between `input` and `target`."""
#     input = torch.sigmoid(input)
#     ce_loss = F.binary_cross_entropy(input, target, reduction='none')
#     p_t = target * input + (1 - target) * (1 - input)
#     focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
#     return focal_loss.mean()

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
    # Flatten the masks to ensure compatibility with different shapes
    pred_flat = torch.flatten(pred_mask)
    gt_flat = torch.flatten(gt_mask) 

    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum()
    
    # Compute the Dice coefficient and loss
    dice_coefficient = (2. * intersection + 1e-6) / (union + 1e-6)  # Adding a small epsilon to avoid division by zero
    dice_loss = 1 - dice_coefficient
    
    return dice_loss

def segm_loss_match_hungarian(
	pred_masks, 
	low_res_masks, 
	gt_masks, 
	all_pred_classes, 
	all_gt_classes, 
	iou_scores):

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
	    pred_classes.append(all_pred_classes[pred_idx])
	    gt_classes.append(all_gt_classes[gt_idx])
	    iou_scores_sam.append(iou_scores[pred_idx])
	    # average_intensity = compute_average_intensity(gt_masks[gt_idx], image) # weight the loss by object intensity
	    total_dice_loss += compute_dice_loss(pred_masks[pred_idx], gt_masks[gt_idx])
	    total_focal_loss += compute_focal_loss(low_res_masks[pred_idx].float(), gt_masks[gt_idx].float())
    
    # Normalize the losses
	mean_dice_loss = total_dice_loss / len(row_ind)
	mean_focal_loss = total_focal_loss / len(row_ind)

    # Combine losses
	total_loss = mean_dice_loss + 20 * mean_focal_loss
    
	return total_loss, preds, gts, gt_classes, pred_classes, iou_scores_sam #* average_intensity

def segm_loss_match_iou_based(pred_masks, low_res_masks, gt_masks):
    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)  # Compute IoU matrix for all pairs
    
    # Initialize total loss
    total_loss = 0
    
    # Track used ground truth masks to avoid multiple matchings
    matched_gt_masks = set()
    
    for pred_idx in range(iou_matrix.shape[0]):
        # Find the ground truth mask with the highest IoU for each predicted mask
        iou_scores = iou_matrix[pred_idx]
        gt_idx = torch.argmax(iou_scores).item()
        
        # Ensure each ground truth mask is only matched once
        if gt_idx not in matched_gt_masks:
            matched_gt_masks.add(gt_idx)
            # average_intensity = compute_average_intensity(gt_masks[gt_idx], images[gt_idx])
            
            # Compute Dice and focal losses for matched pairs
            dice_loss = compute_dice_loss(pred_masks[pred_idx], gt_masks[gt_idx])
            focal_loss = compute_focal_loss(low_res_masks[pred_idx].float(), gt_masks[gt_idx].float())
            
            # Weight the loss for the current pair by the average intensity
            weighted_loss = (dice_loss + 20 * focal_loss) #* average_intensity
            
            # Accumulate the total loss
            total_loss += weighted_loss
        else:
            continue  # Skip if this ground truth mask is already matched
    
    # Normalize the total loss by the number of matched pairs
    if len(matched_gt_masks) > 0:
        normalized_total_loss = total_loss / len(matched_gt_masks)
    else:
        normalized_total_loss = total_loss
    
    return normalized_total_loss


def dice_loss_per_mask_pair(pred, target, mask_areas, negative_mask=None, smooth=1):
    
    assert pred.size() == target.size(), "Prediction and target must have the same shape"
    batch_size, height, width = pred.size()
    total_masks_area = target.sum()
    dice_loss = 0.0
    
    for i in range(batch_size):
        pred_mask = pred[i].contiguous()
        target_mask = target[i].contiguous()
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
        # # Assuming masks are 2D tensors, convert them to numpy for plotting
        # pred_mask_np = pred_mask.detach().cpu().numpy() if pred_mask.is_cuda else pred_mask.numpy()
        # target_mask_np = target_mask.detach().cpu().numpy() if target_mask.is_cuda else target_mask.numpy()
        
        # ax[0].imshow(pred_mask_np, cmap='gray')
        # ax[0].set_title('Predicted Mask')
        # ax[0].axis('off')
        
        # ax[1].imshow(target_mask_np, cmap='gray')
        # ax[1].set_title('Target Mask')
        # ax[1].axis('off')
        
        # plt.tight_layout()
        # plt.show()
        

        # if negAative_mask is not None:
        #     neg_mask = negative_mask[i].bool()
        #     pred_mask = pred_mask * neg_mask
        #     target_mask = target_mask * neg_mask
        
        intersection = (pred_mask * target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()
        mask_loss = (2. * intersection + smooth) / (total + smooth)
        # dice_loss += ((1 - mask_loss) * mask_areas[i] / total_masks_area) # weighted loss given mask size
        dice_loss += (1 - mask_loss)
    
    dice_loss /= batch_size
    
    return dice_loss

def focal_loss_per_mask_pair(inputs, targets, mask_areas, alpha=0.8, gamma=2):
    
    assert inputs.size() == targets.size(), "Inputs and targets must have the same shape"
    batch_size, height, width = inputs.size()
    total_masks_area = targets.sum()
    focal_loss = 0.0
    
    for i in range(batch_size):
        input_mask = inputs[i].unsqueeze(0)
        target_mask = targets[i].unsqueeze(0)
        mask_focal_loss = compute_focal_loss(input_mask, target_mask)
        focal_loss += mask_focal_loss
		
        # BCE = F.binary_cross_entropy_with_logits(input_mask, target_mask, reduction='mean')
        # BCE_EXP = torch.exp(-BCE)
        # mask_focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE # * mask_areas[i]/total_masks_area # weighted loss given mask size
        # focal_loss += mask_focal_loss
    
    focal_loss /= batch_size
    
    return focal_loss
