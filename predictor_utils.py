import traceback
import numpy as np
import cv2
from PIL import Image
import supervision as sv
from astropy.io import fits
import matplotlib.pyplot as plt
import torch
from astronomy_utils import data_norm
from torchvision.transforms.functional import resize


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks(masks, ax, random_color=False):
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='orange', facecolor=(0,0,0,0), lw=2))    

def mask_and_plot_image(fits_file, plot_ = False):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
    mask = data <= 0    
    if plot_ and len(data[data>0])>=1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(data, cmap='gray', norm = data_norm(data[data>0]))
        ax1.set_title('Image')
        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Mask')
        plt.show()
        plt.close()        
    return mask

def dice_loss_numpy(pred, target, area=None, smooth = 1): 
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice
    
    return dice_loss
        
def compute_loss(gt_masks, pred_masks):
    if gt_masks.size == 0:
        return 0 
    if pred_masks.size == 0:
        return 1
    losses = []
    for gt_mask in gt_masks:
        mask_losses = []
        max_iou = 0.0
        mask_loss = 0.5
        for pred_mask in pred_masks:
            iou = np.sum(pred_mask.flatten() * gt_mask.flatten())
            if iou > max_iou:
                mask_loss = dice_loss_numpy(pred_mask, gt_mask)
        losses.append(mask_loss)
    return np.mean(losses) 

def remove_masks(sam_result, mask_on_negative, threshold, remove_big_masks=False, big_masks_threshold=None, img_shape=None):
    '''
    Given a segmentation result, this function removes the masks 
    if the intersection with the negative pixels gives a number is > than a threshold
    '''
    if img_shape is not None:
        big_masks_threshold = img_shape[0]**2/5 if big_masks_threshold is None else big_masks_threshold
    else:
        big_masks_threshold = None

    bad_indices = np.array([],  dtype=int) 
    for segm_index in range(len(sam_result)):
        count = np.sum((sam_result[segm_index]['segmentation'] == 1) & (mask_on_negative == 0))  

        # remove masks on negative pixels given threshold
        if count > threshold:
            # print(f"Removing mask {segm_index} on negative with {count} counts.")
            bad_indices = np.append(bad_indices, segm_index)
        
        # remove very big (>70) masks
        if remove_big_masks and img_shape is not None and np.sum(sam_result[segm_index]['segmentation']) > big_masks_threshold:
            # print(f"Removing mask {segm_index} with area {np.sum(sam_result[segm_index]['segmentation'])}")
            bad_indices = np.append(bad_indices, segm_index)   
    sam_result = np.delete(sam_result, bad_indices)
    return sam_result

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

def dice_loss_numpyy(pred_mask, gt_mask):
    smooth = 1.0 
    intersection = torch.sum(pred_mask * gt_mask)
    return 1 - ((2. * intersection + smooth) / (torch.sum(pred_mask) + torch.sum(gt_mask) + smooth))

def instance_segmentation_loss(pred_masks, gt_masks):

    pred_masks = [torch.tensor(mask, dtype=torch.float32) for mask in pred_masks]
    gt_masks = [torch.tensor(mask, dtype=torch.float32) for mask in gt_masks]

    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            intersection = torch.logical_and(pred_mask, gt_mask).float().sum()
            union = torch.logical_or(pred_mask, gt_mask).float().sum()
            iou_matrix[i, j] = (intersection / union).item() if union != 0 else 0

    # Match predicted masks to ground truth masks using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximizing IoU

    # Compute the Dice loss for each matched pair
    loss = 0.0
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(gt_masks[gt_idx].float())
        # ax[1].imshow(pred_masks[pred_idx].float())
        # plt.show()
        # plt.close()
        loss += dice_loss_numpyy(pred_masks[pred_idx].float(), gt_masks[gt_idx].float())
        # print('dice loss: ', loss)

    # Normalize the loss
    # matched_pairs = len(row_ind)
    # if matched_pairs > 0:
    #     loss /= matched_pairs
    
    # print('normalised loss: ', loss)

    # Penalize unmatched predicted masks (False Positives)
    unmatched_pred = set(range(len(pred_masks))) - set(row_ind)
     # Penalize unmatched predicted masks (False Positives)
    fp_loss = sum(dice_loss_numpyy(pred_masks[i], torch.zeros_like(pred_masks[i])) for i in unmatched_pred)

    # Penalize unmatched ground truth masks (False Negatives)
    unmatched_gt = set(range(len(gt_masks))) - set(col_ind)
    # Penalize unmatched ground truth masks (False Negatives)
    fn_loss = sum(dice_loss_numpyy(torch.zeros_like(gt_masks[j]), gt_masks[j]) for j in unmatched_gt)

    # Total loss
    total_loss = loss + fp_loss + fn_loss

    # Normalize the total loss by the total number of masks (matched and unmatched)
    total_masks = len(pred_masks) + len(gt_masks)
    normalized_loss = total_loss / total_masks if total_masks > 0 else total_loss

    return normalized_loss


def amg_predict(any_sam_model, AMG, data_set_gt_masks, model_name,  IMAGE_PATH, use_negative=None, mask_on_negative=False, show_plot=False):
    
    image_name = IMAGE_PATH.split("/")[-1]
    predicted_masks = []
    gt_image_masks = np.array([mask for key, mask in data_set_gt_masks.items() if key.startswith(image_name)])
 
    with torch.no_grad():
        image_bgr = cv2.imread(IMAGE_PATH)
        annotated_image = None

        # here also set the negative masked pixels to 0 after pre-processing
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # image_rgb = 255 - image_rgb
        negative_mask = None
        if mask_on_negative:
            negative_mask = np.where(image_rgb>0, True, False)
            negative_mask = torch.from_numpy(negative_mask)  
            negative_mask = negative_mask.permute(2, 0, 1)
            negative_mask = resize(negative_mask, [1024, 1024], antialias=True) 
            negative_mask = negative_mask.unsqueeze(0)
        try:
            mask_generator = AMG(any_sam_model, negative_mask=negative_mask,
                points_per_side=40,  # Increased number of points
                points_per_batch=64,  
                pred_iou_thresh=0.85,  # Slightly lower threshold
                stability_score_thresh=0.90,  # Slightly lower threshold
                stability_score_offset=1.0,  
                box_nms_thresh=0.6,  # Lower threshold for stricter NMS
                crop_n_layers=2,   # More crop layers
                crop_nms_thresh=0.7,  
                crop_overlap_ratio=512 / 1500,  
                crop_n_points_downscale_factor=2,  # Higher downscale factor
                point_grids=None,  
                min_mask_region_area=10,  # Small value to eliminate tiny regions
                output_mode="binary_mask"  
                )
        except:
            print("❗️Did not use negative mask in AMG.❗️")
            mask_generator = AMG(any_sam_model,
                                points_per_side=40,  # Increased number of points
                                points_per_batch=64,  
                                pred_iou_thresh=0.85,  # Slightly lower threshold
                                stability_score_thresh=0.90,  # Slightly lower threshold
                                stability_score_offset=1.0,  
                                box_nms_thresh=0.6,  # Lower threshold for stricter NMS
                                crop_n_layers=2,   # More crop layers
                                crop_nms_thresh=0.7,  
                                crop_overlap_ratio=512 / 1500,  
                                crop_n_points_downscale_factor=2,  # Higher downscale factor
                                point_grids=None,  
                                min_mask_region_area=10,  # Small value to eliminate tiny regions
                                output_mode="binary_mask"  
                                )
            pass

        sam_result = mask_generator.generate(image_rgb)
        output_file = './plots/'+image_name+'_'+model_name+'.png'

        if mask_on_negative:
            img_negative_mask = np.where(image_rgb>0, 1, 0) 
            img_negative_mask = np.min(img_negative_mask, axis=2) # to make it 2D
            # plt.imshow(img_negative_mask, cmap='gray')
            # plt.show()
            # plt.close()
            sam_result = remove_masks(sam_result=sam_result, mask_on_negative=img_negative_mask, \
                                      threshold=300, remove_big_masks=False, img_shape=image_rgb.shape)
            output_file = './plots/'+image_name+'_'+model_name+'_segmented_removed_negative.png'

        # !!! takes the predicted masks, and removes the ones that are covering more than 50% of the image
        predicted_masks = np.array([out_pred['segmentation'] for out_pred in sam_result if np.sum(out_pred['segmentation']) <image_rgb.shape[0]**2/2]) 
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result) # type: ignore
        if detections is not None and detections.mask is not None:
            # !!! takes the detection masks and removes the ones that are covering more than 50% of the image (for detections)
            detections.mask = np.array([detmask for detmask in detections.mask if np.sum(detmask) <image_rgb.shape[0]**2/2])

        annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        image = Image.fromarray(annotated_image)
        # image.save(output_file)

        iou_assoc_loss = instance_segmentation_loss(predicted_masks, gt_image_masks)
        
        if show_plot:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            
            # Ground truth mask
            axs[0].imshow(image_bgr, cmap='viridis')
            for mask_ in gt_image_masks:
                show_mask(mask_, axs[0])
            axs[0].set_title('Ground Truth Masks\n'+image_name.split(".")[0])
            
            # Predicted mask
            axs[1].imshow(annotated_image, cmap='viridis')
            axs[1].set_title('Predicted Masks')

            plt.show()
            plt.close()
    return annotated_image, iou_assoc_loss


def SAM_predictor(AMG, sam, IMAGE_PATH, mask_on_negative = None, img_grid_points=None):
    """
    This function infers the SAM (Segment Anything) and returns the annnotated image. 
    
    Args:
        IMAGE_PATH (str): The path to the image file.
        remove_masks_on_negative (bool, optional): If True, masks on negative detections are removed.

    Returns:
        tuple: A tuple containing the original image and the annotated image.
    """
    
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    annotated_image = None
    try:

        mask_generator = AMG(sam, points_per_side=None, point_grids=img_grid_points) if img_grid_points is not None else AMG(sam)

        sam_result = mask_generator.generate(image_rgb)
        # output_file = IMAGE_PATH.replace("scaled_raw", "segmented_SAM").replace(".png", "_segmented.png")
        
        if mask_on_negative is not None:
            sam_result = remove_masks(sam_result=sam_result, mask_on_negative=mask_on_negative.astype(int), 
                                      threshold=100, remove_big_masks=True, img_shape=image_rgb.shape)
            # output_file = IMAGE_PATH.replace("scaled_raw", "segmented_SAM").replace(".png", "_segmented_removed_negative.png")

        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        # image = Image.fromarray(annotated_image)
        # image.save(output_file)
    
        fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 

        axs[0].imshow(image_bgr)
        axs[0].set_title(f'source image {IMAGE_PATH.split("/")[-1]}')

        axs[1].imshow(annotated_image)
        axs[1].set_title(f'original SAM segmented image')

        if img_grid_points is not None:
            axs[2].imshow(cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB))
            axs[2].scatter(img_grid_points[0][:, 0]*255, img_grid_points[0][:, 1]*255, s=10, c='r', marker='o')
            axs[2].set_title('Grid points as prompts')
        else:
            fig.delaxes(axs[2])

        plt.tight_layout()
        plt.show()
        plt.close() 
    except Exception as e:
        print("Exception:\n", e)
        traceback.print_exc()
        return None, None, None
            
    return sam_result, detections, annotated_image