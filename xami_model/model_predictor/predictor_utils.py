import numpy as np
import cv2
from PIL import Image
import supervision as sv
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import resize
from typing import List, Dict, Any, Optional, Tuple
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import re
from ..dataset import dataset_utils
from ..losses import loss_utils

def transform_image(model, transform, image, k, device):
    
    image_tensor = torch.from_numpy(image).to(device).float()  
    mask_nonzero = image_tensor > 0
    image_nonzero = image_tensor[mask_nonzero]
    mean_ = image_nonzero.mean()
    std_ = image_nonzero.std()
    model.register_buffer("pixel_mean", mean_.repeat((3, 1, 1)), persistent=False)
    model.register_buffer("pixel_std", std_.repeat((3, 1, 1)), persistent=False)
    negative_mask = (image_tensor > 0).to(torch.float32)
    negative_mask = negative_mask.permute(2, 0, 1)
    negative_mask = resize(negative_mask, [1024, 1024], antialias=True).unsqueeze(0)

    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device).permute(2, 0, 1).unsqueeze(0)
    
    input_image = model.preprocess(input_image_torch)
    original_image_size = torch.tensor(image.shape[:2], device=device)
    input_size = input_image_torch.shape[-2:]
    input_image[~negative_mask.bool()] = 0

    transformed_data = {
        'image': input_image,
        'input_size': input_size,
        'image_id': k,
        'original_image_size': original_image_size
    }
    
    return transformed_data

def set_mean_and_transform(image, model, transform, device):
    
    input_image = transform_image(model, transform, image, 'dummy_image_id', device)['image']
    input_image = torch.as_tensor(input_image, dtype=torch.float, device=device) # (B, C, 1024, 1024)
    
    return input_image

def check_requires_grad(model, show=True):
    for name, param in model.named_parameters():
        if param.requires_grad and show:
            print("✅ Param", name, " requires grad.")
        elif param.requires_grad == False:
            print("❌ Param", name, " doesn't require grad.")

def prints_and_wandb(epoch, epoch_sam_loss_train, epoch_sam_loss_val, all_metrics, metric_thresholds, wandb=None):  
    
    for metric in metric_thresholds:
        train_map = all_metrics[tuple(metric)]['train']['map']*100
        valid_map = all_metrics[tuple(metric)]['valid']['map']*100
        print(f"Train mAP{metric}: {train_map}. Valid mAP{metric}: {valid_map}")
        
        if wandb is not None:
            wandb.log({'train/mAP'+str(metric): train_map/100, 'valid/mAP'+str(metric): valid_map/100})
    
    print(f"Epoch {epoch}. Train loss: {np.round(epoch_sam_loss_train, 7)}")
    print(f"Epoch {epoch}. Validation loss: {np.round(epoch_sam_loss_val, 7)}")

    # Wandb
    if wandb is not None:
        wandb.log({'train_SAM_loss': np.round(epoch_sam_loss_train, 7), 'valid_SAM_loss': np.round(epoch_sam_loss_val, 7)})
        
def print_training_intro(train_dir_list, valid_dir_list, device, num_epochs, batch_size, lr, wd, wandb_track, model, optimizer_name):

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    color_start = "\033[1;36m"  
    color_end = "\033[0m" 

    print(f"{color_start}Model Training Configuration:{color_end}")
    print(f" - Training images: {len(train_dir_list)}")
    print(f" - Validation images: {len(valid_dir_list)}")
    print(f" - Number of Epochs: {num_epochs}")
    print(f" - Batch Size: {batch_size}")
    print(f" - Learning Rate: {lr}")
    print(f" - Weight Decay: {wd}")
    print(f" - Device: {device}")
    print(f" - Early Stopping: Stop if no improvement after {num_epochs // 10 + 5} epochs.")
    print(f" - Weights & Biases Tracking: {'Enabled' if wandb_track else 'Disabled'}.")
    print(f" - Optimizer: {optimizer_name}.")
    print(f" - Total Trainable Parameters: {total_params:,}")

def create_gaussian_kernel(kernel_size=5, sigma=2, in_channels=1, out_channels=1):
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

def dice_loss_numpy(pred_mask, gt_mask):
    smooth = 1.0 
    intersection = torch.sum(pred_mask * gt_mask)
    return 1 - ((2. * intersection + smooth) / (torch.sum(pred_mask) + torch.sum(gt_mask) + smooth))

def remove_masks(
    sam_result: List[Dict[str, Any]], 
    mask_on_negative: np.ndarray, 
    threshold: int, 
    remove_big_masks: bool = False, 
    big_masks_threshold: Optional[int] = None, 
    img_shape: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
    """
    Removes masks from the segmentation result based on the intersection with negative pixels and size constraints.
    
    Parameters:
    - sam_result (List[Dict[str, Any]]): The segmentation result to process, where each element represents a mask.
    - mask_on_negative (np.ndarray): A mask indicating negative areas where masks should potentially be removed.
    - threshold (int): The threshold for the number of intersecting negative pixels to consider a mask for removal.
    - remove_big_masks (bool, optional): If True, masks exceeding a certain size are removed. Defaults to False.
    - big_masks_threshold (Optional[int], optional): The area threshold above which big masks are removed. 
    If None and `remove_big_masks` is True, it's calculated based on `img_shape`. Defaults to None.
    - img_shape (Optional[tuple], optional): The shape of the input image, used to calculate `big_masks_threshold` if the former is provided. Defaults to None.
    
    Returns:
    - List[Dict[str, Any]]: The modified segmentation result with specified masks removed.
    """
    if img_shape is not None:
        big_masks_threshold = img_shape[0]**2/4 if big_masks_threshold is None else big_masks_threshold
    else:
        big_masks_threshold = None

    bad_indices = np.array([],  dtype=int) 
    for segm_index in range(len(sam_result)):
        sam_result[segm_index]['segmentation'] = sam_result[segm_index]['segmentation'] * (~mask_on_negative)
		
        count = np.sum((sam_result[segm_index]['segmentation'] == 1) & (mask_on_negative == 0))  

        if count > threshold:
            bad_indices = np.append(bad_indices, segm_index)
        
        # # remove very big masks
        # if remove_big_masks and img_shape is not None and np.sum(sam_result[segm_index]['segmentation']) > big_masks_threshold:
        #     bad_indices = np.append(bad_indices, segm_index)   
    sam_result = np.delete(sam_result, bad_indices) # type: ignore
    return sam_result

def show_predicted_masks(masks, ax, random_color=False, colours=None):
    for i in range(len(masks)):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        elif colours is not None:
            color = np.array([c/255.0 for c in colours[i]]+[0.6])
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = masks[i].shape[-2:]
        mask_image = masks[i].reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
def compute_iou(mask1, mask2):
    """Compute Intersection over Union of two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou_score

def compute_metrics(gt_masks, pred_masks, iou_threshold, image=None):
    """Compute the True Positive, False Positive, and False Negative BINARY masks for multiple segmentations."""
    
    # if image is not None:
    #     plt.imshow(image)
    #     for mask in gt_masks:
    #         dataset_utils.show_mask(mask[0], plt.gca())
    #     plt.show()
    #     plt.close()
    
    #     plt.imshow(image)
    #     for mask in pred_masks:
    #         dataset_utils.show_mask(mask, plt.gca())
    #     plt.show()
    #     plt.close()
        
    combined_gt_mask = np.zeros_like(gt_masks[0][0], dtype=bool)
    combined_pred_mask = np.zeros_like(pred_masks[0], dtype=bool)
    filtered_pred_masks = np.zeros_like(pred_masks, dtype=bool)
    
    # print(gt_masks.shape, pred_masks.shape) # (N, 1, H, W), (N, H, W)
    for i, pred_mask in enumerate(pred_masks):
        max_iou = 0  # Max IoU for this pred_mask with any gt_mask
        for gt_mask in gt_masks:
            intersection = np.logical_and(gt_mask[0], pred_mask)
            union = np.logical_or(gt_mask[0], pred_mask)
            iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            max_iou = max(max_iou, iou_score)  

        if max_iou > iou_threshold: # take IoUs above threshold
            filtered_pred_masks[i] = pred_mask

    for gt_mask in gt_masks:
        combined_gt_mask = np.logical_or(combined_gt_mask, gt_mask)

    for pred_mask in filtered_pred_masks:
        combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask.astype(bool))

    true_positive_mask = np.logical_and(combined_gt_mask, combined_pred_mask)
    false_negative_mask = np.logical_and(combined_gt_mask, np.logical_not(combined_pred_mask))
    false_positive_mask = np.logical_and(combined_pred_mask, np.logical_not(combined_gt_mask))

    return true_positive_mask, false_positive_mask, false_negative_mask

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
        loss += dice_loss_numpy(pred_masks[pred_idx].float(), gt_masks[gt_idx].float())
        # print('dice loss: ', loss)

    # Normalize the loss
    # matched_pairs = len(row_ind)
    # if matched_pairs > 0:
    #     loss /= matched_pairs
    
    # print('normalised loss: ', loss)

    # Penalize unmatched predicted masks (False Positives)
    unmatched_pred = set(range(len(pred_masks))) - set(row_ind)
     # Penalize unmatched predicted masks (False Positives)
    fp_loss = sum(dice_loss_numpy(pred_masks[i], torch.zeros_like(pred_masks[i])) for i in unmatched_pred)

    # Penalize unmatched ground truth masks (False Negatives)
    unmatched_gt = set(range(len(gt_masks))) - set(col_ind)
    # Penalize unmatched ground truth masks (False Negatives)
    fn_loss = sum(dice_loss_numpy(torch.zeros_like(gt_masks[j]), gt_masks[j]) for j in unmatched_gt)

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
                dataset_utils.show_mask(mask_, axs[0])
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
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # (H, W, C)

    annotated_image = None
    mask_generator = AMG(sam, points_per_side=None, point_grids=img_grid_points) if img_grid_points is not None else AMG(sam)
    sam_result = mask_generator.generate(image_rgb)
    if mask_on_negative is not None:
        sam_result = remove_masks(sam_result=sam_result, 
                                            mask_on_negative=mask_on_negative, 
                                            threshold=image_rgb.shape[0]**2/10,
                                            remove_big_masks=True, 
                                            img_shape = image_rgb.shape)

    color = sv.Color(0, 255, 0)
    mask_annotator = sv.MaskAnnotator(color=color, color_lookup=None)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    image_rgb = 255 - image_rgb
    annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    # annotated_image = annotated_image * (image_rgb<255).astype(float) # mask negative pixels
    if annotated_image.max() <= 1.0:
        annotated_image *= 255
    
    annotated_image = annotated_image.astype(np.uint8)

    return sam_result, detections, annotated_image

def MobileSAM_predict(image_path: str, 
					  model: nn.Module, 
					  predictor: Any,
					  generator: Any,
					  device: str,
                      output_path: Optional[str] = None,
                      mask_on_negative: Optional[np.ndarray] = None,
					  use_specific_distr: Optional[bool] = False
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts segmentation masks for an input image using the MobileSAM model and optionally annotates the image with the segmentation results.
    
    Parameters:
    - image_path (str): Path to the input image.
    - mask_on_negative (Optional[np.ndarray], optional): A negative map specifying the bad (-ve) pixel regions used for removing unuseful segmentations masks.
    If provided, masks identified as negative based on this parameter are removed from the annotation. Defaults to None.
    - output_path (str): The output path for the annotated image. Defaults to None. 
    
    Returns:
    - Tuple[np.ndarray, np.ndarray] : The annotated image array and the resulted masks.
    
    Notes:
    - The function uses OpenCV to read and preprocess the input image.
    - It utilizes a pretrained MobileSAM model for prediction.
    - The prediction process involves normalizing the image based on its pixel mean and standard deviation.
    - The `mask_on_negative` parameter allows for further processing to remove certain masks based on the provided negative mask.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, C)

    with torch.no_grad():
        # using the pixel mean and std specific to each image instead of the standard one (this step can be ignored)
        image_T = np.transpose(image, (2, 1, 0))
        pixel_mean = torch.as_tensor([np.mean(image_T[0]), np.mean(image_T[1]),np.mean(image_T[2])], dtype=torch.float, device=device)
        pixel_std = torch.as_tensor([np.std(image_T[0]), np.std(image_T[1]),np.std(image_T[2])], dtype=torch.float, device=device)
        model.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        model.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        predictor = predictor(model)
        predictor.set_image(image)
        
        mask_generator = generator(model)
        model_result = mask_generator.generate(image)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        if mask_on_negative is not None:
            model_result = remove_masks(sam_result=model_result, 
                                             mask_on_negative=mask_on_negative, 
                                             threshold=image.shape[0]**2/4, 
                                             remove_big_masks=True, 
                                             img_shape = image.shape)

        detections = sv.Detections.from_sam(model_result)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

        annotated_image = annotated_image * (image>0).astype(float) # mask negative pixels
        
        if annotated_image.max() <= 1.0:
            annotated_image *= 255

        annotated_image = annotated_image.astype(np.uint8)
        image = Image.fromarray(annotated_image)
        
        if output_path:
            image.save(output_path)

    return image, model_result

def process_faint_masks(image, pred_masks, yolo_masks, predicted_classes, device, wt_threshold=0.6, wt_classes=[1.0, 4.0]):
    wt_mask, _ = dataset_utils.isolate_background(image[:, :, 0], decomposition='db1', level=2, sigma=1)
    wt_mask = torch.from_numpy(wt_mask).to(device)  # Ensure wavelet mask is on GPU

    for i, low_res_mask in enumerate(pred_masks):
        thresholded_mask = (low_res_mask > 0.5).to(torch.float32)
        sum_thresholded_mask = torch.clamp(thresholded_mask.sum(), min=1)  # Avoid division by zero
        wt_on_mask = wt_mask * thresholded_mask / sum_thresholded_mask
        wt_count = wt_on_mask.sum()
        if wt_count > wt_threshold and predicted_classes[i] in wt_classes:
            pred_masks[i] = torch.from_numpy(yolo_masks[i]).float().to(device).unsqueeze(0)
            
    return pred_masks

def calculate_iou_loss(threshold_masks, gt_threshold_masks, iou_predictions, mask_areas):
    """
    Calculate IoU per mask and the corresponding image loss.

    Args:
    - threshold_masks (torch.Tensor): The predicted threshold masks.
    - gt_threshold_masks (torch.Tensor): The ground truth threshold masks.
    - iou_predictions (torch.Tensor): The predicted IoU values.
    - mask_areas (torch.Tensor): The areas of each mask.

    Returns:
    - ious (list): A list of IoU values per mask.
    - iou_image_loss (list): A list of IoU image losses per mask.
    """
    ious = []
    iou_image_loss = []
    total_mask_areas = np.array(mask_areas).sum()
    for i in range(threshold_masks.shape[0]):
        iou_per_mask = loss_utils.iou_single(threshold_masks[i][0], gt_threshold_masks[i])
        ious.append(iou_per_mask)
        iou_image_loss.append((torch.abs(iou_predictions.permute(1, 0)[0][i] - iou_per_mask)) * mask_areas[i] / total_mask_areas)
    
    return ious, torch.stack(iou_image_loss)


def get_next_directory_name(base_dir):
    # Get the base directory name without index
    base_name = os.path.basename(base_dir)
    parent_dir = os.path.dirname(base_dir)
    
    # Get all directories in the parent directory
    all_dirs = os.listdir(parent_dir)
    
    # Filter directories that match the pattern base_name_index
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)")
    indices = [
        int(match.group(1)) for match in (pattern.match(d) for d in all_dirs) if match
    ]
    
    # If there are no matching directories, start with index 1
    if not indices:
        next_index = 1
    else:
        next_index = max(indices) + 1
    
    return os.path.join(parent_dir, f"{base_name}_{next_index}")

def convert_tensors(data):
    if isinstance(data, dict):
        return {key: convert_tensors(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.tolist() if data.ndim > 0 else data.item()
    else:
        return data
    
def compute_indiv_metrics(pred_mask, gt_mask, iou_threshold=0.5):
    """Compute precision, recall, F1 score, and accuracy for aligned predicted and ground truth masks."""
    iou = compute_iou(pred_mask, gt_mask)
    if iou >= iou_threshold:
        true_positive = np.logical_and(pred_mask, gt_mask).sum()
        false_positive = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
        false_negative = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positive / gt_mask.sum() if gt_mask.sum() > 0 else 0
    else:
        precision = 0
        recall = 0
        f1 = 0
        accuracy = 0

    metrics = {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }
    return metrics

def compute_scores(metric_name, all_pred_masks, all_gt_masks, thresholds):
    means = []
    stds = []
    for threshold in thresholds:
        all_metrics = []
        for i, (pred_mask, gt_mask) in enumerate(zip(all_pred_masks, all_gt_masks)):
            for mask_i in range(gt_mask.shape[0]):
                metrics = compute_indiv_metrics(pred_mask[mask_i].detach().cpu().numpy(), gt_mask[mask_i].detach().cpu().numpy(), iou_threshold=threshold)
                if metrics is not None:
                    all_metrics.append(metrics)
        values = [m[metric_name] for m in all_metrics]
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    return metric_name, np.array(means), np.array(stds)

def plot_conf_m(
    self,
    save_path=None,
    title=None,
    classes= None,
    normalize=False,
    fig_size=(12, 10)):

    array = self.matrix.copy()

    if normalize:
        eps = 1e-8
        array = array / (array.sum(0).reshape(1, -1) + eps)

    array[array < 0.005] = np.nan

    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True, facecolor="white")

    class_names = classes if classes is not None else self.classes
    use_labels_for_ticks = class_names is not None and (0 < len(class_names) < 99)
    if use_labels_for_ticks:
        x_tick_labels = class_names + ["Background"]
        y_tick_labels = class_names + ["Background"]
        num_ticks = len(x_tick_labels)
    else:
        x_tick_labels = None
        y_tick_labels = None
        num_ticks = len(array)
    im = ax.imshow(array, cmap="Blues")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.mappable.set_clim(vmin=0, vmax=np.nanmax(array))

    if x_tick_labels is None:
        tick_interval = 2
    else:
        tick_interval = 1
    ax.set_xticks(np.arange(0, num_ticks, tick_interval), labels=x_tick_labels)
    ax.set_yticks(np.arange(0, num_ticks, tick_interval), labels=y_tick_labels)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="default")

    labelsize = 10 if num_ticks < 50 else 8
    ax.tick_params(axis="both", which="both", labelsize=labelsize)

    if num_ticks < 30:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                n_preds = array[i, j]
                if not np.isnan(n_preds):
                    ax.text(
                        j,
                        i,
                        f"{n_preds:.2f}" if normalize else f"{n_preds:.0f}",
                        ha="center",
                        va="center",
                        color="black"
                        if n_preds < 0.5 * np.nanmax(array)
                        else "white",
                    )

    if title:
        ax.set_title(title, fontsize=20)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_facecolor("white")
    if save_path:
        fig.savefig(
            save_path, dpi=250, facecolor=fig.get_facecolor(), transparent=True
        )
    return fig

def is_corner_zero_pixels(image, threshold=0.8, corner_size=20):
    height, width = image.shape[:2]

    # Define corner regions
    corners = [
        image[:corner_size, :corner_size],  # Top-left
        image[:corner_size, -corner_size:],  # Top-right
        image[-corner_size:, :corner_size],  # Bottom-left
        image[-corner_size:, -corner_size:],  # Bottom-right
    ]

    # Check if 80% of the pixels in the corners are zero
    for corner in corners:
        if np.mean(corner == 0) < threshold:
            return False
    return True

def mask_in_corner_region(mask, image_shape, threshold=0.2, corner_size=20):
    height, width = image_shape[:2]

    # Define corner regions
    corners_masks = [
        mask[:corner_size, :corner_size],  # Top-left
        mask[:corner_size, -corner_size:],  # Top-right
        mask[-corner_size:, :corner_size],  # Bottom-left
        mask[-corner_size:, -corner_size:],  # Bottom-right
    ]

    total_mask_pixels = torch.sum(mask)
    for corner_mask in corners_masks:
        corner_pixels = torch.sum(corner_mask)
        if corner_pixels / total_mask_pixels >= threshold:
            return True
    return False