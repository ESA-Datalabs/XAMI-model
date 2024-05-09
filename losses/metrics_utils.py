from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint
import torch
import numpy as np

def flatten_ious_areas(pred_classes, all_iou_scores, mask_areas=None):
    all_ious_flatten = []
    if mask_areas is not None:
        mask_areas_flatten = []
    for i in range(len(pred_classes)):
        for j in range(len(pred_classes[i])):
            all_ious_flatten.append(all_iou_scores[i][j][0])
            if mask_areas is not None:
                mask_areas_flatten.append(mask_areas[i][j])

    if mask_areas is not None:
        return all_ious_flatten, mask_areas_flatten
    else:
        return all_ious_flatten
        
def mAP_metrics(map_metric,
                preds,
                gts, 
                gt_classes, 
                pred_classes, 
                all_iou_scores,
                mask_areas,
			    show_metrics=False):
    
    all_preds, all_gts, all_ious = [], [], []
    for i in range(len(preds)):
        gt_i = np.array([gts[i][j][0] for j in range(len(gts[i]))])
        pred_i = np.array([preds[i][j] for j in range(len(preds[i]))])
        ious_i = np.array([all_iou_scores[i][j][0] for j in range(len(all_iou_scores[i]))])
        all_gts.append(gt_i)    
        all_preds.append(pred_i)  
        all_ious.append(ious_i)
        
    all_gt_classes = [np.array(gt_classes[i], dtype=np.int8) for i in range(len(gt_classes))]
    all_pred_classes = [np.array(pred_classes[i], dtype=np.int8) for i in range(len(pred_classes))]
    
    preds_per_image = []
    gts_per_image = []
    
    for img_i in range(len(all_preds)):
        img_preds_dict = dict(
            masks=tensor(all_preds[img_i], dtype=torch.bool),
            scores=tensor(all_ious[img_i]),
            labels=tensor(all_pred_classes[img_i], dtype=torch.int16),
          )
        img_gts_dict = dict(
            masks=tensor(all_gts[img_i], dtype=torch.bool),
            labels=tensor(all_gt_classes[img_i], dtype=torch.int16),
          )
        
        preds_per_image.append(img_preds_dict)
        gts_per_image.append(img_gts_dict)

    map_metric.update(preds_per_image, gts_per_image)
    if show_metrics:
        pprint(map_metric.compute())
		
    return map_metric.compute()

def compute_metrics(gt_masks, pred_masks, iou_threshold, image=None):
    """Compute the True Positive, False Positive, and False Negative BINARY masks for multiple segmentations."""
    
    # if image is not None:
        # plt.imshow(image)
        # for mask in gt_masks:
        #     dataset_utils.show_mask(mask[0], plt.gca())
        # plt.show()
        # plt.close()
    
        # plt.imshow(image)
        # for mask in pred_masks:
        #     dataset_utils.show_mask(mask, plt.gca())
        # plt.show()
        # plt.close()
        
    combined_gt_mask = np.zeros_like(gt_masks[0][0], dtype=bool)
    combined_pred_mask = np.zeros_like(pred_masks[0], dtype=bool)
    filtered_pred_masks = np.zeros_like(pred_masks, dtype=bool)
    combined_filtered_pred_masks = np.zeros_like(pred_masks[0], dtype=bool)
    
    # print(gt_masks.shape, pred_masks.shape) # (N, 1, H, W), (N, H, W)
    for i, pred_mask in enumerate(pred_masks):
        max_iou = 0  
        
        for gt_mask in gt_masks:
            intersection = np.logical_and(gt_mask[0], pred_mask)
            union = np.logical_or(gt_mask[0], pred_mask)
            iou_score = np.sum(intersection)/np.sum(union) if np.sum(union) > 0 else 0
            max_iou = max(max_iou, iou_score)  

        if max_iou > iou_threshold: # take IoUs above threshold
            filtered_pred_masks[i] = pred_mask

    for gt_mask in gt_masks:
        combined_gt_mask = np.logical_or(combined_gt_mask, gt_mask.astype(bool))
    for pred_mask in filtered_pred_masks:
        combined_filtered_pred_masks = np.logical_or(combined_filtered_pred_masks, pred_mask.astype(bool))
    for pred_mask in pred_masks:
        combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask.astype(bool))

    true_positive_mask = np.logical_and(combined_gt_mask, combined_filtered_pred_masks)
    false_negative_mask = np.logical_and(combined_gt_mask, np.logical_not(combined_filtered_pred_masks))
    false_positive_mask = np.logical_and(combined_pred_mask, np.logical_not(combined_gt_mask))

    return true_positive_mask, false_positive_mask, false_negative_mask