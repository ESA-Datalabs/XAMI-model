from torch import tensor
from pprint import pprint
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

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

def ious_pred_vs_gt(gts, preds):
    all_ious_pred_vs_gt_flatten = []
    all_ious_pred_vs_gt = []
    
    for i in range(len(gts)):
        iou_image_scores = []
        for gt_mask, pred_mask in zip(gts[i], preds[i]):
            intersection = np.logical_and(gt_mask, pred_mask)
            union = np.logical_or(gt_mask, pred_mask)
            iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            iou_image_scores.append(iou_score)
            all_ious_pred_vs_gt_flatten.append(iou_score)
        all_ious_pred_vs_gt.append(iou_image_scores)
        
    return all_ious_pred_vs_gt_flatten, all_ious_pred_vs_gt

def compute_metrics_with_range(gt_masks, pred_masks, image=None):
    """Compute the True Positive, False Positive, and False Negative BINARY masks for multiple segmentations."""
    
    try: # the pred_mask array is empty
        combined_gt_mask = np.zeros_like(gt_masks[0][0], dtype=bool) 
        combined_pred_mask = combined_gt_mask.copy()
    except: # the gt array is empty
        combined_pred_mask = np.zeros_like(pred_masks[0], dtype=bool)
        combined_gt_mask = combined_pred_mask.copy()
        
    for gt_mask in gt_masks:
        combined_gt_mask = np.logical_or(combined_gt_mask, gt_mask) # all gt masks combined

    for pred_mask in pred_masks:
        combined_pred_mask = np.logical_or(combined_pred_mask, pred_mask.astype(bool)) # all pred masks combined

    intersection = np.sum(np.logical_and(combined_gt_mask, combined_pred_mask))
    union = np.sum(np.logical_or(combined_gt_mask, combined_pred_mask))

    iou = intersection/union if union > 0 else 0
    
    true_positive_mask = np.logical_and(combined_gt_mask, combined_pred_mask)
    false_negative_mask = np.logical_and(combined_gt_mask, np.logical_not(combined_pred_mask))
    false_positive_mask = np.logical_and(combined_pred_mask, np.logical_not(combined_gt_mask))

    return true_positive_mask, false_positive_mask, false_negative_mask, iou

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

def plot_ious(train_all_ious_pred_vs_gt, valid_all_ious_pred_vs_gt, box_anchor=(0.63, 0.21)):
    rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14,
        'font.family': 'sans-serif'
    })

    sns.set(style="whitegrid")
    colors = ['#EE82EE', '#318CE7']

    # Calculate histograms and cumulative distributions
    vals_train, base_train = np.histogram(train_all_ious_pred_vs_gt, bins=1000, range=(0, 1), density=True)
    cumulative_train = np.cumsum(vals_train * np.diff(base_train))
    vals_valid, base_valid = np.histogram(valid_all_ious_pred_vs_gt, bins=1000, range=(0, 1), density=True)
    cumulative_valid = np.cumsum(vals_valid * np.diff(base_valid))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: Cumulative Distribution Function (CDF)
    axes[0].plot(base_valid[:-1], cumulative_valid, label='Validation', color=colors[1], linewidth=2.5, linestyle='--')
    axes[0].plot(base_train[:-1], cumulative_train, label='Train', color=colors[0], linewidth=2.5)

    # Add quartiles
    quartiles_y = np.quantile(cumulative_valid, [0.8, 0.9])
    quartiles_x = np.interp(quartiles_y, cumulative_valid, base_valid[:-1])
    for qx, qy in zip(quartiles_x, quartiles_y):
        axes[0].hlines(qy, xmin=0, xmax=qx, colors='grey', linestyles='dashed')
        axes[0].vlines(qx, ymin=0, ymax=qy, colors='grey', linestyles='dashed')
        axes[0].text(qx, qy, f'({qx:.2f}, {qy:.2f})', fontsize=21, verticalalignment='bottom', horizontalalignment='right', color='black')

    axes[0].set_xticks(np.arange(0, 1.1, 0.2))
    axes[0].set_yticks(np.arange(0, 1.1, 0.2))
    axes[0].set_xlabel('IoU', fontsize=24)
    axes[0].set_ylabel('Cumulative Frequency', fontsize=24)
    axes[0].tick_params(axis='both', which='major', labelsize=21)
    axes[0].legend(fontsize=18, bbox_to_anchor=box_anchor)
    axes[0].grid(False)

    # Plot 2: Box Plot
    box = axes[1].boxplot(
        [train_all_ious_pred_vs_gt, valid_all_ious_pred_vs_gt],
        labels=['Training', 'Validation'],
        patch_artist=True,
        medianprops=dict(color='black'),
        widths=0.3
    )
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for whisker in box['whiskers']:
        whisker.set(color='gray', linewidth=2)
    for cap in box['caps']:
        cap.set(color='gray', linewidth=1.5)
    for median in box['medians']:
        median.set(color='black', linewidth=2)
    for flier in box['fliers']:
        flier.set(marker='o', color='gray', alpha=0.6)

    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Training', 'Validation'], fontsize=21)
    axes[1].set_yticks(np.arange(0, 1.1, 0.2))
    axes[1].set_ylabel('IoU', fontsize=25)
    axes[1].tick_params(axis='both', which='major', labelsize=21)
    axes[1].grid(False)

    plt.tight_layout()
    plt.show()
