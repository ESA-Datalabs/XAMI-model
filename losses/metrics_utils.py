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
                mask_areas):
    
    all_ious_flatten = flatten_ious_areas(pred_classes, all_iou_scores)

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
    
    print(len(all_preds), len(all_gts), len(all_ious), len(all_gt_classes), len(all_pred_classes))

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

    print(preds_per_image[1]['masks'].shape, preds_per_image[1]['scores'].shape, preds_per_image[1]['labels'].shape)
    
    map_metric.update(preds_per_image, gts_per_image)
    pprint(map_metric.compute())
    return map_metric.compute()
