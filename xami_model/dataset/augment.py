import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import random
import math
from typing import Dict, List, Any
from . import dataset_utils
import pycocotools.mask as maskUtils

BOX_COLOR = (255, 0, 0) 
TEXT_COLOR = (0, 0, 255) 

def enlarge_bbox(bbox, delta, image_size):
    x_min, y_min, w, h = bbox
    x_min = max(0, math.floor(x_min))
    y_min = max(0,  math.floor(y_min))
    
    w = min(image_size[1], math.ceil(w))
    h = min(image_size[0], math.ceil(h))
    return np.array([x_min, y_min, w, h])

def align_masks_and_bboxes(augmented_set):
    '''
    Sometimes, the masks and bboxes returned by the Augmentation process do not have the same size, and (thus) they are not aligned by index. 
    This function will keep the non-empty masks and generate another bboxes given those masks.
    '''
    
    bboxes_augm = []
    masks_augm = []
    
    for mask_i in augmented_set['masks']:
        if np.any(mask_i):
            bbox = cv2.boundingRect(mask_i)
            bboxes_augm.append(bbox)
            masks_augm.append(mask_i)
    
    augmented_set['masks'] = masks_augm
    augmented_set['bboxes'] = bboxes_augm
    # augmented_set['category_id'] = [1] * len(masks_augm)
    return augmented_set

def show_augmented_images(
    augmented, 
    bboxes, 
    categories, 
    category_id_to_name, 
    image, 
    masks=None, 
    filename=None, 
    font_scale_orig=0.35, 
    font_scale_aug=0.35,
    **kwargs):
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    image_aug = cv2.cvtColor(augmented['image'].copy(), cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        dataset_utils.visualize_bbox(image, bbox, BOX_COLOR, **kwargs)

    for bbox in augmented['bboxes']:
        dataset_utils.visualize_bbox(image_aug, bbox, TEXT_COLOR, **kwargs)

    for bbox,cat_id in zip(bboxes, categories):
        dataset_utils.visualize_titles(image, bbox, category_id_to_name[cat_id], font_scale=font_scale_orig, **kwargs)
    for bbox,cat_id in zip(augmented['bboxes'], augmented['category_id']):
        dataset_utils. visualize_titles(image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)

    if masks is None:
        f, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(image)
        ax[0].set_title('Original image')
        ax[1].imshow(image_aug)
        ax[1].set_title('Augmented image')
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 16))

        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Original image')
        ax[0, 1].imshow(image_aug)
        ax[0, 1].set_title('Augmented image')

        height, width = masks[0].shape
        black_image = np.zeros((height, width, 3), dtype=np.uint8)

        for mask in masks:
            colored_mask = np.zeros_like(black_image)  
            colored_mask[mask == 1] = dataset_utils.random_color() 
            black_image = cv2.addWeighted(black_image, 1, colored_mask, 0.5, 0)

        ax[1, 0].imshow(black_image, interpolation='nearest')
        ax[1, 0].set_title('Original masks')
        
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        for mask in augmented['masks']:
            colored_mask = np.zeros_like(black_image)  
            colored_mask[mask == 1] = dataset_utils.random_color()  
            black_image = cv2.addWeighted(black_image, 1, colored_mask, 0.5, 0)
        ax[1, 1].imshow(black_image, interpolation='nearest')
        ax[1, 1].set_title('Augmented masks')
    f.tight_layout()
    plt.show()
    plt.close()

    if filename is not None:
        f.savefig(filename)
            
def augment_and_show(aug, image, masks=None, bboxes=[], categories=[], category_id_to_name=[], filename=None, 
                     font_scale_orig=0.35, font_scale_aug=0.35, show_=True, **kwargs):

    augmented = aug(image=image, masks=masks, bboxes=bboxes, category_id=categories)
    augmented['masks'] = [mask for mask in augmented['masks'] if np.any(mask)]
    
    if show_:
        show_augmented_images(
            augmented, 
            bboxes, 
            categories, 
            category_id_to_name, 
            image, 
            masks, 
            filename, 
            font_scale_orig, 
            font_scale_aug, 
            **kwargs)
        
    return augmented

def update_dataset_with_augms(
                              augmented_set: Dict[str, Any], 
                              new_filename: str, bbox_coords: Dict[str, Any], 
                              ground_truth_masks: Dict[str, Any], 
                              image_paths: List[str],
                              classes: Dict[str, Any],
                              **kwargs):
    """
    Updates the dataset with augmented images, masks, and bounding box coordinates.

    Args:
        augmented_set (Dict[str, Any]): The augmented set containing masks, bboxes, and the image.
        new_filename (str): The path where the new image will be saved.
        bbox_coords (Dict[str, Any]): The bounding box coordinates of the dataset.
        ground_truth_masks (Dict[str, Any]): The ground truth masks of the dataset.
        image_paths (List[str]): The list of image paths in the dataset.

    Returns:
        None (the values are updated through pass by reference)
    """

    # Save the new image
    cv2.imwrite(new_filename, augmented_set['image'])

    # Add image path to the dataset
    image_paths.append(new_filename)
    # print('nb masks:', len(augmented_set['masks']), 'cats:', len(augmented_set['category_id']))

    # new_annotations = []
    
    # image_id = len(data.get('images', [])) + 1
    # from datetime import datetime
    # image = cv2.imread(new_filename)

    # height, width = image.shape[:2]
    # image={
	# 		'id': image_id,
	# 		"license": 1,
	# 		'file_name': new_filename,
	# 		"height": height,
	# 		"width": width,
	# 		"date_captured": datetime.now().isoformat()
	# 	}
    # data['images'].append(image)
    
    # Add image masks and bboxes to the dataset
    for mask_i in range(len(augmented_set['bboxes'])):
        xyxy_bbox = np.array([augmented_set['bboxes'][mask_i][0], augmented_set['bboxes'][mask_i][1],
                              augmented_set['bboxes'][mask_i][2] + augmented_set['bboxes'][mask_i][0],
                              augmented_set['bboxes'][mask_i][3] + augmented_set['bboxes'][mask_i][1]])

        bbox_coords[f'{new_filename.split("/")[-1]}_mask{mask_i}'] = xyxy_bbox
        ground_truth_masks[f'{new_filename.split("/")[-1]}_mask{mask_i}'] = maskUtils.encode(np.asfortranarray(augmented_set['masks'][mask_i]))
        classes[f'{new_filename.split("/")[-1]}_mask{mask_i}'] = augmented_set['category_id'][mask_i]
        
    #     new_annotations = {
    #         'image_id': image_id,
    #         'category_id': augmented_set['category_id'][mask_i],
    #         'bbox': xyxy_bbox.tolist(),
    #         'segmentation': augmented_set['masks'][mask_i],
    #         'area': np.sum(augmented_set['masks'][mask_i]),
    #         'iscrowd': 0
    #     }
        
    #     data['annotations'].append(new_annotations)
    
    # return data
        