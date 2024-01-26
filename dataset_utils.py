import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Any

BOX_COLOR = (255, 0, 0) 
TEXT_COLOR = (0, 0, 255) 

def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def visualize_bbox(img, bbox, color, thickness=1, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img

def visualize_titles(img, bbox, title, font_thickness = 1, font_scale=0.35, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                font_thickness, lineType=cv2.LINE_AA)
    return img

def plot_bboxes_and_masks(image, bboxes, masks):
    
    image_copy = image.copy()
    image_copy = image_copy.astype('uint8')

    for i in range(len(bboxes)): 
        start_point = (int(bboxes[i][0]), int(bboxes[i][1]))  
        end_point = (int(bboxes[i][2]+bboxes[i][0]), int(bboxes[i][3]+bboxes[i][1]))  
        color = random_color()
        cv2.rectangle(image_copy, start_point, end_point, color, 2)

        colored_mask = np.zeros_like(image_copy)
        colored_mask[masks[i] >0] = color

        image_copy = cv2.addWeighted(image_copy, 1, colored_mask, .2, 0.9)

        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        plt.close()

def get_coords_and_masks_from_json(input_dir, data_in, image_key=None):
    """
    Extracts masks and bounding box coordinates from a JSON object containing image annotations.
    
    Parameters:
    - data_in (dict): The JSON dataset split containing image and annotation details.
    - image_key (str, optional): A string key that identifies the specific image for which annotations are required. 
    If provided, the function will only process the image with the matching key.
    
    Returns:
    - result_masks (dict): A dictionary of type {mask: mask array}.
    - bbox_coords (dict): A dictionary of type {mask:bounding box coordinates corresponding to that mask}.
    
    For each annotation, it keeps it only if its bounding box size is significant (h,w) > (5,5).
    """
    result_masks, bbox_coords, result_class = {}, {}, {}
	
    class_categories = {data_in['categories'][a]['id']:data_in['categories'][a]['name'] for a in range(len(data_in['categories']))}

    if image_key is not None:
        image_id = None
        for img_id in range(len(data_in['images'])):
            if data_in['images'][img_id]['file_name'] == image_key:
                image_id = img_id 
                
        temp_img = cv2.imread(input_dir+data_in['images'][img_id]['file_name'])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        h_img, w_img = temp_img.shape[0], temp_img.shape[1]
        del temp_img
        
        masks = [data_in['annotations'][a] for a in range(len(data_in['annotations'])) if data_in['annotations'][a]['image_id'] == image_id]
        for i in range(len(masks)):
            xyhw = masks[i]['bbox']
            if xyhw[2]>5 or xyhw[3]>5: # ignore masks with an (h,w) < (5,5)
                points = masks[i]['segmentation'][0]
                mask = create_mask(points, (h_img, w_img)) # Roboflow segmentations are polygon points, and are be converted to masks
                result_masks[f'{image_key}_mask{i}'] = mask
                bbox_coords[f'{image_key}_mask{i}'] = [xyhw[0], xyhw[1], xyhw[2]+ xyhw[0], xyhw[3]+xyhw[1]]
        return result_masks, bbox_coords

    # for all images in set
    for im in data_in['images']:        
        masks = [data_in['annotations'][a] for a in range(len(data_in['annotations'])) if data_in['annotations'][a]['image_id'] == im['id']]
        classes = [data_in['annotations'][a]['category_id'] for a in range(len(data_in['annotations'])) if data_in['annotations'][a]['image_id'] == im['id']]
        for i in range(len(masks)):
            xyhw = masks[i]['bbox']
            if xyhw[2]>5 or xyhw[3]>5: # ignore masks with an (h,w) < (5,5)
                points = masks[i]['segmentation'][0]
                temp_img = cv2.imread(input_dir+im["file_name"])
                temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                h_img, w_img = temp_img.shape[0], temp_img.shape[1]
                del temp_img
        
                mask = create_mask(points, (h_img, w_img)) # Roboflow segmentations are polygon points, and are be converted to masks
                result_masks[f'{im["file_name"]}_mask{i}'] = mask
                bbox_coords[f'{im["file_name"]}_mask{i}'] = [xyhw[0], xyhw[1], xyhw[2]+ xyhw[0], xyhw[3]+xyhw[1]]
                result_class[f'{im["file_name"]}_mask{i}'] = classes[i]
	
    return result_masks, bbox_coords, result_class, class_categories

def create_mask(points, image_size):
    polygon = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
    mask = np.zeros(image_size, dtype=np.uint8)
    
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask

def augment_and_show(aug, image, masks=None, bboxes=[], categories=[], category_id_to_name=[], filename=None, 
                     font_scale_orig=0.35, font_scale_aug=0.35, show_title=True, **kwargs):

    augmented = aug(image=image, masks=masks, bboxes=bboxes, category_id=categories)
    # image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # image_aug = cv2.cvtColor(augmented['image'].copy(), cv2.COLOR_BGR2RGB)

    # for bbox in bboxes:
    #     visualize_bbox(image, bbox, BOX_COLOR, **kwargs)

    # for bbox in augmented['bboxes']:
    #     visualize_bbox(image_aug, bbox, TEXT_COLOR, **kwargs)

    # if show_title:
    #     for bbox,cat_id in zip(bboxes, categories):
    #         visualize_titles(image, bbox, category_id_to_name[cat_id], font_scale=font_scale_orig, **kwargs)
    #     for bbox,cat_id in zip(augmented['bboxes'], augmented['category_id']):
    #         visualize_titles(image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)

    # if masks is None:
    #     f, ax = plt.subplots(1, 2, figsize=(12, 6))
        
    #     ax[0].imshow(image)
    #     ax[0].set_title('Original image')
    #     ax[1].imshow(image_aug)
    #     ax[1].set_title('Augmented image')
    # else:
    #     f, ax = plt.subplots(2, 2, figsize=(16, 16))

    #     ax[0, 0].imshow(image)
    #     ax[0, 0].set_title('Original image')
    #     ax[0, 1].imshow(image_aug)
    #     ax[0, 1].set_title('Augmented image')

    #     height, width = masks[0].shape
    #     black_image = np.zeros((height, width, 3), dtype=np.uint8)

    #     for mask in masks:
    #         colored_mask = np.zeros_like(black_image)  
    #         colored_mask[mask == 1] = random_color() 
    #         black_image = cv2.addWeighted(black_image, 1, colored_mask, 0.5, 0)

    #     ax[1, 0].imshow(black_image, interpolation='nearest')
    #     ax[1, 0].set_title('Original masks')
        
    #     black_image = np.zeros((height, width, 3), dtype=np.uint8)
    #     for mask in augmented['masks']:
    #         colored_mask = np.zeros_like(black_image)  
    #         colored_mask[mask == 1] = random_color()  
    #         black_image = cv2.addWeighted(black_image, 1, colored_mask, 0.5, 0)
    #     ax[1, 1].imshow(black_image, interpolation='nearest')
    #     ax[1, 1].set_title('Augmented masks')
    # f.tight_layout()

    # if filename is not None:
    #     f.savefig(filename)
        
    return augmented

def update_dataset_with_augms(augmented_set: Dict[str, Any], new_filename: str, bbox_coords: Dict[str, Any], ground_truth_masks: Dict[str, Any], image_paths: List[str]):
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
    # Remove empty masks
    # print('Initial number of masks:', len(augmented_set['masks']), ' Number of bboxes:', len(augmented_set['bboxes']))

    # if len(augmented_set['masks']) != len(augmented_set['bboxes']):
        # print(f"❗The number of masks is different than the number of bboxes. #masks = {len(augmented_set['masks'])}, #bboxes: {len(augmented_set['bboxes'])}")

    augmented_set['masks'] = [mask for mask in augmented_set['masks'] if np.any(mask)]

    # Reduce masks/bboxes to use the same length
    min_len = min(len(augmented_set['masks']), len(augmented_set['bboxes']))
    augmented_set['masks'] = augmented_set['masks'][:min_len]
    augmented_set['bboxes'] = augmented_set['bboxes'][:min_len]

    # Save the new image
    cv2.imwrite(new_filename, augmented_set['image'])

    # Add image path to the dataset
    image_paths.append(new_filename)

    # Add image masks and bboxes to the dataset
    for mask_i in range(len(augmented_set['masks'])):
        hywh_bbox = np.array([augmented_set['bboxes'][mask_i][0], augmented_set['bboxes'][mask_i][1],
                              augmented_set['bboxes'][mask_i][2] + augmented_set['bboxes'][mask_i][0],
                              augmented_set['bboxes'][mask_i][3] + augmented_set['bboxes'][mask_i][1]])

        bbox_coords[f'{new_filename.split("/")[-1]}_mask{mask_i}'] = hywh_bbox
        ground_truth_masks[f'{new_filename.split("/")[-1]}_mask{mask_i}'] = augmented_set['masks'][mask_i]