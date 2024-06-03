from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Any
import pycocotools.mask as maskUtils
import json
from astropy.io import fits
from pyparsing import col
import pywt

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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([65/255, 174/255, 255/255, 0.4]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks(masks, ax, random_color=False, colours=None):
    for i in range(len(masks)):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        elif colours is not None:
            color = np.array([c/255.0 for c in colours[i]]+[0.6])
            # color = np.array(list(colours[i])+[0.6])
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = masks[i].shape[-2:]
        mask_image = masks[i].reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=1))    

def mask_and_plot_image(fits_file, plot_ = False):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data

    data = np.flipud(data)
    mask = data <= 0    
    if plot_ and len(data[data>0])>=1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(data, cmap='gray', norm = dataset_utils.data_norm(data[data>0]))
        ax1.set_title(f'{fits_file.split("/")[-1].split(".")[0]}')
        ax1.axis('off')
        ax2.imshow(255-mask, cmap='gray')
        ax2.set_title('Negative pixels map')
        ax2.axis('off')
        plt.savefig(f'./negative_mask.png')
        plt.show()
        plt.close()      
    return mask

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

def draw_label(img, text, pos, bg_color, alpha=0.8):

    if 'read-out-streak' in text:
        text = 'ROS'
    if 'smoke' in text:
        text = 'SR'
    if 'central' in text:
        text = 'CR'

    if 'loop' in text:
        text = 'SG'
    if 'scattered' in text:
        text = 'scattered'
        
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 0, 0)  # Black text
    thickness = cv2.FILLED
    margin = 0

    txt_size = cv2.getTextSize(text, font_face, scale, 1)
    txt_width, txt_height = txt_size[0][0], txt_size[0][1]

    # Position of the rectangle
    rect_start = (pos[0], pos[1] - txt_height - margin * 2)
    rect_end = (pos[0] + txt_width + margin * 2, pos[1])

    # Create a transparent overlay for the rectangle
    overlay = img.copy()

    if not (text.endswith('star') or text.endswith('star')):
        cv2.rectangle(overlay, rect_start, rect_end, bg_color, thickness)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw the text on the original image
        text_pos = (pos[0] + margin * 2, pos[1] - margin)
        cv2.putText(img, text, text_pos, font_face, scale, color, 1, cv2.LINE_AA)

def visualize_masks(image_path, image, masks, labels, colors, alpha=0.4, is_gt=True):
    """
    Visualize masks on an image with contours and dynamically sized text labels with a background box.

    Parameters:
    - image_path: The path to the image file.
    - image: The original image (numpy array).
    - masks: A list of masks (numpy arrays), one for each label.
    - labels: A list of labels corresponding to each mask.
    - colors: A list of colors corresponding to each label.
    - alpha: Transparency of masks.
    """
    # Pad the image (when labels are too big and they appear at image corners)
    pad_size = 20  # Padding size
    if len(image.shape) == 3:
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                          cv2.BORDER_CONSTANT, value=[255,255,255])
    else:  # For grayscale images
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                          cv2.BORDER_CONSTANT, value=255)

    # Work on a copy of the padded image
    temp_image = padded_image.copy()

    # Ensure the temp_image is in RGB format
    if len(temp_image.shape) == 2 or temp_image.shape[2] == 1:
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2RGB)
    elif temp_image.shape[2] == 4:
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGRA2RGB)

    # Create a color overlay
    overlay = temp_image.copy()
    for mask, label in zip(masks, labels):
        # Adjust mask for padding
        padded_mask = cv2.copyMakeBorder(mask, pad_size, pad_size, pad_size, pad_size,
                                         cv2.BORDER_CONSTANT, value=0)

        color = colors[label]
        bg_color = (int(color[0]), int(color[1]), int(color[2]))  # Background color

        overlay[padded_mask == 1] = [color[0], color[1], color[2]]

        contours, _ = cv2.findContours(padded_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_color = [c // 4 for c in color]  # Darker shade of the mask color
        cv2.drawContours(temp_image, contours, -3, contour_color, 2) 

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            draw_label(temp_image, label, (x, y), bg_color)

    cv2.addWeighted(overlay, alpha, temp_image, 1 - alpha, 0, temp_image)

    # Visualization code remains the same
    plt.figure(figsize=(50, 50)) 
    plt.imshow(temp_image)
    if is_gt:
        plt.title(f'Ground truth classes \n{image_path.split(".")[0]}', fontsize=100)
        output_file= './plots/'+image_path.split(".")[0]+'_gt.png'
    else:
        plt.title(f'Predicted classes \n{image_path.split(".")[0]}', fontsize=100)
        output_file= './plots/'+image_path.split(".")[0]+'_pred.png'

    plt.axis('off')
    plt.show()
    plt.imsave(output_file, temp_image, dpi=1000)
    plt.close()
    
def isolate_background(image, decomposition='db1', level=2, sigma=1):
    """
    Isolates and visualizes parts of an image that are close to the estimated background.
    
    Parameters:
    - image: numpy.ndarray. The input image for background isolation.
    - decomposition: str, optional. The name of the wavelet to use for decomposition. Default is 'db1'.
    - level: int, optional. The level of wavelet decomposition to perform. Default is 2.
    - sigma: float, optional. The number of standard deviations from the background mean to consider as background. Default is 1.
    
    Returns:
    - mask:  numpy.ndarray. A binary mask indicating parts of the image close to the background.
    - close_to_background: numpy.ndarray. The parts of the original image that are close to the background.
    """
    
    # Perform a multi-level wavelet decomposition
    coeffs = pywt.wavedec2(image, decomposition, level=level)
    if len(coeffs) == 3:
        cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
        # Reconstruct the background image from the approximation coefficients
        background = pywt.waverec2((cA2, (None, None, None), (None, None, None)), decomposition)
    
    elif len(coeffs) == 2:
        cA1, (cH1, cV1, cD1) = coeffs
        background = pywt.waverec2((cA1, (None, None, None), (None, None, None)), decomposition)
        

    # Calculate the mean and standard deviation of the background
    mean_bg = np.mean(background)
    std_bg = np.std(background)
    
    # Define a threshold based on mean and standard deviation
    lower_bound = mean_bg - std_bg * sigma
    upper_bound = mean_bg + std_bg * sigma
    
    # Create a mask where pixel intensities are close to the background
    mask = (image >= lower_bound) & (image <= upper_bound)
    
    # Apply the mask to the image
    close_to_background = image * mask
    
    # # Visualize the results
    # plt.imshow(mask, cmap='gray')
    # plt.title("Binary Mask")
    # plt.show()
    # plt.close()
    
    # plt.imshow(close_to_background, cmap='gray')
    # plt.title("Image Close to Background")
    # plt.show()
    # plt.close()
    
    return mask, close_to_background

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

    for im in data_in['images']:  

        masks = [data_in['annotations'][a] for a in range(len(data_in['annotations'])) 
                 if data_in['annotations'][a]['image_id'] == im['id']]
        temp_img = cv2.imread(input_dir+im["file_name"])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        classes = [data_in['annotations'][a]['category_id'] for a in range(len(data_in['annotations'])) 
                   if data_in['annotations'][a]['image_id'] == im['id']]
        
        for i in range(len(masks)):
            segmentation = masks[i]['segmentation']
            if isinstance(segmentation, list):
                if len(segmentation) > 0 and isinstance(segmentation[0], list):
                    points = segmentation[0]
                    h_img, w_img = temp_img.shape[:2]
                    mask = create_mask(points, (h_img, w_img)) # COCO segmentations are polygon points, and must be converted to masks
                    bbox = mask_to_bbox(mask)  #xyhw[0], xyhw[1], xyhw[2]+xyhw[0], xyhw[3]+xyhw[1] 
                    # binary mask to RLE 
                    result_masks[f'{im["file_name"]}_mask{i}'] = maskUtils.encode(np.asfortranarray(mask)) #mask
                    bbox_coords[f'{im["file_name"]}_mask{i}'] = bbox
                    result_class[f'{im["file_name"]}_mask{i}'] = classes[i]

            elif isinstance(segmentation, dict): # TODO: handle this
                # Handle RLE segmentation
                if 'counts' in segmentation and 'size' in segmentation:
                    rle = maskUtils.frPyObjects([segmentation], segmentation['size'][0], segmentation['size'][1])
                    mask = maskUtils.decode(rle)
                    # result_masks[f'{im["file_name"]}_mask{i}'] = mask
                    # bbox_coords[f'{im["file_name"]}_mask{i}'] = [xyhw[0], xyhw[1], xyhw[2]+ xyhw[0], xyhw[3]+xyhw[1]]
                    # result_class[f'{im["file_name"]}_mask{i}'] = classes[i]
                    # Now `mask` is a binary mask of shape `(height, width)` where `segmentation['size']` = [height, width]
                    
        del temp_img
	
    return result_masks, bbox_coords, result_class, class_categories

def mask_to_bbox(mask):
    """
    Calculate the bounding box from the mask.
    mask: binary mask of shape (height, width) with non-zero values indicating the object
    Returns: bbox in the format [x_min, y_min, x_max, y_max]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [x_min, y_min, x_max, y_max]

def create_mask(points, image_size):
    polygon = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
    mask = np.zeros(image_size, dtype=np.uint8)
    
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask

def create_mask_0_1(points, image_size):
    """
    Create a binary mask from polygon points.

    :param points: List of normalized points (x, y) of the polygon, values between 0 and 1.
    :param image_size: Tuple of (height, width) of the image.
    :return: A binary mask as a numpy array.
    """
    height, width = image_size
    
    # Scale points from normalized coordinates to pixel coordinates
    polygon = [(int(x * width), int(y * height)) for x, y in zip(points[::2], points[1::2])]
    
    mask = np.zeros(image_size, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask

def split_list(input_list, percentages):
    size = len(input_list)
    idx = 0
    output = []
    for percentage in percentages:
        chunk_size = round(percentage * size)
        chunk = input_list[idx : idx + chunk_size]
        output.append(chunk)
        idx += chunk_size
    return output

def create_dataset(image_paths, ground_truth_masks, bbox_coords):
        
    d_gt_masks, d_bboxes = {}, {}
    for img_path in image_paths:
        id = img_path.split('/')[-1]

        d_gt_masks.update({mask_id:mask 
				   for mask_id, mask in ground_truth_masks.items() if mask_id.startswith(id)})
        d_bboxes.update({bbox_id:bbox for bbox_id, bbox in bbox_coords.items() if bbox_id.startswith(id)}) 

    return d_gt_masks, d_bboxes

def load_json(path):
    with open(path) as f:
        return json.load(f)

def merge_coco_jsons(first_json, second_json, output_path):
        
    # Load the first JSON file
    with open(first_json) as f:
        coco1 = json.load(f)

    # Load the second JSON file
    with open(second_json) as f:
        coco2 = json.load(f)

    # Update IDs in coco2 to ensure they are unique and do not overlap with coco1
    max_image_id = max(image['id'] for image in coco1['images'])
    max_annotation_id = max(annotation['id'] for annotation in coco1['annotations'])
    max_category_id = max(category['id'] for category in coco1['categories'])

    # Add an offset to the second coco IDs
    image_id_offset = max_image_id + 1
    annotation_id_offset = max_annotation_id + 1
    # category_id_offset = max_category_id + 1

    # Apply offset to images, annotations, and categories in the second JSON
    for image in coco2['images']:
        image['id'] += image_id_offset

    for annotation in coco2['annotations']:
        annotation['id'] += annotation_id_offset
        annotation['image_id'] += image_id_offset  # Update the image_id reference

    # Merge the two datasets
    merged_coco = {
        'images': coco1['images'] + coco2['images'],
        'annotations': coco1['annotations'] + coco2['annotations'],
        'categories': coco1['categories']  # If categories are the same; otherwise, merge as needed
    }

    # Save the merged annotations to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(merged_coco, f)

