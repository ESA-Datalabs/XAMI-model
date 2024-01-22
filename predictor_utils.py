import traceback
import numpy as np
import cv2
from PIL import Image
import supervision as sv
from astropy.io import fits
import matplotlib.pyplot as plt
from astronomy_utils import data_norm

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

def remove_masks(sam_result, mask_on_negative, threshold, remove_big_masks=False, big_masks_threshold=None, img_shape=None):
    '''
    Given a segmentation result, this function removes the masks 
    if the intersection with the negative pixels gives a number is > than a threshold
    '''
    big_masks_threshold = img_shape[0]**2/5 if big_masks_threshold is None else big_masks_threshold
    bad_indices = np.array([],  dtype=int) 
    print(sam_result[0]['segmentation'].shape, mask_on_negative.shape)
    for segm_index in range(len(sam_result)):
        count = np.sum((sam_result[segm_index]['segmentation'] == 1) & (mask_on_negative == 1))            
        # remove masks on negative pixels given threshold
        if count > threshold:
            bad_indices = np.append(bad_indices, segm_index)
        
        # remove very big (>70) masks
        if remove_big_masks and img_shape is not None and np.sum(sam_result[segm_index]['segmentation']) > big_masks_threshold:
            print(f"Removing mask {segm_index} with area {np.sum(sam_result[segm_index]['segmentation'])}")
            bad_indices = np.append(bad_indices, segm_index)   
    sam_result = np.delete(sam_result, bad_indices)
    return sam_result

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