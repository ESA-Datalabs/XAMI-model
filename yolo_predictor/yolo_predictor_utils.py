from typing import Any, Generator, List
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import dataset_utils

# get masks from dataset (in YOLOv8 format) given an image file
def get_label_file_path(dataset_path, image_location):
    dataset_path = '/'.join(dataset_path.split('/')[:-2])+'/'+'labels'+'/'
    label_file_path = os.path.join(dataset_path, image_location)
    label_loc = '.'.join(image_location.split('.')[:-1]) + '.txt'
    label_file_path = dataset_path+label_loc
    return label_file_path

def read_annotations(label_file_path):
    annotations = []
    k = 0
    with open(label_file_path, 'r') as file:
        for line in file:
            k+=1
            parts = line.strip().split()
            class_id = int(parts[0])
            segmentation_points = [float(p) for p in parts[1:]]
            annotations.append({
                'class_id': class_id,
                'segmentation_points': segmentation_points
            })
    return annotations

def get_masks_from_image(yolo_dataset_path, image_location):
    label_file_path = get_label_file_path(yolo_dataset_path, image_location)
    annotations = read_annotations(label_file_path)
    masks = [dataset_utils.create_mask_0_1(annot['segmentation_points'], (512, 512)) for annot in annotations]
    return masks

def get_classes_from_image(yolo_dataset_path, image_location):
    label_file_path = get_label_file_path(yolo_dataset_path, image_location)
    annotations = read_annotations(label_file_path)
    class_ids = [annot['class_id'] for annot in annotations]
    return class_ids

def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]