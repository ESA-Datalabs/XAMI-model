import json
import os
import shutil
from dataset import dataset_utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml

class COCOToYOLOConverter:
    def __init__(self, input_path, output_path, annotations_file, plot_yolo_masks=False):
        self.input_path = input_path
        self.output_path = output_path
        self.plot_yolo_masks = plot_yolo_masks
        
        with open(os.path.join(self.input_path, annotations_file)) as f:
            self._data = json.load(f)

        self.file_names = []
        
        # Ensure output directories (will) exist
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.mkdir(os.path.dirname(self.output_path))
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.output_path+'/images/'):
            os.mkdir(self.output_path+'/images/')
        if not os.path.exists(self.output_path+'/labels/'):
            os.mkdir(self.output_path+'/labels/')
    
    def load_images_from_folder(self, folder):
        count = 0
        filenames_from_json = list(set([file_['file_name'] for file_ in self._data['images']]))
        
        for filename in filenames_from_json:
            if filename.split('.')[-1] in ['jpg', 'jpeg', 'png']: 
                source = os.path.join(folder, filename)
                destination = f"{self.output_path}/images/{filename}"
        
                try:
                    shutil.copy(source, destination)
                except shutil.SameFileError:
                    print("Source and destination represent the same file.")
        
                self.file_names.append(filename)
                count += 1
        return count

    def get_img_ann(self, image_id):
        img_ann = []
        isFound = False
        for ann in self._data['annotations']:
            if ann['image_id'] == image_id:
                img_ann.append(ann)
                isFound = True
        return img_ann if isFound else None

    def get_img(self, filename):
        for img in self._data['images']:
            if img['file_name'] == filename:
                return img
            
    def plot_segmentation_contours(self, filename, segments):
        
            image = cv2.imread(f"{self.output_path}/images/{filename}")
            fig, ax = plt.subplots()
            
            # Display the image
            ax.imshow(image)
            
            # Check if the coordinates need normalization
            image_height, image_width = image.shape[:2]
            segm_points = []
            
            for parts in segments:
                class_label = int(parts[0])
                # Check if the points are normalized (values between 0 and 1)
                normalized = all(0 <= float(parts[i]) <= 1 for i in range(1, len(parts), 2))
                
                points = []
                for i in range(1, len(parts), 2):
                    x, y = float(parts[i]), float(parts[i + 1])
                    if normalized:
                        x *= image_width
                        y *= image_height
                    points.append((x, y))
                
                segm_points.append((class_label, points))
            
            for class_label, points in segm_points:
                polygon = plt.Polygon(points, edgecolor='r', facecolor='none')
                ax.add_patch(polygon)
                ax.text(points[0][0], points[0][1], str(class_label), color='b', fontsize=12)
            
            ax.set_aspect('equal', 'box')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'YOLOv8 Segmentation Contours\n{filename.split(".")[0]}')
            plt.show()

    def coco_to_yolo(self):
        count = 0
        for filename in self.file_names:
            img = self.get_img(filename)
            img_id = img['id']          
            img_w = img['width']
            img_h = img['height']
            annotation_path = f"{self.output_path}/labels/{'.'.join(filename.split('.')[:-1])}.txt"
        
            img_ann = self.get_img_ann(img_id)
            if img_ann is None:  # the image doesn't have annotations
                with open(annotation_path, 'w') as f:
                    pass  # An empty file is created
                continue

            # plt.figure(figsize=(30, 30))
            # plt.imshow(plt.imread(f"{self.output_path}images/{filename}"))
            
            with open(annotation_path, "a") as file_object:
                segments = []
                
                for ann in img_ann:
                    cls = ann['category_id']
                    current_category = ann['category_id']
                    flat_points = ann['segmentation'][0]
                    # polygon_points = np.array(flat_points).reshape(-1, 2)
                    mask = dataset_utils.create_mask(flat_points, (img_h, img_w))
                    bbox = dataset_utils.mask_to_bbox(mask)
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    # dataset_utils.show_mask(mask, plt.gca())
                    # dataset_utils.show_box([x, y, x+w, y+h], plt.gca())

                    # Calculate and normalize midpoints
                    x_centre = (x + w / 2) / img_w
                    y_centre = (y + h / 2) / img_h
                    w = w / img_w
                    h = h / img_h
                    
                    if len(ann["segmentation"]) > 1:
                        s = self.merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([img_w, img_h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([img_w, img_h])).reshape(-1).tolist()
                    s = [cls] + s
                    segments.append(s)
                    
                    # Write bbox
                    # file_object.write(f"{current_category} {x_centre:.6f} {y_centre:.6f} {w:.6f} {h:.6f}\n")

                    # Write image segmentation
                    
                for i in range(len(segments)):
                    line = (
                        *(segments[i]),
                    ) 
                    file_object.write(("%g " * len(line)).rstrip() % line + "\n")
                        
                if self.plot_yolo_masks:
                    self.plot_segmentation_contours(filename, segments)

            count += 1

    def convert(self):
        self.load_images_from_folder(self.input_path)
        self.coco_to_yolo()
        print(f"Processed {len(self.file_names)} files.")
        
    def min_index(self, arr1, arr2): # From the ultralytics converter.py source code: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py#L449
        """
        Find a pair of indexes with the shortest distance between two arrays of 2D points.

        Args:
            arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
            arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

        Returns:
            (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def merge_multi_segment(self, segments): # From the ultralytics converter.py source code: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py#L449
        """
        Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
        This function connects these coordinates with a thin line to merge all segments into one.

        Args:
            segments (List[List]): Original segmentations in COCO's JSON file.
                                Each element is a list of coordinates, like [segmentation1, segmentation2,...].

        Returns:
            s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list = [[] for _ in range(len(segments))]

        # Record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # Use two round to connect all the segments
        for k in range(2):
            # Forward connection
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # Middle segments have two indexes, reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # Deal with the first segment and the last one
                    if i in {0, len(idx_list) - 1}:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0] : idx[1] + 1])

            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in {0, len(idx_list) - 1}:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s
    
def convert_coco_to_yolo(dir_absolute_path, dataset_path, yolo_dataset_path, convert=True):
    if not convert:
        return

    json_file_path = os.path.join(dataset_path, 'train', '_annotations.coco.json')

    with open(json_file_path) as f:
        data_in = json.load(f)

    classes = [str(cat['name']) for cat in data_in['categories']]

    for mode in ['train', 'valid']:
        input_path = os.path.join(dataset_path, mode)
        output_path = os.path.join(yolo_dataset_path, mode)
        input_json_train = '_annotations.coco.json'
        converter = COCOToYOLOConverter(input_path, output_path, input_json_train, plot_yolo_masks=False)
        converter.convert()

        if mode == 'valid':  # train and valid folders successfully created
            yaml_path = os.path.join(os.path.dirname(output_path), 'data.yaml')
            yolo_data = {
                'names': classes,
                'nc': len(classes),
                'train': os.path.join(dir_absolute_path, os.path.dirname(output_path).replace(".", "").replace("//", "/"), 'train/images'),
                'val': os.path.join(dir_absolute_path, os.path.dirname(output_path).replace(".", "").replace("//", "/"), 'valid/images')
            }

            with open(yaml_path, 'w') as file:
                yaml.dump(yolo_data, file, default_flow_style=False)

            print(f"YAML file {yaml_path} created and saved.")